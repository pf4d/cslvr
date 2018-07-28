from dolfin            import *
from dolfin_adjoint    import *
from cslvr.physics     import Physics
from cslvr.d2model     import D2Model
from cslvr.inputoutput import get_text, print_text, print_min_max
import numpy               as np
import matplotlib.pyplot   as plt
import sys
import os


class BalanceVelocity(Physics):
  r"""
  Class representing balance velocity physics.

  :model: a :class:`~d2model.D2Model` instance holding all pertinent 
          variables, saved to ``self.model``.

  :kappa: a floating-point value representing direction smoothing 
          radius in units of ice thickness :math:`H = S-B`, where
          :math:`H` is given by ``self.model.H``, surface height
          :math:`S` is given by ``self.model.S`` and bed height
          :math:`B` is given by ``self.model.B``.

  Use like this:
 
  >>> d = (-model.S.dx(0), -model.S.dx(1))
  >>> bv = BalanceVelocity(model, kappa=5.0, stabilization_method='SUPG')
  ::: INITIALIZING VELOCITY-BALANCE PHYSICS :::
      - using streamline-upwind/Petrov-Galerkin stabilization -
  >>> bv.solve_direction_of_flow(d)
  ::: solving for smoothed x-component of flow direction with kappa = 0 :::
  Solving linear variational problem.
  d_x <min, max> : <-1.127e+04, 9.275e+03>
  ::: solving for smoothed y-component of flow direction with kappa = 0 :::
  Solving linear variational problem.
  d_y <min, max> : <-7.880e+03, 5.821e+03>
  ::: calculating normalized flux direction from \nabla S:::
  uhat <min, max> : <-1.000e+00, 1.000e+00>
  vhat <min, max> : <-1.000e+00, 1.000e+00>
  >>> bv.solve()
  ::: solving velocity balance magnitude :::
  Solving linear variational problem.
  Ubar <min, max> : <-8.388e+10, 4.470e+10>
  ::: removing negative values of balance velocity :::
  Ubar <min, max> : <0.000e+00, 4.470e+10>
  """ 
  
  def __init__(self, model, kappa=5.0, stabilization_method='SUPG'):
    """
    balance velocity init.
    """ 
    self.kappa                = kappa
    self.stabilization_method = stabilization_method
    
    s    = "::: INITIALIZING VELOCITY-BALANCE PHYSICS :::"
    print_text(s, cls=self)
    
    if type(model) != D2Model:
      s = ">>> BalanceVelocity REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    S        = model.S
    B        = model.B
    H        = S - B
    h        = model.h
    N        = model.N
    uhat     = model.uhat
    vhat     = model.vhat
    adot     = model.adot
    ubarm    = model.Ubar
    Fb       = model.Fb
    dOmega   = model.dOmega()
    dGamma   = model.dGamma()
      

    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :
    if stabilization_method == 'BDM':
      # system unknown function space is created now if periodic boundaries 
      # are not used (see model.generate_function_space()) :
      if model.use_periodic:
        Q    = model.BDM
      else:
        Q    = FunctionSpace(model.mesh, model.BDMMe)
      Phi                = TestFunction(Q)
      U                  = TrialFunction(Q)
      self.U_s           = Function(Q)
      phi_x, phi_y, psi  = Phi
      j_x,   j_y,   ubar = U
      phi                = as_vector([phi_x,  phi_y ])
      j                  = as_vector([j_x,    j_y   ])
    elif stabilization_method == 'DG':
      # system unknown function space is created now if periodic boundaries 
      # are not used (see model.generate_function_space()) :
      if model.use_periodic:
        Q    = model.DG1
      else:
        Q    = FunctionSpace(model.mesh, model.DG1e)
      Phi   = TestFunction(model.DG1)
      ubar  = TrialFunction(model.DG1)
    else:
      Q           = model.Q
      Phi         = TestFunction(Q)
      ubar        = TrialFunction(Q)
      self.uv     = Function(model.Q)
      self.vv     = Function(model.Q)
      self.U_vert = as_vector([self.uv, self.vv])
      self.Uvnorm = Function(model.Q)
      self.Uvnorm.assign(Constant(1e16))
    self.Ubar = ubar

    kappa = Constant(kappa)
    gamma = Constant(0.0)#Constant(10000)

    # stabilization test space :
    if   len(grad(Phi)) == 3:    Uhat = as_vector([uhat, vhat, 0])
    elif len(grad(Phi)) == 2:    Uhat = as_vector([uhat, vhat])
    Ut      = H*Uhat
    Unorm   = sqrt(dot(Ut, Ut) + DOLFIN_EPS)
    PE      = Unorm*h/(2*gamma)

    R        = abs(div(H*Uhat))
    #mk       = 1/3.
    #pe_1     = 2 * gamma/(mk*R*h**2)
    #pe_2     = mk*H/h
    #tau      = h**2 / (R*h**2 + 2*gamma/mk*pe_2)
    #xi       = 1/tanh(PE) - 1/PE
    #tau      = h*xi / (2 * Unorm)
    tau      = 1 / (4*gamma/h**2 + 2*H/h + R)
    phihat   = Phi + tau * dot(Uhat, grad(Phi)) 

    # the Peclet number : 
    #tnorm    = sqrt(dot(grad(ubarm), grad(ubarm)) + DOLFIN_EPS)
    #U_vert   = H*dot(Uhat, grad(ubarm)) * grad(ubarm) / tnorm 
    #Uvnorm   = sqrt(dot(U_vert, U_vert) + DOLFIN_EPS)
    #tau_2    = h / (2 * Uvnorm)
    ##tau_2    = dot(Uhat, grad(ubarm)) * tau
    tau_2    = dot(Uhat, grad(ubarm)) *  h / (2 * self.Uvnorm)
      
    # the linear-differential operator : 
    #def L(u):      return div(u*H*Uhat)
    #def L(u):      return + u*H*div(Uhat) \
    #                      + dot(grad(u*H), Uhat)
    def L_adv(u,H):  return + H*dot(Uhat, grad(u))
    def L_dis(u):    return + dot(self.U_vert, grad(u))
    def L(u,H):      return + u*H*div(Uhat) \
                            + u*dot(grad(H), Uhat) \
                            + L_adv(u,H)
    def L_star(u,H): return + u*H*div(Uhat) \
                            + u*dot(grad(H), Uhat) \
                            - L_adv(u,H)

    # inverse permeability tensor :
    def B_inv():
      B_inv_xx = 1 / (H*uhat + DOLFIN_EPS)
      B_inv_yy = 1 / (H*vhat + DOLFIN_EPS)
      B_v = as_matrix([[B_inv_xx, 0.0     ],
                       [0.0,      B_inv_yy]])
      return B_v
   
    Nb = sqrt(B.dx(0)**2 + B.dx(1)**2 + 1) 
    Ns = sqrt(S.dx(0)**2 + S.dx(1)**2 + 1)
    f  = Ns*adot - Nb*Fb

    if stabilization_method == 'BDM':
      s      = "    - using Brezzi-Douglas-Marini elements -"
      self.a = + dot(j, B_inv()*phi) * dOmega \
               + ubar * div(phi) * dOmega \
               + psi * div(j) * dOmega
      self.L = f*psi*dOmega

    elif stabilization_method == 'DG':
      s      = "    - using discontinuous-Galerkin elements -"
      u      = H*Uhat
      un     = 0.5*(dot(u, N) + abs(dot(u, N)))
      h_avg  = (h('+') + h('-'))/2
      alpha  = Constant(5.0)
      #self.a = + ubar*H * div(Uhat) * Phi * dOmega \
      #         - ubar*H * dot(Uhat, grad(Phi)) * dOmega \
      #         + gamma * dot(grad(ubar), grad(Phi)) * dOmega \
      #         + dot(jump(Phi), jump(un*ubar))*dS \
      #         + gamma*(alpha/h('+'))*dot(jump(Phi, N), jump(ubar, N))*dS \
      #         - gamma*dot(avg(grad(Phi)), jump(ubar, N))*dS \
      #         - gamma*dot(jump(Phi, N), avg(grad(ubar)))*dS \
      #         + dot(Phi, un*ubar)*dGamma
      self.a = + ubar*H*div(Uhat) * Phi * dOmega \
               + ubar*dot(grad(H), Uhat) * Phi * dOmega  \
               - ubar*H * dot(Uhat, grad(Phi)) * dOmega  \
               + dot(jump(Phi), jump(un*ubar))*dS \
               + dot(Phi, un*ubar)*dGamma
      self.L = + f * Phi * dOmega

    # use streamline-upwind/Petrov-Galerkin :
    elif stabilization_method == 'SUPG':
      s      = "    - using streamline-upwind/Petrov-Galerkin stabilization -"
      self.a = + L(ubar,H) * Phi * dOmega \
               + inner(L_adv(Phi,H), tau*L(ubar,H)) * dOmega \
               + inner(L_dis(Phi), tau_2*L(ubar,H)) * dOmega \
               #+ gamma * dot(grad(ubar), grad(Phi)) * dOmega \
      self.L = + f * Phi * dOmega \
               + inner(L_adv(Phi,H), tau*f) * dOmega \
               + inner(L_dis(Phi), tau_2*f) * dOmega \

    # use Galerkin/least-squares
    elif stabilization_method == 'GLS':
      s      = "    - using Galerkin/least-squares stabilization -"
      self.a = + L(ubar,H) * Phi * dOmega \
               + inner(L(Phi,H), tau*L(ubar,H)) * dOmega
      self.L = + f * Phi * dOmega \
               + inner(L(Phi,H), tau*f) * dOmega

    # use subgrid-scale-model :
    elif stabilization_method == 'SSM':
      s      = "    - using subgrid-scale-model stabilization -"
      self.a = + L(ubar,H) * Phi * dOmega \
               - inner(L_star(Phi,H), tau*L(ubar,H)) * dOmega
      self.L = + f * Phi * dOmega \
               - inner(L_star(Phi,H), tau*f) * dOmega
    
    print_text(s, cls=self)

  def solve_direction_of_flow(self, d, annotate=False):
    r"""
    Solve for the direction of flow, attained in two steps :

    :param d: a 2D vector of velocity direction from data 
              :math:`\mathbf{d}^{\text{data}}`.

    Solve for the unit-vector direction of flow :math:`\mathbf{d}`
    in two parts :

    1. Solve for the smoothed components of :math:`\mathbf{d}^{\text{data}}` :

       .. math::
       
          \mathbf{d} = \big( \kappa H \big)^2 \nabla \cdot \big( \nabla \mathbf{d} \big) + \mathbf{d}^{\text{data}},
 
       for components :math:`d_x` and :math:`d_y` saved respectively 
       to ``self.model.d_x`` and ``self.model.d_y``. 
    
    2. Calculate the normalized balance velocity direction :
       
       .. math::
       
          \hat{u} = \frac{d_x}{\Vert \mathbf{d} \Vert}, \hspace{10mm}
          \hat{v} = \frac{d_y}{\Vert \mathbf{d} \Vert},
 
       saved respectively to ``self.model.uhat`` and ``self.model.vhat``. 
    """
    model  = self.model
    Q      = model.Q
    S      = model.S
    B      = model.B
    H      = S - B
    N      = model.N
    phi    = TestFunction(Q)
    d_x    = TrialFunction(Q)
    d_y    = TrialFunction(Q)
    kappa  = Constant(self.kappa)
    dOmega = model.dOmega()
    dGamma = model.dGamma()

    # horizontally smoothed direction of flow :
    a_dSdx = + d_x * phi * dOmega \
             + (kappa*H)**2 * dot(grad(phi), grad(d_x)) * dOmega \
             - (kappa*H)**2 * dot(grad(d_x), N) * phi * dGamma
    L_dSdx = d[0] * phi * dOmega
    
    a_dSdy = + d_y * phi * dOmega \
             + (kappa*H)**2 * dot(grad(phi), grad(d_y)) * dOmega \
             - (kappa*H)**2 * dot(grad(d_y), N) * phi * dGamma
    L_dSdy = d[1] * phi*dOmega

    # update velocity direction :
    s    = "::: solving for smoothed x-component of flow direction " + \
           "with kappa = %g :::" % self.kappa
    print_text(s, cls=self)
    solve(a_dSdx == L_dSdx, model.d_x, annotate=annotate)
          #solver_parameters=self.linear_solve_params(), annotate=annotate)
    print_min_max(model.d_x, 'd_x')
    
    s    = "::: solving for smoothed y-component of flow direction " + \
           "with kappa = %g :::" % self.kappa
    print_text(s, cls=self)
    solve(a_dSdy == L_dSdy, model.d_y, annotate=annotate)
          #solver_parameters=self.linear_solve_params(), annotate=annotate)
    print_min_max(model.d_y, 'd_y')
      
    # normalize the direction vector :
    s    =  r"::: normalizing flow direction :::"
    print_text(s, cls=self)
    d_x_v = model.d_x.vector().get_local()
    d_y_v = model.d_y.vector().get_local()
    d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + DOLFIN_EPS)
    model.init_uhat(d_x_v / d_n_v)
    model.init_vhat(d_y_v / d_n_v)

  def linear_solve_params(self):
    """
    """
    params = {"linear_solver"  : "cg",
              "preconditioner" : "hypre_amg"}
    return params
  
  def solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    params = {'newton_solver' : {'linear_solver'            : 'mumps',
                                 'preconditioner'           : 'none',
                                 'relative_tolerance'       : 1e-13,
                                 'relaxation_parameter'     : 1.0,
                                 'maximum_iterations'       : 12,
                                 'error_on_nonconvergence'  : False}}
    return params

  def shock_capture(self):
    """
    """
    model       = self.model
    Uvn_a       = self.Uvnorm.vector().array()
    gub         = project(grad(model.Ubar), model.V)
    gu, tnorm_a = model.get_norm(gub, 'l2')
    H_a         = (model.S.vector().array() - model.B.vector().array())
    uhat_a      = model.uhat.vector().array()
    vhat_a      = model.vhat.vector().array()

    uv_a     = H_a * (uhat_a*gu[0] + vhat_a*gu[1]) * gu[0] / tnorm_a
    vv_a     = H_a * (uhat_a*gu[0] + vhat_a*gu[1]) * gu[1] / tnorm_a
    Uvnorm_a = np.sqrt(uv_a**2 + vv_a**2 + 1e-8)
    model.assign_variable(self.uv, uv_a)
    model.assign_variable(self.vv, vv_a)
    model.assign_variable(self.Uvnorm, Uvnorm_a)

  def solve(self, annotate=False):
    r"""
    Solve the balance velocity magnitude 
    :math:`\Vert \bar{\mathbf{u}} \Vert = \bar{u}` from

    .. math::

       \nabla \cdot \left( \bar{\mathbf{u}} H \right) = f,

    saved to ``self.model.Ubar``.

    """
    model = self.model
    
    # calculate balance-velocity :
    s    = "::: solving velocity balance magnitude :::"
    print_text(s, cls=self)
    
    params = {"linear_solver"  : "mumps"}
    
    if self.stabilization_method == 'BDM':
      solve(self.a == self.L, self.U_s,
            solver_parameters = self.solve_params(),
            annotate=annotate)
      j_xn, j_yn, ubarn = self.U_s
      jn = as_vector([j_xn, j_yn])
      print_min_max(ubarn, 'ubarn')
      ubarn    = project(ubarn, model.Q, annotate=annotate)
      ubarn    = project(sqrt(dot(grad(ubarn), grad(ubarn)) + DOLFIN_EPS),
                         model.Q, annotate=annotate)
      j_xn = project(j_xn, model.Q)
      j_yn = project(j_yn, model.Q)
      #ubarn = project(sqrt(dot(grad(ubarn), grad(ubarn))), model.Q)
      model.init_Ubar(ubarn)
      model.u.assign(j_xn)
      model.v.assign(j_yn)
    elif self.stabilization_method == 'DG':
      Ubar = Function(model.DG1)
      solve(self.a == self.L, Ubar,
            solver_parameters=params, annotate=annotate)
      Ubarn = project(Ubar, model.Q)
      model.Ubar.assign(Ubarn)
    else:
      solve(self.a == self.L, model.Ubar,
            solver_parameters=params, annotate=annotate)
    print_min_max(model.Ubar, 'Ubar')
    
    ## enforce positivity of balance-velocity :
    #s    = "::: removing negative values of balance velocity :::"
    #print_text(s, cls=self)
    #Ubar_v = model.Ubar.vector().array()
    #Ubar_v[Ubar_v < 0] = 0
    #model.assign_variable(model.Ubar, Ubar_v)

  def get_U(self):
    """
    """
    return self.model.Ubar



