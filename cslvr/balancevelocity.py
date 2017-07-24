from fenics            import *
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
    dOmega_g = model.dOmega_g()
    dOmega_w = model.dOmega_w()
    dOmega   = model.dOmega()
    dGamma   = model.dGamma()

    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :
    if stabilization_method == 'BDM':
      Q                  = model.BDM
      Phi                = TestFunction(Q)
      U                  = TrialFunction(Q)
      self.U_s           = Function(Q)
      phi_x, phi_y, psi  = Phi
      j_x,   j_y,   ubar = U
      phi                = as_vector([phi_x,  phi_y ])
      j                  = as_vector([j_x,    j_y   ])
    elif stabilization_method == 'DG':
      Q     = model.DG1
      Phi   = TestFunction(model.DG1)
      ubar  = TrialFunction(model.DG1)
    else:
      Q     = model.Q
      Phi   = TestFunction(Q)
      ubar  = TrialFunction(Q)
      self.uv  = Function(model.Q)
      self.vv  = Function(model.Q)
      self.U_vert = as_vector([self.uv, self.vv])
      self.Uvnorm = Function(model.Q)
      self.Uvnorm.assign(Constant(1e16))
    self.Ubar = ubar

    kappa = Constant(kappa)
    gamma = Constant(0.0)#Constant(10000)

    # stabilization test space :
    Uhat    = as_vector([uhat, vhat])
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
    print_min_max(model.d_x, 'd_x')
    
    s    = "::: solving for smoothed y-component of flow direction " + \
           "with kappa = %g :::" % self.kappa
    print_text(s, cls=self)
    solve(a_dSdy == L_dSdy, model.d_y, annotate=annotate)
    print_min_max(model.d_y, 'd_y')
      
    # normalize the direction vector :
    s    =  r"::: normalizing flow direction :::"
    print_text(s, cls=self)
    d_x_v = model.d_x.vector().array()
    d_y_v = model.d_y.vector().array()
    d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + DOLFIN_EPS)
    model.init_uhat(d_x_v / d_n_v)
    model.init_vhat(d_y_v / d_n_v)
  
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
      solve(self.a == self.L, Ubar, annotate=annotate)
      Ubarn = project(Ubar, model.Q)
      model.Ubar.assign(Ubarn)
    else:
      solve(self.a == self.L, model.Ubar, annotate=annotate)
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

  #def form_obj_ftn(self, integral, kind='log'):
  #  """
  #  Forms and returns an objective functional for use with adjoint.
  #  Saves to ``self.J``.
  #  """
  #  self.obj_ftn_type = kind         # need to save this for printing values.
  #  self.integral     = integral     # this too.
  #  
  #  # note that even if self.reset() or self.linearize_viscosity() are called,
  #  # this will remain pointing to the unknown of interest :
  #  Ubar     = self.get_U()
  #  U_ob     = self.model.U_ob
  #  
  #  if type(integral) == list:
  #    self.bndrys = self.model.boundaries[integral[0]]
  #    dJ          = self.model.dx(integral[0])
  #    for i in integral[1:]:
  #      dJ          += self.model.dx(i)
  #      self.bndrys += ' and ' + self.model.boundaries[i]
  #  else:
  #    dJ          = self.model.dx(integral)
  #    self.bndrys = self.model.boundaries[integral]

  #  if kind == 'log':
  #    self.J  = 0.5 * ln( (Ubar + 0.01) / (U_ob + 0.01) )**2 * dJ 
  #  
  #  elif kind == 'l2':
  #    self.J  = 0.5 * (Ubar - U_ob)**2 * dJ
  #  
  #  elif kind == 'ratio':
  #    self.J  = 0.5 * (1 -  (Ubar + 0.01) / (U_ob + 0.01))**2 * dJ
  #  
  #  elif kind == 'abs': 
  #    self.J  = abs(Ubar - U_ob) * dJ

  #  else:
  #    s = ">>> ADJOINT OBJECTIVE FUNCTIONAL MAY BE 'l2', " + \
  #        "'log', 'ratio', OR 'abs', NOT '%s' <<<" % kind
  #    print_text(s, 'red', 1)
  #    sys.exit(1)

  #  s  = "::: forming '%s' objective functional integrated over %s:::"
  #  print_text(s % (kind, self.bndrys), self.color())

  #def calc_misfit(self):
  #  r"""
  #  Calculates and returns the misfit of model and observations, 

  #    D = \left\Vert \bar{u} - \Vert \underline{u}_{ob} \Vert_2 \right\Vert_{\infty}

  #  over shelves or grounded depending on the paramter ``integral`` sent to
  #  :func:`~balancevelocity.BalanceVelocity.form_obj_ftn`.
  #  """
  #  s  = "::: calculating misfit ||U - U_ob||_{oo} over %s :::"
  #  print_text(s % self.bndrys, cls=self)

  #  model    = self.model
  #  integral = self.integral

  #  # convert everything for low-level manipulations :
  #  Ubar_v   = model.Ubar.vector().array()
  #  U_ob_v   = model.U_ob.vector().array()

  #  # the magnitude of error :
  #  D_v  = abs(Ubar_v - U_ob_v)

  #  # assign to a function :
  #  D    = Function(model.Q)
  #  D.vector().set_local(D_v)

  #  # calculate L_inf vector norm :
  #  D      = MPI.max(mpi_comm_world(), D.vector().max())

  #  s    = "||U - U_ob||_{oo} : %.3E" % D
  #  print_text(s, '208', 1)
  #  return D
  #
  #def calc_functionals(self):
  #  """
  #  Used to facilitate printing the objective function in adjoint solves.
  #  """
  #  s   = "::: calculating functionals :::"
  #  print_text(s, cls=self)

  #  ftnls = []

  #  R = assemble(self.Rp, annotate=False)
  #  print_min_max(R, 'R')
  #  ftnls.append(R)
  #  
  #  J = assemble(self.J, annotate=False)
  #  print_min_max(J, 'J')
  #  ftnls.append(J)

  #  if self.obj_ftn_type == 'log_L2_hybrid':
  #    J1 = assemble(self.J1, annotate=False)
  #    print_min_max(J1, 'J1')
  #    ftnls.append(J1)
  #    
  #    J2 = assemble(self.J2, annotate=False)
  #    print_min_max(J2, 'J2')
  #    ftnls.append(J2)

  #  if self.reg_ftn_type == 'TV_Tik_hybrid':
  #    R1 = assemble(self.R1, annotate=False)
  #    print_min_max(R1, 'R1')
  #    ftnls.append(R1)
  #    
  #    R2 = assemble(self.R2, annotate=False)
  #    print_min_max(R2, 'R2')
  #    ftnls.append(R2)
  #  return ftnls 
  #  
  #def optimize_ubar(self, control, bounds,
  #                  method            = 'ipopt',
  #                  max_iter          = 100,
  #                  adj_save_vars     = None,
  #                  adj_callback      = None,
  #                  post_adj_callback = None):
  #  """
  #  """
  #  s    = "::: solving optimal control to minimize ||u - u_ob|| with " + \
  #         "control parameter%s :::"
  #  if type(control) != list:
  #    control = [control]
  #  tx = 's '
  #  for i in control:
  #    tx += "'" + i.name() + "'"
  #    if i != control[-1]: tx += ' and '
  #  print_text(s % tx, cls=self)

  #  model = self.model

  #  # reset entire dolfin-adjoint state :
  #  adj_reset()

  #  # starting time :
  #  t0   = time()

  #  # need this for the derivative callback :
  #  global counter
  #  counter = 0 
  #  
  #  # functional lists to be populated :
  #  global Rs, Js, Ds, J1s, J2s, R1s, R2s
  #  Rs     = []
  #  Js     = []
  #  Ds     = []
  #  if self.obj_ftn_type == 'log_L2_hybrid':
  #    J1s  = []
  #    J2s  = []
  #  if self.reg_ftn_type == 'TV_Tik_hybrid':
  #    R1s  = []
  #    R2s  = []
  # 
  #  # solve the momentum equations with annotation enabled :
  #  s    = '::: solving mass-conservation forward problem :::'
  #  print_text(s, cls=self)
  #  self.solve(annotate=True)
  #  
  #  # now solve the control optimization problem : 
  #  s    = "::: starting adjoint-control optimization with method '%s' :::"
  #  print_text(s % method, cls=self)

  #  # objective function callback function : 
  #  def eval_cb(I, c):
  #    s    = '::: adjoint objective eval post callback function :::'
  #    print_text(s, cls=self)
  #    print_min_max(I,    'I')
  #    for ci in c:
  #      print_min_max(ci,    'control: ' + ci.name())
  #  
  #  # objective gradient callback function :
  #  def deriv_cb(I, dI, c):
  #    global counter, Rs, Js, J1s, J2s
  #    if method == 'ipopt':
  #      s0    = '>>> '
  #      s1    = 'iteration %i (max %i) complete'
  #      s2    = ' <<<'
  #      text0 = get_text(s0, 'red', 1)
  #      text1 = get_text(s1 % (counter, max_iter), 'red')
  #      text2 = get_text(s2, 'red', 1)
  #      if MPI.rank(mpi_comm_world())==0:
  #        print text0 + text1 + text2
  #      counter += 1
  #    s    = '::: adjoint obj. gradient post callback function :::'
  #    print_text(s, cls=self)
  #    for (dIi,ci) in zip(dI,c):
  #      print_min_max(dIi,    'dI/control: ' + ci.name())
  #      self.model.save_xdmf(dIi, 'dI_control_' + ci.name())
  #      self.model.save_xdmf(ci, 'control_' + ci.name())
  #    
  #    # update the DA current velocity to the model for evaluation 
  #    # purposes only; the model.assign_variable function is 
  #    # annotated for purposes of linking physics models to the adjoint
  #    # process :
  #    u_opt = DolfinAdjointVariable(model.Ubar).tape_value()
  #    model.init_Ubar(u_opt)

  #    # print functional values :
  #    for i in range(len(control)):
  #      control[i].assign(c[i], annotate=False)
  #    ftnls = self.calc_functionals()
  #    D     = self.calc_misfit()

  #    # functional lists to be populated :
  #    Rs.append(ftnls[0])
  #    Js.append(ftnls[1])
  #    Ds.append(D)
  #    if self.obj_ftn_type == 'log_L2_hybrid':
  #      J1s.append(ftnls[2])
  #      J2s.append(ftnls[3])
  #    if self.reg_ftn_type == 'TV_Tik_hybrid':
  #      R1s.append(ftnls[4])
  #      R2s.append(ftnls[5])

  #    # call that callback, if you want :
  #    if adj_callback is not None:
  #      adj_callback(I, dI, c)
  # 
  #  # get the cost, regularization, and objective functionals :
  #  I = self.J + self.R
  #  
  #  # define the control parameter :
  #  m = []
  #  for i in control:
  #    m.append(Control(i, value=i))
  #  
  #  # create the reduced functional to minimize :
  #  F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
  #                        derivative_cb_post=deriv_cb)

  #  # optimize with scipy's fmin_l_bfgs_b :
  #  if method == 'l_bfgs_b': 
  #    out = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=bounds,
  #                   options={"disp"    : True,
  #                            "maxiter" : max_iter,
  #                            "gtol"    : 1e-5})
  #    b_opt = out
  #  
  #  # or optimize with IPOpt (preferred) :
  #  elif method == 'ipopt':
  #    try:
  #      import pyipopt
  #    except ImportError:
  #      info_red("""You do not have IPOPT and/or pyipopt installed.
  #                  When compiling IPOPT, make sure to link against HSL,
  #                  as it is a necessity for practical problems.""")
  #      raise
  #    problem = MinimizationProblem(F, bounds=bounds)
  #    parameters = {"tol"                : 1e-8,
  #                  "acceptable_tol"     : 1e-6,
  #                  "maximum_iterations" : max_iter,
  #                  "print_level"        : 5,
  #                  "ma97_order"         : "metis",
  #                  "linear_solver"      : "ma97"}
  #    solver = IPOPTSolver(problem, parameters=parameters)
  #    b_opt  = solver.solve()

  #  # make the optimal control parameter available :
  #  for c,b in zip(control, b_opt):
  #    model.assign_variable(c, b)
  #  #Control(control).update(b_opt)  # FIXME: does this work?
  #  
  #  # call the post-adjoint callback function if set :
  #  if post_adj_callback is not None:
  #    s    = '::: calling optimize_ubar() post-adjoined callback function :::'
  #    print_text(s, cls=self)
  #    post_adj_callback()
  #  
  #  # save state to unique hdf5 file :
  #  if isinstance(adj_save_vars, list):
  #    s    = '::: saving variables in list arg adj_save_vars :::'
  #    print_text(s, cls=self)
  #    out_file = model.out_dir + 'u_opt.h5'
  #    foutput  = HDF5File(mpi_comm_world(), out_file, 'w')
  #    
  #    for var in adj_save_vars:
  #      model.save_hdf5(var, f=foutput)
  #    
  #    foutput.close()

  #  # calculate total time to compute
  #  tf = time()
  #  s  = tf - t0
  #  m  = s / 60.0
  #  h  = m / 60.0
  #  s  = s % 60
  #  m  = m % 60
  #  text = "time to optimize ||u - u_ob||: %02d:%02d:%02d" % (h,m,s)
  #  print_text(text, 'red', 1)
  #  
  #  # save all the objective functional values with rudimentary plot : 
  #  d    = model.out_dir + 'objective_ftnls_history/'
  #  s    = '::: saving objective functionals to %s :::'
  #  print_text(s % d, cls=self)
  #  if model.MPI_rank==0:
  #    if not os.path.exists(d):
  #      os.makedirs(d)
  #    np.savetxt(d + 'time.txt', np.array([tf - t0]))
  #    np.savetxt(d + 'Rs.txt',   np.array(Rs))
  #    np.savetxt(d + 'Js.txt',   np.array(Js))
  #    np.savetxt(d + 'Ds.txt',   np.array(Ds))
  #    if self.obj_ftn_type == 'log_L2_hybrid':
  #      np.savetxt(d + 'J1s.txt',  np.array(J1s))
  #      np.savetxt(d + 'J2s.txt',  np.array(J2s))
  #    if self.reg_ftn_type == 'TV_Tik_hybrid':
  #      np.savetxt(d + 'R1s.txt',  np.array(R1s))
  #      np.savetxt(d + 'R2s.txt',  np.array(R2s))

  #    fig = plt.figure()
  #    ax  = fig.add_subplot(111)
  #    ax.set_yscale('log')
  #    ax.set_ylabel(r'$\mathscr{J}\left( \mathbf{u} \right)$')
  #    ax.set_xlabel(r'iteration')
  #    ax.plot(np.array(Js), 'r-', lw=2.0)
  #    plt.grid()
  #    plt.savefig(d + 'J.png', dpi=100)
  #    plt.close(fig)

  #    fig = plt.figure()
  #    ax  = fig.add_subplot(111)
  #    ax.set_yscale('log')
  #    ax.set_ylabel(r'$\mathscr{R}\left( \beta \right)$')
  #    ax.set_xlabel(r'iteration')
  #    ax.plot(np.array(Rs), 'r-', lw=2.0)
  #    plt.grid()
  #    plt.savefig(d + 'R.png', dpi=100)
  #    plt.close(fig)

  #    fig = plt.figure()
  #    ax  = fig.add_subplot(111)
  #    ax.set_yscale('log')
  #    ax.set_ylabel(r'$\mathscr{D}\left( \mathbf{u} \right)$')
  #    ax.set_xlabel(r'iteration')
  #    ax.plot(np.array(Ds), 'r-', lw=2.0)
  #    plt.grid()
  #    plt.savefig(d + 'D.png', dpi=100)
  #    plt.close(fig)
  #    
  #    if self.obj_ftn_type == 'log_L2_hybrid':

  #      fig = plt.figure()
  #      ax  = fig.add_subplot(111)
  #      #ax.set_yscale('log')
  #      ax.set_ylabel(r'$\mathscr{J}_1\left( \mathbf{u} \right)$')
  #      ax.set_xlabel(r'iteration')
  #      ax.plot(np.array(J1s), 'r-', lw=2.0)
  #      plt.grid()
  #      plt.savefig(d + 'J1.png', dpi=100)
  #      plt.close(fig)
 
  #      fig = plt.figure()
  #      ax  = fig.add_subplot(111)
  #      #ax.set_yscale('log')
  #      ax.set_ylabel(r'$\mathscr{J}_2\left( \mathbf{u} \right)$')
  #      ax.set_xlabel(r'iteration')
  #      ax.plot(np.array(J2s), 'r-', lw=2.0)
  #      plt.grid()
  #      plt.savefig(d + 'J2.png', dpi=100)
  #      plt.close(fig)
  #    
  #    if self.reg_ftn_type == 'TV_Tik_hybrid':

  #      fig = plt.figure()
  #      ax  = fig.add_subplot(111)
  #      #ax.set_yscale('log')
  #      ax.set_ylabel(r'$\mathscr{R}_{tik}\left( \beta \right)$')
  #      ax.set_xlabel(r'iteration')
  #      ax.plot(np.array(R1s), 'r-', lw=2.0)
  #      plt.grid()
  #      plt.savefig(d + 'R1.png', dpi=100)
  #      plt.close(fig)
 
  #      fig = plt.figure()
  #      ax  = fig.add_subplot(111)
  #      #ax.set_yscale('log')
  #      ax.set_ylabel(r'$\mathscr{R}_{TV}\left( \beta \right)$')
  #      ax.set_xlabel(r'iteration')
  #      ax.plot(np.array(R2s), 'r-', lw=2.0)
  #      plt.grid()
  #      plt.savefig(d + 'R2.png', dpi=100)
  #      plt.close(fig)






