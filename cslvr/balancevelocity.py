from fenics            import *
from dolfin_adjoint    import *
from cslvr.physics     import Physics
from cslvr.d2model     import D2Model
from cslvr.inputoutput import print_text, print_min_max
import numpy as np
import sys


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
    self.kappa  = kappa
    
    s    = "::: INITIALIZING VELOCITY-BALANCE PHYSICS :::"
    print_text(s, cls=self)
    
    if type(model) != D2Model:
      s = ">>> BalanceVelocity REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    Q      = model.Q
    S      = model.S
    B      = model.B
    H      = S - B
    h      = model.h
    N      = model.N
    uhat   = model.uhat
    vhat   = model.vhat
    adot   = model.adot
    Fb     = model.Fb
        
    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :
    phi   = TestFunction(Q)
    ubar  = TrialFunction(Q)
    kappa = Constant(kappa)
    
    # stabilization test space :
    Uhat     = as_vector([uhat, vhat])
    tau      = 1 / (2*H/h + div(H*Uhat))
    phihat   = phi + tau * dot(Uhat, grad(phi)) 
   
    # the left-hand side : 
    def L(u):      return u*H*div(Uhat) + dot(grad(u*H), Uhat)
    def L_star(u): return u*H*div(Uhat) - dot(grad(u*H), Uhat)
    def L_adv(u):  return dot(grad(u*H), Uhat)
   
    Nb = sqrt(B.dx(0)**2 + B.dx(1)**2 + 1) 
    Ns = sqrt(S.dx(0)**2 + S.dx(1)**2 + 1)
    f  = Ns*adot - Nb*Fb

    # use streamline-upwind/Petrov-Galerkin :
    if stabilization_method == 'SUPG':
      s      = "    - using streamline-upwind/Petrov-Galerkin stabilization -"
      self.B = + L(ubar) * phi * dx \
               + inner(L_adv(phi), tau*L(ubar)) * dx
      self.a = + f * phi * dx \
               + inner(L_adv(phi), tau*f) * dx

    # use Galerkin/least-squares
    elif stabilization_method == 'GLS':
      s      = "    - using Galerkin/least-squares stabilization -"
      self.B = + L(ubar) * phi * dx \
               + inner(L(phi), tau*L(ubar)) * dx
      self.a = + f * phi * dx \
               + inner(L(phi), tau*f) * dx

    # use subgrid-scale-model :
    elif stabilization_method == 'SSM':
      s      = "    - using subgrid-scale-model stabilization -"
      self.B = + L(ubar) * phi * dx \
               - inner(L_star(phi), tau*L(ubar)) * dx
      self.a = + f * phi * dx \
               - inner(L_star(phi), tau*f) * dx
    
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
    model = self.model
    Q     = model.Q
    S     = model.S
    B     = model.B
    H     = S - B
    N     = model.N
    phi   = TestFunction(Q)
    d_x   = TrialFunction(Q)
    d_y   = TrialFunction(Q)
    kappa = Constant(self.kappa)
    
    # horizontally smoothed direction of flow :
    a_dSdx = + d_x * phi * dx \
             + (kappa*H)**2 * dot(grad(phi), grad(d_x)) * dx \
             - (kappa*H)**2 * dot(grad(d_x), N) * phi * ds
    L_dSdx = d[0] * phi * dx
    
    a_dSdy = + d_y * phi * dx \
             + (kappa*H)**2 * dot(grad(phi), grad(d_y)) * dx \
             - (kappa*H)**2 * dot(grad(d_y), N) * phi * ds
    L_dSdy = d[1] * phi*dx
    
    # update velocity direction :
    s    = "::: solving for smoothed x-component of flow direction " + \
           "with kappa = %g :::" % self.kappa
    print_text(s, cls=self)
    solve(a_dSdx == L_dSdx, model.d_x, annotate=annotate)
    print_min_max(model.d_x, 'd_x', cls=self)
    
    s    = "::: solving for smoothed y-component of flow direction " + \
           "with kappa = %g :::" % self.kappa
    print_text(s, cls=self)
    solve(a_dSdy == L_dSdy, model.d_y, annotate=annotate)
    print_min_max(model.d_y, 'd_y', cls=self)
    
    # normalize the direction vector :
    s    =  r"::: calculating normalized flux direction from \nabla S:::"
    print_text(s, cls=self)
    d_x_v = model.d_x.vector().array()
    d_y_v = model.d_y.vector().array()
    d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + 1e-16)
    model.assign_variable(model.uhat, d_x_v / d_n_v, cls=self)
    model.assign_variable(model.vhat, d_y_v / d_n_v, cls=self)
  
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
    solve(self.B == self.a, model.Ubar, annotate=annotate)
    print_min_max(model.Ubar, 'Ubar', cls=self)
    
    # enforce positivity of balance-velocity :
    s    = "::: removing negative values of balance velocity :::"
    print_text(s, cls=self)
    Ubar_v = model.Ubar.vector().array()
    Ubar_v[Ubar_v < 0] = 0
    model.assign_variable(model.Ubar, Ubar_v, cls=self)



