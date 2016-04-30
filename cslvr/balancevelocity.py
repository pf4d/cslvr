from fenics            import *
from dolfin_adjoint    import *
from cslvr.physics     import Physics
from cslvr.d2model     import D2Model
from cslvr.io          import print_text, print_min_max
import numpy as np
import sys


class BalanceVelocity(Physics):
  """
  Balance velocity solver.

  Class representing balance velocity physics.

  Use like this:

  >>> bv = BalanceVelocity(model, 5.0)
  ::: INITIALIZING VELOCITY-BALANCE PHYSICS :::
  >>> bv.solve()
  ::: solving BalanceVelocity :::
  ::: calculating surface gradient :::
  Process 0: Solving linear system of size 9034 x 9034 (PETSc Krylov solver).
  Process 0: Solving linear system of size 9034 x 9034 (PETSc Krylov solver).
  dSdx <min, max> : <-1.107e+00, 8.311e-01>
  dSdy <min, max> : <-7.928e-01, 1.424e+00>
  ::: solving for smoothed x-component of driving stress with kappa = 5.0 :::
  Process 0: Solving linear variational problem.
  Nx <min, max> : <-1.607e+05, 3.628e+05>
  ::: solving for smoothed y-component of driving stress :::
  Process 0: Solving linear variational problem.
  Ny <min, max> : <-2.394e+05, 2.504e+05>
  ::: calculating normalized velocity direction from driving stress :::
  d_x <min, max> : <-1.000e+00, 9.199e-01>
  d_y <min, max> : <-9.986e-01, 1.000e+00>
  ::: solving velocity balance magnitude :::
  Process 0: Solving linear variational problem.
  Ubar <min, max> : <-5.893e+03, 9.844e+03>
  ::: removing negative values of balance velocity :::
  Ubar <min, max> : <0.000e+00, 9.844e+03>


  Args:
    :model: a :class:`~d2model.D2Model` instance holding all pertinent 
            variables.

    :kappa: a floating-point value representing surface smoothing 
            radius in units of ice thickness :math:`H = S-B`.

  Returns:
    text printed to the screen.

  """ 
  
  def __init__(self, model, kappa=5.0):
    """
    balance velocity init.
    """ 
    s    = "::: INITIALIZING VELOCITY-BALANCE PHYSICS :::"
    print_text(s, cls=self)
    
    if type(model) != D2Model:
      s = ">>> BalanceVelocity REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    Q      = model.Q
    g      = model.g
    rho    = model.rhoi
    S      = model.S
    B      = model.B
    H      = S - B
    h      = model.h
    dSdx   = model.dSdx
    dSdy   = model.dSdy
    d_x    = model.d_x
    d_y    = model.d_y
    adot   = model.adot
        
    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :
    phi  = TestFunction(Q)
    Ubar = TrialFunction(Q)
    Nx   = TrialFunction(Q)
    Ny   = TrialFunction(Q)
    
    # calculate horizontally smoothed driving stress :
    a_dSdx = + Nx * phi * dx \
             + (kappa*H)**2 * (phi.dx(0)*Nx.dx(0) + phi.dx(1)*Nx.dx(1)) * dx
    L_dSdx = rho * g * H * dSdx * phi * dx \
    
    a_dSdy = + Ny * phi * dx \
             + (kappa*H)**2 * (phi.dx(0)*Ny.dx(0) + phi.dx(1)*Ny.dx(1)) * dx
    L_dSdy = rho * g * H * dSdy * phi*dx \
    
    # SUPG method :
    #if model.mesh.ufl_cell().topological_dimension() == 3:
    #  dS      = as_vector([d_x, d_y, 0.0])
    dS      = as_vector([d_x, d_y])
    phihat  = phi + h/(2*H) * ((H*dS[0]*phi).dx(0) + (H*dS[1]*phi).dx(1))
    #phihat  = phi + h/(2*H) * (H*dS[0]*phi.dx(0) + H*dS[1]*phi.dx(1))
    
    def L(u, uhat):
      #if model.mesh.ufl_cell().topological_dimension() == 3:
      #  return div(uhat)*u + dot(grad(u), uhat)
      #elif model.mesh.ufl_cell().topological_dimension() == 2:
      l = + (uhat[0].dx(0) + uhat[1].dx(1))*u \
          + u.dx(0)*uhat[0] + u.dx(1)*uhat[1]
      return l
    
    B = L(Ubar*H, dS) * phihat * dx
    a = adot * phihat * dx

    self.kappa  = kappa
    self.a_dSdx = a_dSdx
    self.a_dSdy = a_dSdy
    self.L_dSdx = L_dSdx
    self.L_dSdy = L_dSdy
    self.B      = B
    self.a      = a
  
  def solve(self, annotate=True):
    """
    Solve the balance velocity magnitude :math:`\Vert \\bar{\mathbf{u}} \Vert`.

    This will be completed in four steps,

    1. Calculate the surface gradient :math:`\\frac{\partial S}{\partial x}`
       and :math:`\\frac{\partial S}{\partial y}` saved to ``model.dSdx``
       and ``model.dSdy``.

    2. Solve for the smoothed component of driving stress

       .. math::
       
          \\tau_x = \\rho g H \\frac{\partial S}{\partial x}, \hspace{10mm}
          \\tau_y = \\rho g H \\frac{\partial S}{\partial y}
 
       saved respectively to ``model.Nx`` and ``model.Ny``. 
    
    3. Calculate the normalized flux directions
       
       .. math::
       
          d_x = -\\frac{\\tau_x}{\Vert \\tau_x \Vert}, \hspace{10mm}
          d_y = -\\frac{\\tau_y}{\Vert \\tau_y \Vert},
 
       saved respectively to ``model.d_x`` and ``model.d_y``. 
    
    4. Calculate the balance velocity magnitude 
       :math:`\Vert \\bar{\mathbf{u}} \Vert`
       from

       .. math::

          \\nabla \cdot \\left( \\bar{\mathbf{u}} H \\right) = \dot{a} - F_b

       saved to ``model.Ubar``.

    """
    model = self.model
    
    s    = "::: solving BalanceVelocity :::"
    print_text(s, cls=self)
    
    s    = "::: calculating surface gradient :::"
    print_text(s, cls=self)
    
    dSdx   = project(model.S.dx(0), model.Q, annotate=annotate)
    dSdy   = project(model.S.dx(1), model.Q, annotate=annotate)
    model.assign_variable(model.dSdx, dSdx, cls=self)
    model.assign_variable(model.dSdy, dSdy, cls=self)
    
    # update velocity direction from driving stress :
    s    = "::: solving for smoothed x-component of driving stress " + \
           "with kappa = %g :::" % self.kappa
    print_text(s, cls=self)
    solve(self.a_dSdx == self.L_dSdx, model.Nx, annotate=annotate)
    print_min_max(model.Nx, 'Nx', cls=self)
    
    s    = "::: solving for smoothed y-component of driving stress :::"
    print_text(s, cls=self)
    solve(self.a_dSdy == self.L_dSdy, model.Ny, annotate=annotate)
    print_min_max(model.Ny, 'Ny', cls=self)
    
    # normalize the direction vector :
    s    =   "::: calculating normalized flux direction" \
           + " from driving stress :::"
    print_text(s, cls=self)
    d_x_v = model.Nx.vector().array()
    d_y_v = model.Ny.vector().array()
    d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + 1e-16)
    model.assign_variable(model.d_x, -d_x_v / d_n_v, cls=self)
    model.assign_variable(model.d_y, -d_y_v / d_n_v, cls=self)
    
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



