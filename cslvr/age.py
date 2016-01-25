from cslvr.physics import Physics

class Age(Physics):
  r"""
  Class for calculating the age of the ice in steady state.

  :Very simple PDE:
     .. math::
      \vec{u} \cdot \nabla A = 1

  This equation, however, is numerically challenging due to its being 
  hyperbolic.  This is addressed by using a streamline upwind Petrov 
  Galerkin (SUPG) weighting.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
                  attributes such as velocties, age, and surface climate
  """

  def __init__(self, model, solve_params=None, transient=False,
               use_smb_for_ela=False, ela=None):
    """ 
    Set up the equations 
    """
    s    = "::: INITIALIZING AGE PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D3Model:
      s = ">>> Age REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    h = model.h
    
    Bub = FunctionSpace(model.mesh, "B", 4, constrained_domain=model.pBC)
    model.MQ  = model.Q + Bub

    # Trial and test
    a   = TrialFunction(model.MQ)
    phi = TestFunction(model.MQ)
    self.age = Function(model.MQ)

    # Steady state
    if not transient:
      s    = "    - using steady-state -"
      print_text(s, self.color())
      
      ## SUPG method :
      U      = as_vector([model.u, model.v, model.w])
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))
      
      # Residual 
      R = dot(U,grad(a)) - Constant(1.0)
      self.a = dot(U,grad(a)) * phi * dx
      self.L = Constant(1.0) * phi * dx

      # Weak form of residual
      self.F = R * phihat * dx

    else:
      s    = "    - using transient -"
      print_text(s, self.color())
      
      # Starting and midpoint quantities
      ahat   = model.ahat
      a0     = model.a0
      uhat   = model.uhat
      vhat   = model.vhat
      what   = model.what
      mhat   = model.mhat

      # Time step
      dt     = model.time_step

      # SUPG method (note subtraction of mesh velocity) :
      U      = as_vector([uhat, vhat, what-mhat])
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))

      # Midpoint value of age for Crank-Nicholson
      a_mid = 0.5*(a + self.ahat)
      
      # Weak form of time dependent residual
      self.F = + (a - a0)/dt * phi * dx \
               + dot(U, grad(a_mid)) * phihat * dx \
               - 1.0 * phihat * dx

    # form the boundary conditions :
    if use_smb_for_ela:
      s    = "    - using adot (SMB) boundary condition -"
      print_text(s, self.color())
      self.bc_age = DirichletBC(model.MQ, 0.0, model.ff_acc, 1)
    
    else:
      s    = "    - using ELA boundary condition -"
      print_text(s, self.color())
      def above_ela(x,on_boundary):
        return x[2] > ela and on_boundary
      self.bc_age = DirichletBC(model.Q, 0.0, above_ela)

  def solve(self, ahat=None, a0=None, uhat=None, what=None, vhat=None):
    """ 
    Solve the system
    
    :param ahat   : Observable estimate of the age
    :param a0     : Initial age of the ice
    :param uhat   : Horizontal velocity
    :param vhat   : Horizontal velocity perpendicular to :attr:`uhat`
    :param what   : Vertical velocity
    """
    model  = self.model

    # Assign values to midpoint quantities and mesh velocity
    if ahat:
      model.assign_variable(model.ahat, ahat)
      model.assign_variable(model.a0,   a0)
      model.assign_variable(model.uhat, uhat)
      model.assign_variable(model.vhat, vhat)
      model.assign_variable(model.what, what)

    # Solve!
    s    = "::: solving age :::"
    print_text(s, self.color())
    #solve(lhs(self.F) == rhs(self.F), model.age, self.bc_age)
    solve(self.a == self.L, self.age, self.bc_age)
    model.age.interpolate(self.age)
    print_min_max(model.age, 'age')
  
  
  def init_age(self):
    """ 
    Set up the equations 
    """
    s    = "::: INITIALIZING AGE PHYSICS :::"
    print_text(s, self.D3Model_color)

    config = self.config
    h      = self.h
    U      = self.U3
    
    #Bub = FunctionSpace(self.mesh, "B", 4, constrained_domain=self.pBC)
    self.MQ  = self.Q# + Bub

    # Trial and test
    a   = TrialFunction(self.MQ)
    phi = TestFunction(self.MQ)
    #self.age = Function(self.MQ)

    # Steady state
    if config['mode'] == 'steady':
      s    = "    - using steady-state -"
      print_text(s, self.D3Model_color)
      
      # SUPG method :
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))
      
      # Residual 
      self.age_F = (dot(U,grad(a)) - Constant(1.0)) * phihat * dx
      self.a_a   = dot(U,grad(a)) * phi * dx
      self.a_L   = Constant(1.0) * phi * dx

    else:
      s    = "    - using transient -"
      print_text(s, self.D3Model_color)
      
      # Starting and midpoint quantities
      ahat   = self.ahat
      a0     = self.a0
      uhat   = self.uhat
      vhat   = self.vhat
      what   = self.what
      mhat   = self.mhat

      # Time step
      dt     = config['time_step']

      # SUPG method (note subtraction of mesh velocity) :
      U      = as_vector([uhat, vhat, what-mhat])
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))

      # Midpoint value of age for Crank-Nicholson
      a_mid = 0.5*(a + self.ahat)
      
      # Weak form of time dependent residual
      self.age_F = + (a - a0)/dt * phi * dx \
                   + dot(U, grad(a_mid)) * phihat * dx \
                   - 1.0 * phihat * dx

    # form the boundary conditions :
    if config['age']['use_smb_for_ela']:
      s    = "    - using adot (SMB) boundary condition -"
      print_text(s, self.D3Model_color)
      self.age_bc = DirichletBC(self.MQ, 0.0, self.ff_acc, 1)
    
    else:
      s    = "    - using ELA boundary condition -"
      print_text(s, self.D3Model_color)
      def above_ela(x,on_boundary):
        return x[2] > config['age']['ela'] and on_boundary
      self.age_bc = DirichletBC(self.Q, 0.0, above_ela)

  def solve_age(self, ahat=None, a0=None, uhat=None, what=None, vhat=None):
    """ 
    Solve the system
    
    :param ahat   : Observable estimate of the age
    :param a0     : Initial age of the ice
    :param uhat   : Horizontal velocity
    :param vhat   : Horizontal velocity perpendicular to :attr:`uhat`
    :param what   : Vertical velocity
    """
    # Assign values to midpoint quantities and mesh velocity
    if ahat:
      self.assign_variable(self.ahat, ahat)
      self.assign_variable(self.a0,   a0)
      self.assign_variable(self.uhat, uhat)
      self.assign_variable(self.vhat, vhat)
      self.assign_variable(self.what, what)

    # Solve!
    s    = "::: solving age :::"
    print_text(s, self.D3Model_color)
    solve(lhs(self.age_F) == rhs(self.age_F), self.age, self.age_bc,
          annotate=False)
    #solve(self.a_a == self.a_L, self.age, self.age_bc, annotate=False)
    #self.age.interpolate(self.age)
    print_min_max(self.age, 'age')


class FirnAge(Physics):

  def __init__(self, model, solve_params=None):
    """
    """
    s    = "::: INITIALIZING FIRN AGE PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D1Model:
      s = ">>> FirnAge REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    Q       = model.Q
    w       = model.w                         # velocity
    w_1     = model.w0                        # previous step's velocity
    m       = model.m                         # mesh velocity
    m_1     = model.m0                        # previous mesh velocity
    a       = model.age                       # age
    a_1     = model.age0                      # previous step's age
    dt      = model.time_step                 # timestep
    
    da      = TrialFunction(Q)
    xi      = TestFunction(Q)
    
    model.assign_variable(a,   1.0)
    model.assign_variable(a_1, 1.0)

    # age residual :
    # theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
    #               0.5=Crank-Nicolson, 0=Forward-Euler) :
    # uses Taylor-Galerkin upwinding :
    theta   = 0.5 
    a_mid   = theta*a + (1-theta)*a_1
    f       = + (a - a_1)/dt * xi * dx \
              - 1 * xi * dx \
              + w * a_mid.dx(0) * xi * dx \
              - 0.5 * (w - w_1) * a_mid.dx(0) * xi * dx \
              + w**2 * dt/2 * inner(a_mid.dx(0), xi.dx(0)) * dx \
              - w * w.dx(0) * dt/2 * a_mid.dx(0) * xi * dx
    
    J       = derivative(f, a, da)
    
    self.ageBc = DirichletBC(Q, 0.0, model.surface)

    self.f = f
    self.J = J

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    params = {'solver' : {'relaxation_parameter'     : 1.0,
                           'maximum_iterations'      : 25,
                           'error_on_nonconvergence' : False,
                           'relative_tolerance'      : 1e-10,
                           'absolute_tolerance'      : 1e-10}}
    return params

  def solve(self):
    """
    """
    s    = "::: solving FirnAge :::"
    print_text(s, self.color())
    
    model  = self.model

    # solve for age :
    solve(self.f == 0, self.a, self.ageBc, J=self.J,
          solver_parameters=self.solve_params['solver'])
    model.age.interpolate(self.a)
    print_min_max(model.a, 'age')



