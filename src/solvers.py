from pylab          import *
from dolfin         import *
from physics        import *
from scipy.optimize import fmin_l_bfgs_b

class SteadySolver(object):
  """
  This class solves for velocity, enthalpy (temperature), surface mass balance, 
  and ice age in steady state. The coupling between velocity and enthalpy 
  is performed via a Picard iteration.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
	                attributes such as velocties, age, and surface climate
  """
  def __init__(self, model, config):
    """
    Initialize solver.  Initialize all of the physics classes specified 
    as 'on' in the config object.
    """
    self.model          = model
    self.config         = config
    self.config['mode'] = 'steady'

    # velocity model :
    if self.config['velocity']['on']:
      
      if   config['velocity']['approximation'] == 'fo':
        self.velocity_instance = VelocityBP(model, config)
      
      elif config['velocity']['approximation'] == 'stokes':
        self.velocity_instance = VelocityStokes(model, config)
      
      else:
        print "Please use 'fo' or 'stokes'. "
    
    # enthalpy model :
    if config['enthalpy']['on']:
      self.enthalpy_instance = Enthalpy(model, config)

    # age model :
    if config['age']['on']:
      self.age_instance = Age(model, config)
    
    # surface climate model :
    if config['surface_climate']['on']:
      self.surface_climate_instance = SurfaceClimate(model, config)

  def solve(self):
    """ 
    Solve the problem using a Picard iteration, evaluating the velocity,
    enthalpy, surface mass balance, temperature boundary condition, and
    the age equation.
    """
    model  = self.model
    config = self.config
    T0     = config['velocity']['T0']
    
    # Set the initial Picard iteration (PI) parameters
    # L_\infty norm in velocity between iterations
    inner_error = inf             
   
    # number of iterations      
    counter     = 0                    
   
    # previous velocity for norm calculation
    u_prev      = zeros(len(model.u.vector().array()))
    
    # set an inner tolerance for PI
    inner_tol   = config['coupled']['inner_tol']   
    max_iter    = config['coupled']['max_iter']
    
    # Initialize a temperature field for visc. calc.
    if config['velocity']['use_T0']:
      model.T.vector().set_local( T0 * ones(len(model.T.vector().array())) )  

    # Perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance
    while inner_error > inner_tol and counter < max_iter:

      # Solve surface mass balance and temperature boundary condition
      if config['surface_climate']['on']:
        self.surface_climate_instance.solve()

      # Solve velocity
      if config['velocity']['on']:
        #self.model.u.vector()[:] = 0.0
        self.velocity_instance.solve()
        uMin = model.u.vector().min()
        uMax = model.u.vector().max()
        print 'u <min, max> : <%f, %f>' % (uMin, uMax)

      # Solve enthalpy (temperature, water content)
      if config['enthalpy']['on']:
        self.enthalpy_instance.solve()
        Tmin = model.T.vector().min()
        Tmax = model.T.vector().max()
        print 'T <min, max> : <%f, %f>' % (Tmin, Tmax)

      # Calculate L_\infty norm
      if config['coupled']['on']:
        diff        = (u_prev - model.u.vector().array())
        inner_error = diff.max()
        u_prev      = model.u.vector().array()
        counter    += 1
        print 'inner error :', inner_error
      
      else:
        inner_error = 0.0

    # Solve age equation
    if config['age']['on']:
      self.age_instance.solve()

    if config['log']:
      outpath = config['output_path']
      U       = project(as_vector([model.u, model.v, model.w]))
      File(outpath + 'U' + '.pvd') << U
      File(outpath + 'T' + '.pvd') << model.T

class TransientSolver(object):
  """
  This class solves for velocity and surface mass balance in steady state
  at each time step, and solves enthalpy (temperature), the free surface, 
  and ice age fully dynamically (all the hyperbolic ones are dynamic, that is).

  The class assumes that the model already has suitable initial conditions.
  It is recommended that a 'spin-up' is performed by running the steady model
  for all the variables you intend to solve for dynamically (especially 
  enthalpy.  
  
  You can get away with having an age with all 0 initial values, but 0 
  enthalpy is really cold, and the ice won't move.)

  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
	                attributes such as velocties, age, and surface climate
  
  """
  def __init__(self, model, config):
    """
    Initialize solver.  Initialize all of the physics classes specified 
    as 'on' in the config object.
    
    """
    self.model          = model
    self.config         = config
    self.config['mode'] = 'transient'

    # initialize velocity solver :
    if self.config['velocity']['on']:
      
      if   self.config['velocity']['approximation'] == 'fo':
        self.velocity_instance = VelocityBP(model, config)
      
      elif self.config['velocity']['approximation'] == 'stokes':
        self.velocity_instance = VelocityStokes(model, config)
      
      else:
        print "Please choose 'fo' or 'stokes'. "
    
    # initialized enthalpy solver : 
    if self.config['enthalpy']['on']:
      self.enthalpy_instance = Enthalpy(model, config)

    # initialize age solver :
    if self.config['age']['on']:
      self.age_instance = Age(model, config)

    # initialize surface climate solver :
    if self.config['surface_climate']['on']:
      self.surface_climate_instance = SurfaceClimate(model, config)

    # initialize free surface solver :
    if config['free_surface']['on']:
      self.surface_instance = FreeSurface(model, config)
      self.M_prev           = 1.0

    # Set up files for logging time dependent solutions to paraview files.
    if config['log']:
      self.file_u  = File(self.config['output_path']+'U.pvd')
      self.file_T  = File(self.config['output_path']+'T.pvd')
      self.file_S  = File(self.config['output_path']+'S.pvd')
      self.dheight = []
      self.mass    = []
      self.t_log   = []

    self.step_time = []
    self.M_prev    = 1.0

  def rhs_func_explicit(self, t, y, *f_args):
    """
    This function calculates the change in height of the surface of the
    ice sheet.
    
    :param t : Time
    :param y : Current height of the ice sheet
    :rtype   : Array containing rate of change of the ice surface values
    """
    model             = self.model
    config            = self.config
    thklim            = config['free_surface']['thklim']
    B                 = model.B.compute_vertex_values()
    y[(y-B) < thklim] = thklim + B[(y-B) < thklim]
    if config['periodic_boundary_conditions']:
      v2d = model.Q_non_periodic.dofmap().vertex_to_dof_map(model.flat_mesh)
    else:
      v2d = model.Q.dofmap().vertex_to_dof_map(model.flat_mesh)
    model.S.vector().set_local(y[v2d])
   
    if config['velocity']['on']:
      self.velocity_instance.solve()

    if config['surface_climate']['on']:
      self.surface_climate_instance.solve()
   
    if config['free_surface']['on']:
      self.surface_instance.solve(model.u, model.v, model.w, 
                                         model.S, model.smb)
 
    return model.dSdt.compute_vertex_values()

  def solve(self):
    """
    Performs the physics, evaluating and updating the enthalpy and age as 
    well as storing the velocity, temperature, and the age in vtk files.

    """
    model  = self.model
    config = self.config
    
    t      = config['t_start']
    t_end  = config['t_end']
    dt     = config['time_step']
    thklim = config['free_surface']['thklim']
   
    mesh   = model.mesh 
    smb    = model.smb
    sigma  = model.sigma

    H      = model.H
    u      = model.u
    v      = model.v
    w      = model.w
    mhat   = model.mhat
    A      = model.A
    T      = model.T
    S      = model.S
    B      = model.B
 

    smb.interpolate(config['free_surface']['observed_smb'])

    import time
    if config['periodic_boundary_conditions']:
      v2d = model.Q_non_periodic.dofmap().vertex_to_dof_map(model.mesh)
      mhat_non = Function(model.Q_non_periodic)
    else:
      v2d = model.Q.dofmap().vertex_to_dof_map(model.mesh)

    # Loop over all times
    while t <= t_end:

      B_a = B.compute_vertex_values()
      S_v = S.compute_vertex_values()
      
      tic = time.time()

      S_0 = S_v
      f_0 = self.rhs_func_explicit(t, S_0)
      S_1 = S_0 + dt*f_0
      S_1[(S_1-B_a) < thklim] = thklim + B_a[(S_1-B_a) < thklim]
      S.vector().set_local(S_1[v2d])

      f_1                     = self.rhs_func_explicit(t, S_1)
      S_2                     = 0.5*S_0 + 0.5*S_1 + 0.5*dt*f_1
      S_2[(S_2-B_a) < thklim] = thklim + B_a[(S_2-B_a) < thklim] 
      S.vector().set_local(S_2[v2d])
      
      mesh.coordinates()[:, 2]  = sigma.compute_vertex_values()*(S_2 - B_a) + B_a
      if config['periodic_boundary_conditions']:
        mhat_non.vector().set_local((S_2[v2d] - S_0[v2d])/dt * sigma.vector().get_local())
        m_temp = project(mhat_non,model.Q)
        mhat.vector().set_local(m_temp.vector().get_local())
      else:
        mhat.vector().set_local((S_2[v2d] - S_0[v2d])/dt * sigma.vector().get_local())
      # Calculate enthalpy update
      if self.config['enthalpy']['on']:
        self.enthalpy_instance.solve(H0=H, Hhat=H, uhat=u, vhat=v, what=w, 
                                     mhat=mhat)

      # Calculate age update
      if self.config['age']['on']:
        self.age_instance.solve(A0=A, Ahat=A, uhat=u, vhat=v, what=w, mhat=mhat)

      # Store velocity, temperature, and age to vtk files
      if self.config['log']:
        U = project(as_vector([u, v, w]))
        self.file_u << (U, t)
        self.file_T << (T, t)
        self.file_S << (S, t)
        self.t_log.append(t)
        M = assemble(self.surface_instance.M)
        self.mass.append(M)

      # Increment time step
      if MPI.process_number()==0:
        string = 'Time: {0}, CPU time for last time step: {1}, Mass: {2}'
        print string.format(t, time.time()-tic, M/self.M_prev)

      self.M_prev = M
      t          += dt
      self.step_time.append(time.time() - tic)

class AdjointSolver(object):
  """
  This class minimizes the misfit between an observed surface velocity and 
  the modelled surface velocity by changing the value of the basal traction
  coefficient.  The optimization is performed by calculating the gradient 
  of the objective function by using an incomplete adjoint (the adjoint 
  of the linearized forward model).  Minimization is accomplished with the 
  quasi-Newton BFGS algorithm
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
	                attributes such as velocties, age, and surface climate
  """

  def __init__(self, model, config):
    """
    Initialize some stuff.
    """
    self.model  = model
    self.config = config
    self.config['mode'] = 'steady'

    # initialize instances of the forward model, and the adjoint physics : 
    self.forward_model    = SteadySolver(model, config)
    self.adjoint_instance = AdjointVelocityBP(model, config)

  def set_target_velocity(self, u=None, v=None, U=None):
    """ 
    Set target velocity.

    Accepts a list of surface velocity data, and generates a dolfin
    expression from these.  Then projects this onto the velocity 
    function space.  The sum square error between this velocity 
    and modelled surface velocity is the objective function.
    
    :param u : Surface velocity
    :param v : Surface velocity perpendicular to :attr:`u`
    :param U : 2-D surface velocity data
    
    """
    model = self.model
    S     = model.S
    Q     = model.Q
    
    if u != None and v != None:
      model.u_o = project(u, Q)
      model.v_o = project(v, Q)

    elif U != None:
      Smag = project( sqrt(S.dx(0)**2 + S.dx(1)**2 + 1e-10), Q )
     
      model.U_o.interpolate(U)
      model.U_o.update()

      model.u_o = project( -model.U_o * S.dx(0) / Smag, Q )
      model.v_o = project( -model.U_o * S.dx(1) / Smag, Q )      

      model.u_o.update()
      model.v_o.update()
      
  def solve(self):
    r""" 
    Perform the optimization.

    First, we define functions that return the objective function and Jacobian.
    These are passed to scipy's fmin_l_bfgs_b, which is a python wrapper for the
    Fortran code of Nocedal et. al.

    The functions are needed to make the calculation of the search direction 
    and update of search point take place globally, across all proccessors, 
    rather than on a per-processor basis.

    We also specify bounds:
      
    :Condition:
       .. math::
        \beta_{2} > 0
    """
    model  = self.model
    config = self.config
    
    def get_global(m):
      """
      Takes a distributed object and returns a numpy array that
      contains all global values.
      """
      if type(m) == float:
        return array(m)
     
      # return a numPy array of values or single value of Constant :
      if type(m) == Constant:
        a = p = zeros(m.value_size())
        m.eval(a, p)
        return a
     
      # return a numPy array of values of a FEniCS function : 
      elif type(m) in (function.Function, functions.function.Function):
        m_v = m.vector()
        m_a = DoubleArray(m.vector().size())
     
        try:
          m.vector().gather(m_a, arange(m_v.size(), dtype='intc'))
          return array(m_a.array())
        
        except TypeError:
          return m.vector().gather(arange(m_v.size(), dtype='intc'))
      
      # The following type had to be added to the orginal function so that
      # it could accomodate the return from the adjoint system solve.
      elif type(m) == cpp.la.Vector:
        m_a = DoubleArray(m.size())
     
        try:
          m.gather(m_a, arange(m.size(), dtype='intc'))
          return array(m_a.array())
     
        except TypeError:
          return m.gather(arange(m.size(), dtype='intc'))
      
      else:
        raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

    def set_local_from_global(m, m_global_array):
      """
      Sets the local values of the distrbuted object m to the values contained 
      in the global array m_global_array.
      """
      # This had to be changed, because the dolfin-adjoint constant.Constant is
      # different from the constant of dolfin.
      if type(m) == Constant:
        if m.rank() == 0:
          m.assign(m_global_array[0])
      
        else:
          m.assign(Constant(tuple(m_global_array)))
      
      elif type(m) in (function.Function, functions.function.Function):
        begin, end = m.vector().local_range()
        m_a_local  = m_global_array[begin : end]
        m.vector().set_local(m_a_local)
        m.vector().apply('insert')
      
      else:
        raise TypeError, 'Unknown parameter type'

    def _I_fun(beta2_array, *args):
      """
      Solve forward model with given beta2, calculate objective function
      """
      set_local_from_global(model.beta2, beta2_array)
      self.forward_model.solve()
      I = assemble(self.adjoint_instance.I)
      return I
 
    def _J_fun(x, *args):
      """
      Solve adjoint model, calculate gradient
      """
      # dolfin.adjoint method:
      set_local_from_global(model.beta2, x)
      self.adjoint_instance.solve()

      # This is not the best place for this, but we leave it here for now
      # so that we can see the impact of every line search update on the
      # variables of interest.
      J = assemble(self.adjoint_instance.J)
      U = project(as_vector([model.u, model.v, model.w]))
      dSdt = project(-(model.u*model.S.dx(0) + model.v*model.S.dx(1)) + (model.w + model.adot))
      file_u_xml << U
      file_u_pvd << U
      file_b_xml << model.beta2 
      file_b_pvd << model.beta2
      file_dSdt_pvd << dSdt
      return get_global(J)

    #===========================================================================
    # Set up file I/O
    path       = config['output_path']
    file_b_xml = File(path + 'beta2_opt.xml')
    file_b_pvd = File(path + 'beta2_opt.pvd')
    file_u_xml = File(path + 'U_opt.pvd')
    file_u_pvd = File(path + 'U_opt.xml')
    file_dSdt_pvd = File(path + 'dSdt.pvd')

    # Switching over to the parallel version of the optimization that is found 
    # in the dolfin-adjoint optimize.py file:
    maxfun     = config['adjoint']['max_fun']
    bounds     = config['adjoint']['bounds']
    m_global   = get_global(model.beta2)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
      iprint = -1
    else:
      iprint = 1

    # convert bounds to an array of tuples and serialise it in parallel environ.
    bounds_arr = []
    for i in range(2):
      if type(bounds[i]) == int or type(bounds[i]) == float:
        bounds_arr.append(bounds[i] * ones(model.beta2.vector().size()))
      else:
        bounds_arr.append(get_global(bounds[i]))
    bounds = array(bounds_arr).T
    
    # minimize this stuff :
    mopt, f, d = fmin_l_bfgs_b(_I_fun, m_global, fprime=_J_fun, bounds=bounds,
                               maxfun=maxfun, iprint=iprint)
    set_local_from_global(model.beta2, mopt)


class BalanceVelocitySolver(object):
  def __init__(self, model, config):
    self.bv_instance = VelocityBalance(model, config)

  def solve(self):
    self.bv_instance.solve()





