from pylab          import inf, ones, zeros, array, arange, vstack
from fenics         import project, File, vertex_to_dof_map, Function, \
                           assemble, sqrt, DoubleArray, Constant, function, MPI
from physics        import *
from scipy.optimize import fmin_l_bfgs_b
from time           import time
from termcolor      import colored, cprint
import sys
import numpy as np

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
    the age equation.  Turn off any solver by editing the appropriate config
    dict entry to "False".  If config['coupled']['on'] is "False", solve only
    once.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = '::: solving SteadySolver :::'
      text = colored(s, 'blue')
      print text
    model   = self.model
    config  = self.config
    T0      = config['velocity']['T0']
    outpath = config['output_path']
    
    # Set the initial Picard iteration (PI) parameters
    # L_\infty norm in velocity between iterations
    inner_error = inf
   
    # number of iterations
    counter     = 0
   
    # previous velocity for norm calculation
    u_prev      = project(model.u, model.Q).vector().array()
    
    # set an inner tolerance for PI
    inner_tol   = config['coupled']['inner_tol']
    max_iter    = config['coupled']['max_iter']

    # Initialize a temperature field for visc. calc.
    if config['velocity']['use_T0']:
      model.T.vector().set_local( T0 * ones(len(model.T.vector().array())) )
    
    if not config['coupled']['on']: max_iter = 1
    
    # Perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance
    while inner_error > inner_tol and counter < max_iter:
      
      # Solve surface mass balance and temperature boundary condition
      if config['surface_climate']['on']:
        self.surface_climate_instance.solve()

      # Solve velocity
      if config['velocity']['on']:
        self.velocity_instance.solve()
        U = project(as_vector([model.u, model.v, model.w]))
        if config['log']:
          File(outpath + 'U.pvd') << U
          # if the velocity solve is full-stokes, save pressure too : 
          if config['velocity']['approximation'] == 'stokes':
            File(outpath + 'P.pvd') << model.P
        if MPI.rank(mpi_comm_world())==0:
          model.print_min_max(U, 'U')

      # Solve enthalpy (temperature, water content)
      if config['enthalpy']['on']:
        self.enthalpy_instance.solve()
        if config['log']: 
          File(outpath + 'T.pvd')  << model.T   # save temperature
          File(outpath + 'Mb.pvd') << model.Mb  # save melt rate
          File(outpath + 'W.pvd')  << model.W   # save water content
        if MPI.rank(mpi_comm_world())==0:
          model.print_min_max(model.H,  'H')
          model.print_min_max(model.T,  'T')
          model.print_min_max(model.Mb, 'Mb')
          model.print_min_max(model.W,  'W')

      # Calculate L_infinity norm
      if config['coupled']['on']:
        u_new       = project(model.u, model.Q).vector().array()
        diff        = (u_prev - u_new)
        inner_error = diff.max()
        u_prev      = u_new
      
      counter += 1
      
      if MPI.rank(mpi_comm_world())==0:
        s    = 'Picard iteration %i (max %i) done: r = %.3e (tol %.3e)'
        text = colored(s, 'blue')
        print text % (counter, max_iter, inner_error, inner_tol)

    # Solve age equation
    if config['age']['on']:
      self.age_instance.solve()
      if config['log']: File(outpath + 'age.pvd') << model.age  # save age


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
      self.file_U  = File(self.config['output_path']+'U.pvd')
      self.file_T  = File(self.config['output_path']+'T.pvd')
      self.file_S  = File(self.config['output_path']+'S.pvd')
      self.file_a  = File(self.config['output_path']+'age.pvd')
      self.dheight = []
      self.mass    = []
      self.t_log   = []

    self.step_time = []
    self.M_prev    = 1.0

  def rhs_func_explicit(self, t, S, *f_args):
    """
    This function calculates the change in height of the surface of the
    ice sheet.
    
    :param t : Time
    :param S : Current height of the ice sheet
    :rtype   : Array containing rate of change of the ice surface values
    """
    model             = self.model
    config            = self.config
    thklim            = config['free_surface']['thklim']
    B                 = model.B.compute_vertex_values()
    S[(S-B) < thklim] = thklim + B[(S-B) < thklim]
    # the surface is never on a periodic FunctionSpace :
    if config['periodic_boundary_conditions']:
      #v2d = model.Q_non_periodic.dofmap().vertex_to_dof_map(model.flat_mesh)
      d2v = dof_to_vertex_map(model.Q_non_periodic)
    else:
      #v2d = model.Q.dofmap().vertex_to_dof_map(model.flat_mesh)
      d2v = dof_to_vertex_map(model.Q)
    model.S.vector().set_local(S[d2v])
    model.S.vector().apply('insert')
   
    if config['velocity']['on']:
      model.U.vector()[:] = 0.0
      self.velocity_instance.solve()
      if self.config['log']:
        U = project(as_vector([model.u, model.v, model.w]))
        self.file_U << U
      if MPI.rank(mpi_comm_world())==0:
        model.print_min_max(U, 'U')

    if config['surface_climate']['on']:
      self.surface_climate_instance.solve()
   
    if config['free_surface']['on']:
      self.surface_instance.solve()
      if self.config['log']:
        self.file_S << model.S
      if MPI.rank(mpi_comm_world())==0:
        model.print_min_max(model.S, 'S')
 
    return model.dSdt.compute_vertex_values()

  def solve(self):
    """
    Performs the physics, evaluating and updating the enthalpy and age as 
    well as storing the velocity, temperature, and the age in vtk files.

    """
    if MPI.rank(mpi_comm_world())==0:
      s    = '::: solving TransientSolver :::'
      text = colored(s, 'blue')
      print text
    model  = self.model
    config = self.config
    
    t      = config['t_start']
    t_end  = config['t_end']
    dt     = config['time_step']
    thklim = config['free_surface']['thklim']
   
    mesh   = model.mesh 
    smb    = model.smb
    sigma  = model.sigma

    S      = model.S
    B      = model.B
 
    smb.interpolate(config['free_surface']['observed_smb'])

    if config['periodic_boundary_conditions']:
      d2v = dof_to_vertex_map(model.Q_non_periodic)
      mhat_non = Function(model.Q_non_periodic)
    else:
      d2v = dof_to_vertex_map(model.Q)

    # Loop over all times
    while t <= t_end:

      B_a = B.compute_vertex_values()
      S_v = S.compute_vertex_values()
      
      tic = time()

      S_0 = S_v
      f_0 = self.rhs_func_explicit(t, S_0)
      S_1 = S_0 + dt*f_0
      S_1[(S_1-B_a) < thklim] = thklim + B_a[(S_1-B_a) < thklim]
      S.vector().set_local(S_1[d2v])
      S.vector().apply('')

      f_1                     = self.rhs_func_explicit(t, S_1)
      S_2                     = 0.5*S_0 + 0.5*S_1 + 0.5*dt*f_1
      S_2[(S_2-B_a) < thklim] = thklim + B_a[(S_2-B_a) < thklim] 
      S.vector().set_local(S_2[d2v])
      S.vector().apply('')
     
      mesh.coordinates()[:, 2] = sigma.compute_vertex_values()*(S_2 - B_a) + B_a
      if config['periodic_boundary_conditions']:
        temp = (S_2[d2v] - S_0[d2v])/dt * sigma.vector().get_local()
        mhat_non.vector().set_local(temp)
        mhat_non.vector().apply('')
        m_temp = project(mhat_non,model.Q)
        model.mhat.vector().set_local(m_temp.vector().get_local())
        model.mhat.vector().apply('')
      else:
        temp = (S_2[d2v] - S_0[d2v])/dt * sigma.vector().get_local()
        model.mhat.vector().set_local(temp)
        model.mhat.vector().apply('')
      # Calculate enthalpy update
      if self.config['enthalpy']['on']:
        self.enthalpy_instance.solve(H0=model.H, Hhat=model.H, uhat=model.u, 
                                   vhat=model.v, what=model.w, mhat=model.mhat)
        if self.config['log']:
          self.file_T << model.T
        if MPI.rank(mpi_comm_world())==0:
          model.print_min_max(model.H,  'H')
          model.print_min_max(model.T,  'T')
          model.print_min_max(model.Mb, 'Mb')
          model.print_min_max(model.W,  'W')

      # Calculate age update
      if self.config['age']['on']:
        self.age_instance.solve(A0=model.A, Ahat=model.A, uhat=model.u, 
                                vhat=model.v, what=model.w, mhat=model.mhat)
        if config['log']: 
          self.file_a << model.age
        if MPI.rank(mpi_comm_world())==0:
          model.print_min_max(model.age, 'age')

      # store information : 
      if self.config['log']:
        self.t_log.append(t)
        M = assemble(self.surface_instance.M)
        self.mass.append(M)

      # increment time step :
      if MPI.rank(mpi_comm_world())==0:
        s = '>>> Time: %i yr, CPU time for last dt: %.3f s, Mass: %.2f <<<'
        text = colored(s, 'red', attrs=['bold'])
        print text % (t, time()-tic, M/self.M_prev)

      self.M_prev = M
      t          += dt
      self.step_time.append(time() - tic)

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
    Initialize the model with a forward instance (SteadySolver) and adjoint
    solver (AdjointVelocityBP, only adjoint currently available).
    """
    self.model  = model
    self.config = config
    
    config['mode'] = 'steady' # adjoint only solves steady-state
    
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
      Smag   = project(sqrt(S.dx(0)**2 + S.dx(1)**2 + 1e-10), Q)
      model.U_o.interpolate(U)
      model.u_o = project(-model.U_o * S.dx(0) / Smag, Q)
      model.v_o = project(-model.U_o * S.dx(1) / Smag, Q)      

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
    if MPI.rank(mpi_comm_world())==0:
      s    = '::: solving AdjointSolver :::'
      text = colored(s, 'blue')
      print text
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

    def I(c_array, *args):
      """
      Solve forward model with given control, calculate objective function
      """
      n = len(c_array)/len(config['adjoint']['control_variable'])
      for ii,c in enumerate(config['adjoint']['control_variable']):
        set_local_from_global(c, c_array[ii*n:(ii+1)*n])
      self.forward_model.solve()
      I = assemble(self.adjoint_instance.I)  #FIXME: ISMIP_HOM inverse C fails
      return I
 
    def J(c_array, *args):
      """
      Solve adjoint model, calculate gradient
      """
      # dolfin.adjoint method:
      n = len(c_array)/len(config['adjoint']['control_variable'])
      for ii,c in enumerate(config['adjoint']['control_variable']):
        set_local_from_global(c, c_array[ii*n:(ii+1)*n])
      self.adjoint_instance.solve()

      # This is not the best place for this, but we leave it here for now
      # so that we can see the impact of every line search update on the
      # variables of interest.
      Js = []
      for JJ in self.adjoint_instance.J:
        Js.extend(get_global(assemble(JJ)))
      Js   = array(Js)
      return Js

    #===========================================================================
    # Set up file I/O
    path = config['output_path']
    file_b_pvd    = File(path + 'beta2.pvd')
    file_u_pvd    = File(path + 'U.pvd')
    file_dSdt_pvd = File(path + 'dSdt.pvd')

    # Switching over to the parallel version of the optimization that is found 
    # in the dolfin-adjoint optimize.py file:
    maxfun      = config['adjoint']['max_fun']
    bounds_list = config['adjoint']['bounds']
    beta_0      = []
    for mm in config['adjoint']['control_variable']:
      beta_0.extend(get_global(mm))
    beta_0 = array(beta_0)

    # Shut up all processors but the first one.
    if MPI.rank(mpi_comm_world()) != 0:
      iprint = -1
    else:
      iprint = 1
    b = []
    # convert bounds to an array of tuples and serialize it in parallel environ.
    for bounds in bounds_list:
      bounds_arr = []
      for i in range(2):
        if type(bounds[i]) == int or type(bounds[i]) == float:
          bounds_arr.append(bounds[i] * ones(model.beta2.vector().size()))
        else:
          bounds_arr.append(get_global(bounds[i]))
      b.append(array(bounds_arr).T)
    bounds = vstack(b)  
    if MPI.rank(mpi_comm_world())==0:
      text = colored(bounds, 'red', attrs=['bold'])
      print text
    
    # minimize I with initial guess beta_0 and gradient J :
    mopt, f, d = fmin_l_bfgs_b(I, beta_0, fprime=J, bounds=bounds,
                               maxfun=maxfun, iprint=iprint)

    n = len(mopt)/len(config['adjoint']['control_variable'])
    for ii,c in enumerate(config['adjoint']['control_variable']):
      set_local_from_global(c, mopt[ii*n:(ii+1)*n])
      
    # save the output for this iteration of l_bfgs_b :
    U    = project(as_vector([model.u, model.v, model.w]))
    dSdt = project(- (model.u*model.S.dx(0) + model.v*model.S.dx(1)) \
                   + model.w + model.adot)
    file_b_pvd    << model.extrude(model.beta2, 3, 2)
    file_u_pvd    << U
    file_dSdt_pvd << dSdt


class BalanceVelocitySolver(object):
  def __init__(self, model, config):
    self.bv_instance = VelocityBalance(model, config)

  def solve(self):
    self.bv_instance.solve()


class StokesBalanceSolver(object):

  def __init__(self, model, config):
    """
    Calculate each of the component stresses which define the full stress
    of the ice-sheet.
    
    RETURNS:
      tau_lon - longitudinal stress field
      tau_lat - lateral stress field
      tau_bas - frictional sliding stress at the bed
      tau_drv - driving stress of the system 
    
    Note: tau_drv = tau_lon + tau_lat + tau_bas
    
    """
    print "::: initializing 'stokes-balance' solver :::"
    self.model  = model
    self.config = config
    
    Q       = model.Q
    u       = model.u
    v       = model.v
    w       = model.w
    S       = model.S
    B       = model.B
    H       = S - B
    eta     = model.eta
    beta2   = model.beta2
    
    # get the values at the bed :
    beta2_e = model.extrude(beta2, 3, 2, Q)
    u_b_e   = model.extrude(u,     3, 2, Q)
    v_b_e   = model.extrude(v,     3, 2, Q)
    
    # vertically average :
    etabar = model.vert_integrate(eta, Q)
    etabar = project(model.extrude(etabar, 2, 2, Q) / H)
    ubar   = model.vert_integrate(u, Q)
    ubar   = project(model.extrude(ubar, 2, 2, Q) / H)
    vbar   = model.vert_integrate(v, Q)
    vbar   = project(model.extrude(vbar, 2, 2, Q) / H)

    # set the model variables so the physics object can solve it :
    model.beta2_e = beta2_e
    model.u_b_e   = u_b_e
    model.v_b_e   = v_b_e
    model.etabar  = etabar
    model.ubar    = ubar
    model.vbar    = vbar
    
    # calculate the driving stress and basal drag once :
    model.tau_d   = model.calc_tau_drv(Q)
    model.tau_b   = model.calc_tau_bas(Q)
    
    self.Q = Q

    self.stress_balance_instance = StokesBalance(model, config)

  def solve(self):
    """ 
    """
    model   = self.model
    config  = self.config
    outpath = self.config['output_path']
    
    # Set the initial Picard iteration (PI) parameters
    # L_\infty norm in velocity between iterations
    inner_error = inf
    
    # number of iterations
    counter = 0
   
    # set an inner tolerance for PI
    max_iter = 1
   
    # previous velocity for norm calculation
    u_prev   = zeros(len(model.ubar.vector().array()))
    
    # tolerance to stop solving :
    inner_tol = 0.0
    
    # Perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance
    while counter < max_iter and inner_error > inner_tol:
      
      self.stress_balance_instance.solve()
      
      # Calculate L_infinity norm
      ubar_v      = model.ubar.vector().array()
      vbar_v      = model.vbar.vector().array()
      ubar_n      = np.sqrt(ubar_v**2 + vbar_v**2)
      diff        = abs(u_prev - ubar_n)
      inner_error = diff.max()
      u_prev      = ubar_n
      
      counter += 1
      
      if MPI.rank(mpi_comm_world())==0:
        s    = 'Picard iteration %i (max %i) done: r = %.3e (tol %.3e)'
        text = colored(s, 'blue')
        print text % (counter, max_iter, inner_error, inner_tol)
   
    # solve for the stress balance given the appropriate vertically 
    # averaged velocities :
    self.stress_balance_instance.component_stress_stokes()

