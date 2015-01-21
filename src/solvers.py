from pylab          import inf, ones, zeros, array, arange, vstack, unique
from fenics         import project, File, vertex_to_dof_map, Function, \
                           assemble, sqrt, DoubleArray, Constant, function, MPI
from physics        import *
from scipy.optimize import fmin_l_bfgs_b
from time           import time
from termcolor      import colored, cprint
from helper         import raiseNotDefined
from io             import print_min_max, print_text
import sys
import numpy as np


class Solver(object):
  """
  This abstract class outlines the structure of a VarGlaS solver.
  """
  def solve(self):
    """
    Solves the problem utilizing Physics object(s).
    """
    raiseNotDefined()
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'deep_sky_blue_2'


class SteadySolver(Solver):
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
    s    = "::: INITIALIZING STEADY SOLVER :::"
    print_text(s, self.color())
    self.model          = model
    self.config         = config
    self.config['mode'] = 'steady'
    outpath             = config['output_path']

    # velocity model :
    if self.config['velocity']['on']:
      if   config['velocity']['approximation'] == 'fo':
        self.velocity_instance = VelocityBP(model, config)
      elif config['velocity']['approximation'] == 'stokes':
        self.velocity_instance = VelocityStokes(model, config)
      else:
        s = "Please use 'fo' or 'stokes'. "
        print_text(s, self.color())
      if config['velocity']['log']:
        self.U_file = File(outpath + 'U.pvd')
        self.P_file = File(outpath + 'P.pvd')
      if config['velocity']['init_beta_from_stats']:
        self.beta_file = File(outpath + 'beta.pvd')
    
    # enthalpy model :
    if config['enthalpy']['on']:
      self.enthalpy_instance = Enthalpy(model, config)
      if config['enthalpy']['log']:
        self.T_file   = File(outpath + 'T.pvd')
        self.W_file   = File(outpath + 'W.pvd')
        self.M_file   = File(outpath + 'Mb.pvd')

    # age model :
    if config['age']['on']:
      self.age_instance = Age(model, config)
      if config['age']['log']:
        self.a_file = File(outpath + 'age.pvd')
    
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
    s    = '::: solving SteadySolver :::'
    print_text(s, self.color())
    model   = self.model
    config  = self.config
    outpath = config['output_path']
    
    # Set the initial Picard iteration (PI) parameters
    # L_\infty norm in velocity between iterations
    inner_error = inf
   
    # number of iterations
    counter     = 0
   
    # previous velocity for norm calculation
    u_prev      = model.U.vector().array()
    
    # set an inner tolerance for PI
    inner_tol   = config['coupled']['inner_tol']
    max_iter    = config['coupled']['max_iter']

    # Initialize a temperature field for visc. calc.
    if config['velocity']['use_T0']:
      model.assign_variable(model.T, config['velocity']['T0'])
    
    if not config['coupled']['on']: max_iter = 1
    
    # Perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance
    while inner_error > inner_tol and counter < max_iter:
     
      # reset the velocity for Newton solve to converge : 
      if counter > 0:  
        model.assign_variable(model.U, 0.0)
      
      # Solve surface mass balance and temperature boundary condition
      if config['surface_climate']['on']:
        self.surface_climate_instance.solve()

      # Solve velocity
      if config['velocity']['on']:
        self.velocity_instance.solve()
        if config['velocity']['log'] and config['log']:
          s    = '::: saving velocity and pressure U and P .pvd files :::'
          print_text(s, self.color())
          U = project(as_vector([model.u, model.v, model.w]))
          if config['log_history']:
            self.U_file << U
          else:
            File(outpath + 'U.pvd')  << U
          # if the velocity solve is full-stokes, project pressure onto P1 : 
          if config['velocity']['approximation'] == 'stokes':
            P = project(model.P, model.Q)
          else:
            P = model.P
            if config['log_history']:
              self.P_file << P
            else:
              File(outpath + 'P.pvd') << P
          if config['velocity']['init_beta_from_stats']:
            s    = '::: saving stats beta.pvd file :::'
            print_text(s, self.color())
            if config['log_history']:
              self.beta_file << model.beta
            else:
              File(outpath + 'beta.pvd') << model.beta

      # Solve enthalpy (temperature, water content)
      if config['enthalpy']['on']:
        self.enthalpy_instance.solve()
        if config['enthalpy']['log'] and config['log']: 
          s    = '::: saving enthalpy fields T, Mb, and W .pvd files :::'
          print_text(s, self.color())
          if config['log_history']:
            self.T_file    << model.T    # save temperature
            self.M_file    << model.Mb   # save melt rate
            self.W_file    << model.W    # save water content
          else:
            File(outpath + 'T.pvd')   << model.T
            File(outpath + 'W.pvd')   << model.Mb
            File(outpath + 'Mb.pvd')  << model.W

      # Calculate L_infinity norm
      if config['coupled']['on']:
        u_new       = model.U.vector().array()
        diff        = (u_prev - u_new)
        inner_error = diff.max()
        u_prev      = u_new
      
      counter += 1
      
      if self.model.MPI_rank==0:
        s1    = 'Picard iteration %i (max %i) done: ' % (counter, max_iter)
        s2    = 'r = %.3e ' % inner_error
        s3    = '(tol %.3e)' % inner_tol
        text1 = colored(s1, 'blue')
        text2 = colored(s2, 'red', attrs=['bold'])
        text3 = colored(s3, 'blue')
        print text1 + text2 + text3

    # Solve age equation
    if config['age']['on']:
      self.age_instance.solve()
      if config['log'] and config['log']: 
        s    = '::: saving age age.pvd file :::'
        print_text(s, self.color())
        if config['log_history']:
          self.a_file << model.age  # save age
        else:
          File(outpath + 'age.pvd')  << model.age


class TransientSolver(Solver):
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
    s    = "::: INITIALIZING TRANSIENT SOLVER :::"
    print_text(s, self.color())
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
        s =  "Please choose 'fo' or 'stokes'. "
        print_text(s, self.color())
    
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
      d2v = dof_to_vertex_map(model.Q_non_periodic)
    else:
      d2v = dof_to_vertex_map(model.Q)
    
    model.assign_variable(model.S, S[d2v])
   
    if config['velocity']['on']:
      model.U.vector()[:] = 0.0
      self.velocity_instance.solve()
      if config['velocity']['log']:
        U = project(as_vector([model.u, model.v, model.w]))
        s    = '::: saving velocity U.pvd file :::'
        print_text(s, self.color())
        self.file_U << U
      print_min_max(U, 'U')

    if config['surface_climate']['on']:
      self.surface_climate_instance.solve()
   
    if config['free_surface']['on']:
      self.surface_instance.solve()
      if self.config['log']:
        s    = '::: saving surface S.pvd file :::'
        print_text(s, self.color())
        self.file_S << model.S
      print_min_max(model.S, 'S')
 
    return model.dSdt.compute_vertex_values()

  def solve(self):
    """
    Performs the physics, evaluating and updating the enthalpy and age as 
    well as storing the velocity, temperature, and the age in vtk files.

    """
    s    = '::: solving TransientSolver :::'
    print_text(s, self.color())
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
      model.assign_variable(S, S_1[d2v])

      f_1                     = self.rhs_func_explicit(t, S_1)
      S_2                     = 0.5*S_0 + 0.5*S_1 + 0.5*dt*f_1
      S_2[(S_2-B_a) < thklim] = thklim + B_a[(S_2-B_a) < thklim] 
      model.assign_variable(S, S_2[d2v])
     
      mesh.coordinates()[:, 2] = sigma.compute_vertex_values()*(S_2 - B_a) + B_a
      if config['periodic_boundary_conditions']:
        temp = (S_2[d2v] - S_0[d2v])/dt * sigma.vector().get_local()
        model.assign_variable(mhat_non, temp)
        m_temp = project(mhat_non,model.Q)
        model.assign_variable(model.mhat, m_temp.vector().get_local())
      else:
        temp = (S_2[d2v] - S_0[d2v])/dt * sigma.vector().get_local()
        model.assign_variable(model.mhat, temp)
      # Calculate enthalpy update
      if self.config['enthalpy']['on']:
        self.enthalpy_instance.solve(H0=model.H, Hhat=model.H, uhat=model.u, 
                                   vhat=model.v, what=model.w, mhat=model.mhat)
        if self.config['enthalpy']['log']:
          s    = '::: saving temperature T.pvd file :::'
          print_text(s, self.color())
          self.file_T << model.T
        print_min_max(model.H,  'H')
        print_min_max(model.T,  'T')
        print_min_max(model.Mb, 'Mb')
        print_min_max(model.W,  'W')

      # Calculate age update
      if self.config['age']['on']:
        self.age_instance.solve(A0=model.A, Ahat=model.A, uhat=model.u, 
                                vhat=model.v, what=model.w, mhat=model.mhat)
        if config['log']: 
          s   = '::: saving age age.pvd file :::'
          print_text(s, self.color())
          self.file_a << model.age
        print_min_max(model.age, 'age')

      # store information : 
      if self.config['log']:
        self.t_log.append(t)
        M = assemble(self.surface_instance.M)
        self.mass.append(M)

      # increment time step :
      if self.model.MPI_rank==0:
        s = '>>> Time: %i yr, CPU time for last dt: %.3f s, Mass: %.2f <<<'
        text = colored(s, 'red', attrs=['bold'])
        print text % (t, time()-tic, M/self.M_prev)

      self.M_prev = M
      t          += dt
      self.step_time.append(time() - tic)

class AdjointSolver(Solver):
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
    s    = "::: INITIALIZING ADJOINT SOLVER :::"
    print_text(s, self.color())
    self.model  = model
    self.config = config
    
    config['mode'] = 'steady' # adjoint only solves steady-state
    
    # Set up file I/O
    self.path      = config['output_path']
    self.beta_file = File(self.path + 'beta.pvd')
    self.U_file    = File(self.path + 'U_obs.pvd')
    self.dSdt_file = File(self.path + 'dSdt.pvd')
   
    # ensure that we have lists : 
    if type(config['adjoint']['bounds']) != list:
      config['adjoint']['bounds'] = [config['adjoint']['bounds']]
    if type(config['adjoint']['control_variable']) != list:
      cv = config['adjoint']['control_variable']
      config['adjoint']['control_variable'] = [cv]
    if type(config['adjoint']['alpha']) != list:
      config['adjoint']['alpha'] = [config['adjoint']['alpha']]

    # Switching over to the parallel version of the optimization that is found 
    # in the dolfin-adjoint optimize.py file:
    self.maxfun      = config['adjoint']['max_fun']
    self.bounds_list = config['adjoint']['bounds']
    self.control     = config['adjoint']['control_variable']
    
    # initialize instances of the forward model, and the adjoint physics : 
    self.forward_model    = SteadySolver(model, config)
    self.adjoint_instance = AdjointVelocity(model, config)
  
  def set_velocity(self, u, v, w):
    """
    set the velocity.
    """
    model = self.model
    model.assign_variable(model.u, u)
    model.assign_variable(model.v, v)
    model.assign_variable(model.w, w)

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
      model.assign_variable(model.u_o, u)
      model.assign_variable(model.v_o, v)

    elif U != None:
      Smag   = project(sqrt(S.dx(0)**2 + S.dx(1)**2 + 1e-10), Q)
      u_n    = project(-U * S.dx(0) / Smag, Q)
      v_n    = project(-U * S.dx(1) / Smag, Q)      
      model.assign_variable(model.u_o, u_n)
      model.assign_variable(model.v_o, v_n)

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
    s    = '::: solving AdjointSolver :::'
    print_text(s, self.color())
    model       = self.model
    config      = self.config
    bounds_list = self.bounds_list
    control     = self.control
    maxfun      = self.maxfun
   
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
        model.assign_variable(m, m_a_local)
      
      else:
        raise TypeError, 'Unknown parameter type'

    def I(c_array, *args):
      """
      Solve forward model with given control, calculate objective function
      """
      n = len(c_array)/len(control)
      for ii,c in enumerate(control):
        set_local_from_global(c, c_array[ii*n:(ii+1)*n])
      self.forward_model.solve()
      print_min_max(model.u_o, 'u_o')
      print_min_max(model.v_o, 'v_o')
      I = assemble(self.adjoint_instance.I)
      return I
 
    def J(c_array, *args):
      """
      Solve adjoint model, calculate gradient
      """
      # dolfin.adjoint method:
      n = len(c_array)/len(control)
      for ii,c in enumerate(control):
        set_local_from_global(c, c_array[ii*n:(ii+1)*n])
      self.adjoint_instance.solve()

      for i,c in enumerate(control):
        print_min_max(c, 'c_' + str(i))

      Js = []
      for JJ in self.adjoint_instance.J:
        Js.extend(get_global(assemble(JJ)))
      Js   = array(Js)
      return Js

    #===========================================================================
    # begin the optimization :

    # form the initial guess :
    beta_0      = []
    for c in control:
      beta_0.extend(get_global(c))
    beta_0 = array(beta_0)

    # shut up all processors but the first one :
    if self.model.MPI_rank != 0:
      iprint = -1
    else:
      iprint = 1
    
    # convert bounds to an array of tuples and serialize it in parallel environ.
    b = []
    for bounds in bounds_list:
      bounds_arr = []
      for i in range(2):
        if type(bounds[i]) == int or type(bounds[i]) == float:
          bounds_arr.append(bounds[i] * ones(model.beta.vector().size()))
        else:
          bounds_arr.append(get_global(bounds[i]))
      b.append(array(bounds_arr).T)
    bounds = vstack(b)
    
    # print the bounds :
    if self.model.MPI_rank==0:
      """
      find the unique values of each row of array <S>.
      """
      unq  = unique(bounds)
      text = colored("unique bounds:\n" + str(unq), 'red', attrs=['bold'])
      print text
    
    # minimize function I with initial guess beta_0 and gradient function J :
    mopt, f, d = fmin_l_bfgs_b(I, beta_0, fprime=J, bounds=bounds,
                               maxfun=maxfun, iprint=iprint)

    n = len(mopt)/len(control)
    for ii,c in enumerate(control):
      set_local_from_global(c, mopt[ii*n:(ii+1)*n])
      
    # save the output :
    s    = '::: saving adjoint beta, U_obs, U_adj, and DSdt .pvd files :::'
    print_text(s, self.color())
    U_obs = project(as_vector([model.u_o, model.v_o, 0]))
    dSdt  = project(- (model.u*model.S.dx(0) + model.v*model.S.dx(1)) \
                    + model.w + model.adot)
    self.beta_file << model.extrude(model.beta, [3,5], 2) 
    self.U_file    << U_obs
    self.dSdt_file << dSdt


class StokesBalanceSolver(Solver):

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
    s    = "::: INITIALIZING STOKES-BALANCE SOLVER :::"
    print_text(s, self.color())
    self.model  = model
    self.config = config
    
    s    = "::: initializing '3D-stokes-balance' solver :::"
    print_text(s, self.color())
    
    self.stress_balance_instance = StokesBalance3D(model, config)

  def solve(self):
    """ 
    """
    model   = self.model
    config  = self.config
    outpath = self.config['output_path']
    
    print_min_max(model.u, 'u')
    print_min_max(model.v, 'v')
    print_min_max(model.w, 'w')
    
    # calculate ubar, vbar :
    self.stress_balance_instance.solve()
    if config['log']:
      U_s = as_vector([model.u_s, model.v_s])
      File(outpath + 'U_s.pvd') << project(U_s)
    
    # solve for the stress balance given the appropriate vertically 
    # averaged velocities :
    self.stress_balance_instance.component_stress_stokes()
    if config['log']: 
      File(outpath + "memb_n.pvd")   << model.memb_n
      File(outpath + "memb_t.pvd")   << model.memb_t
      File(outpath + "membrane.pvd") << model.membrane
      File(outpath + "driving.pvd")  << model.driving
      File(outpath + "basal.pvd")    << model.basal
      File(outpath + "basal_2.pvd")  << model.basal_2
      File(outpath + "pressure.pvd") << model.pressure
      File(outpath + "total.pvd")    << model.total



