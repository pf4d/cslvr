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
    return 'turquoise_2'


class BalanceVelocitySolver(Solver):
    
  def __init__(self, model, config):
    s    = "::: INITIALIZING BALANCE VELOCITY SOLVER :::"
    print_text(s, self.color())
    self.model  = model
    self.config = config
    outpath     = config['output_path']
      
    self.BV_instance = VelocityBalance(model, config)
    
  def solve(self):
    """
    Solves for the velocity balance.
    """
    s    = '::: solving BalanceVelocitySolver :::'
    print_text(s, self.color())
    model   = self.model
    config  = self.config

    self.BV_instance.solve()

    if self.config['balance_velocity']['log']:
      if config['log_history']:
        self.BV_file << model.Ubar
      else:
        model.save_pvd(model.Ubar, 'Ubar')
  

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

    # velocity model :
    if self.config['velocity']['on']:
      if   config['model_order'] == 'BP':
        if config['use_dukowicz']:
          self.velocity_instance = VelocityDukowiczBP(model, config)
        else:
          if config['velocity']['full_BP']:
            self.velocity_instance = VelocityBPFull(model, config)
          else :
            self.velocity_instance = VelocityBP(model, config)
      elif config['model_order'] == 'stokes':
        if config['use_dukowicz']:
          self.velocity_instance = VelocityDukowiczStokes(model, config)
        else:
          self.velocity_instance = VelocityStokes(model, config)
      elif config['model_order'] == 'L1L2':
        self.velocity_instance = VelocityHybrid(model, config)
      else:
        s = "Please use 'BP', 'stokes', or 'L1L2'. "
        print_text(s, self.color())
    
    # enthalpy model :
    if config['enthalpy']['on']:
      if   self.config['model_order'] == 'L1L2':
        self.enthalpy_instance = EnergyHybrid(model, config)
      else:
        self.enthalpy_instance = Enthalpy(model, config)

    # age model :
    if config['age']['on']:
      self.age_instance = Age(model, config)

    # balance velocity model :
    if config['balance_velocity']['on']:
      self.balance_velocity_instance = VelocityBalance(model, config)

    # stress balance model :
    if config['stokes_balance']['on']:
      self.stokes_balance_instance = BP_Balance(model, config)
    
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
    
    # Set the initial Picard iteration (PI) parameters
    # L_\infty norm in velocity between iterations
    inner_error = inf
   
    # number of iterations
    counter     = 0
   
    # previous velocity for norm calculation
    #u_prev      = model.u.vector().array()
    U_prev      = project(as_vector([model.u, model.v, model.w]))
    
    # set an inner tolerance for PI
    inner_tol   = config['coupled']['inner_tol']
    max_iter    = config['coupled']['max_iter']

    if not config['coupled']['on']: max_iter = 1
    
    # Perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance
    while inner_error > inner_tol and counter < max_iter:
     
      # need zero initial guess for Newton solve to converge : 
      model.assign_variable(model.U, DOLFIN_EPS)
      model.assign_variable(model.w, DOLFIN_EPS)
      
      # Solve surface mass balance and temperature boundary condition
      if config['surface_climate']['on']:
        self.surface_climate_instance.solve()

      # Solve velocity
      if config['velocity']['on']:
        self.velocity_instance.solve()
        if config['velocity']['log'] and config['log']:
          if config['model_order'] == 'L1L2':
            U_s  = project(as_vector([model.u_s, model.v_s, model.w_s]))
            U_b  = project(as_vector([model.u_b, model.v_b, model.w_b]))
            model.save_pvd(U_s, 'Us')
            model.save_pvd(U_b, 'Ub')
          else:
            U = project(as_vector([model.u, model.v, model.w]))
            model.save_pvd(U, 'U')
            # save pressure if desired :
            if config['velocity']['calc_pressure']:
              model.save_pvd(model.P, 'P')

      # Solve enthalpy (temperature, water content)
      if config['enthalpy']['on']:
        self.enthalpy_instance.solve()
        if config['enthalpy']['log'] and config['log']: 
          if config['model_order'] == 'L1L2':
            model.save_pvd(model.Ts, 'Ts')
            model.save_pvd(model.Tb, 'Tb')
            model.save_pvd(model.Mb, 'Mb')
          else :
            model.save_pvd(model.T,  'T')
            model.save_pvd(model.W,  'W')
            model.save_pvd(model.Mb, 'Mb')
    
      ## re-compute the friction field :
      #if config['velocity']['transient_beta'] == 'stats':
      #  s    = "::: updating statistical beta :::"
      #  print_text(s, self.color())
      #  beta   = project(model.beta_f, model.Q)
      #  beta_v = beta.vector().array()
      #  ##betaSIA_v = model.betaSIA.vector().array()
      #  ##beta_v[beta_v < 10.0]   = betaSIA_v[beta_v < 10.0]
      #  beta_v[beta_v < 0.0]    = 0.0
      #  #beta_v[beta_v > 2500.0] = 2500.0
      #  model.assign_variable(model.beta, beta_v)
      #  print_min_max(model.beta, 'beta')
      #  if config['log']:
      #    model.save_pvd(model.extrude(model.beta, [3,5], 2), 'beta')

      counter += 1
      # Calculate L_infinity norm
      if config['coupled']['on']:
        #u_new         = model.u.vector().array()
        #diff          = (u_prev - u_new)
        #inner_error_n = MPI.max(mpi_comm_world(), diff.max())
        #u_prev        = u_new
        inner_error_n = norm(project(U_prev - U))
        U_prev        = U
        if self.model.MPI_rank==0:
          s1    = 'Picard iteration %i (max %i) done: ' % (counter, max_iter)
          s2    = 'r0 = %.3e'  % inner_error
          s3    = ', '
          s4    = 'r = %.3e ' % inner_error_n
          s5    = '(tol %.3e)' % inner_tol
          text1 = colored(s1, 'blue')
          text2 = colored(s2, 'red', attrs=['bold'])
          text3 = colored(s3, 'blue')
          text4 = colored(s4, 'red', attrs=['bold'])
          text5 = colored(s5, 'blue')
          print text1 + text2 + text3 + text4 + text5
        inner_error = inner_error_n

    # Solve age equation
    if config['age']['on']:
      self.age_instance.solve()
      if config['age']['log'] and config['log']: 
        model.save_pvd(model.age, 'age')

    # solve balance velocity :
    if config['balance_velocity']['on']:
      self.balance_velocity_instance.solve()
      if config['balance_velocity']['log'] and config['log']: 
        model.save_pvd(model.Ubar, 'Ubar')

    # solve stress balance :
    if config['stokes_balance']['on']:
      self.stokes_balance_instance.solve()
      if config['stokes_balance']['log'] and config['log']: 
        memb_n   = as_vector([model.tau_ii, model.tau_ij, model.tau_iz])
        memb_t   = as_vector([model.tau_ji, model.tau_jj, model.tau_jz])
        memb_x   = model.tau_ii + model.tau_ij + model.tau_iz
        memb_y   = model.tau_ji + model.tau_jj + model.tau_jz
        membrane = as_vector([model.memb_x, model.memb_y, 0.0])
        driving  = as_vector([model.tau_id, model.tau_jd, 0.0])
        basal    = as_vector([model.tau_ib, model.tau_jb, 0.0])
        basal_2  = as_vector([model.tau_iz, model.tau_jz, 0.0])
        pressure = as_vector([model.tau_ip, model.tau_jp, 0.0])
        
        total    = membrane + basal + pressure - driving
        
        # attach the results to the model :
        s    = "::: projecting '3D-stokes-balance' terms onto vector space :::"
        print_text(s, self.color())
        
        memb_n   = project(memb_n)
        memb_t   = project(memb_t)
        membrane = project(membrane)
        driving  = project(driving)
        basal    = project(basal)
        basal_2  = project(basal_2)
        pressure = project(pressure)
        total    = project(total)

        print_min_max(memb_n,   "memb_n")
        print_min_max(memb_t,   "memb_t")
        print_min_max(membrane, "membrane")
        print_min_max(driving,  "driving")
        print_min_max(basal,    "basal")
        print_min_max(basal_2,  "basal_2")
        print_min_max(pressure, "pressure")
        print_min_max(total,    "total")

        model.save_pvd(memb_n,   'memb_n')
        model.save_pvd(memb_t,   'memb_t')
        model.save_pvd(membrane, 'membrane')
        model.save_pvd(driving,  'driving')
        model.save_pvd(basal,    'basal')
        model.save_pvd(pressure, 'pressure')
        model.save_pvd(total,    'total')
    


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
      
      if   self.config['model_order'] == 'BP':
        if config['use_dukowicz']:
          self.velocity_instance = VelocityDukowiczBP(model, config)
        else:
          self.velocity_instance = VelocityBP(model, config)
      
      elif self.config['model_order'] == 'stokes':
        if config['use_dukowicz']:
          self.velocity_instance = VelocityDukowiczStokes(model, config)
        else:
          self.velocity_instance = VelocityStokes(model, config)
      
      elif config['model_order'] == 'L1L2':
        self.velocity_instance = VelocityHybrid(model, config)
      
      else:
        s =  "Please choose 'BP' or 'stokes'. "
        print_text(s, self.color())
    
    # initialized enthalpy solver : 
    if self.config['enthalpy']['on']:
      if   self.config['model_order'] == 'L1L2':
        self.enthalpy_instance = EnergyHybrid(model, config)
      else:
        self.enthalpy_instance = Enthalpy(model, config)

    # initialize age solver :
    if self.config['age']['on']:
      self.age_instance = Age(model, config)

    # initialize surface climate solver :
    if self.config['surface_climate']['on']:
      self.surface_climate_instance = SurfaceClimate(model, config)

    # initialize free surface solver :
    if config['free_surface']['on']:
      if   self.config['model_order'] == 'L1L2':
        self.surface_instance = MassBalanceHybrid(model, config)
      else:
        self.surface_instance = FreeSurface(model, config)
        self.M_prev           = 1.0

    # Set up files for logging time dependent solutions to paraview files.
    if config['log']:
      self.file_U  = File(self.config['output_path']+'U.pvd')
      self.file_T  = File(self.config['output_path']+'T.pvd')
      self.file_Ts = File(self.config['output_path']+'Ts.pvd')
      self.file_Tb = File(self.config['output_path']+'Tb.pvd')
      self.file_S  = File(self.config['output_path']+'S.pvd')
      self.file_H  = File(self.config['output_path']+'H.pvd')
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
    adot   = model.adot
    sigma  = model.sigma

    S      = model.S
    B      = model.B

    if config['periodic_boundary_conditions']:
      d2v      = dof_to_vertex_map(model.Q_non_periodic)
      mhat_non = Function(model.Q_non_periodic)
    else:
      d2v      = dof_to_vertex_map(model.Q)

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


class AdjointSolverNew(Solver):
  """
  """

  def __init__(self, model, config):
    """
    Initialize the model with a forward instance (SteadySolver) and adjoint
    solver (AdjointDukowiczVelocity, AdjointVelocity).
    """
    s    = "::: INITIALIZING NEW ADJOINT SOLVER :::"
    print_text(s, self.color())
    self.model  = model
    self.config = config
    
    config['mode']  = 'steady' # adjoint only solves steady-state
   
    # Switching over to the parallel version of the optimization that is found 
    # in the dolfin-adjoint optimize.py file:
    self.maxfun      = config['adjoint']['max_fun']
    self.bounds_list = config['adjoint']['bounds']
    self.control     = config['adjoint']['control_variable']
   
    # ensure that we have lists : 
    if type(config['adjoint']['bounds']) != list:
      config['adjoint']['bounds'] = [config['adjoint']['bounds']]
    if type(config['adjoint']['control_variable']) != list:
      cv = config['adjoint']['control_variable']
      config['adjoint']['control_variable'] = [cv]
    if type(config['adjoint']['alpha']) != list:
      config['adjoint']['alpha'] = [config['adjoint']['alpha']]
    
    # initialize instances of the forward model, and the adjoint physics : 
    self.forward_model    = SteadySolver(model, config)
    if config['use_dukowicz']:
      self.adjoint_instance = AdjointDukowiczVelocity(model, config)
    else:
      self.adjoint_instance = AdjointVelocity(model, config)
        
  def set_target_velocity_from_surface(self, U):
    """ 
    Set target velocity u_ob, v_ob from surface magnitude <U>, going down 
    the surface gradient.
    """
    model  = self.model
    S      = model.S
    Q      = model.Q
    
    Smag   = project(sqrt(S.dx(0)**2 + S.dx(1)**2 + DOLFIN_EPS), Q)
    u_n    = project(-U * S.dx(0) / Smag, Q)
    v_n    = project(-U * S.dx(1) / Smag, Q)      

    model.assign_variable(model.u_ob, u_n)
    model.assign_variable(model.v_ob, v_n)

  def solve(self):
    r""" 
    Perform the optimization.
    """

    def H(c, a_n=None, p_n=None):
      """
      Evaluate the hamiltonian.
      """
      s = '::: evaluating the Hamiltonian :::'
      print_text(s, self.color())
      
      if a_n != None and p_n != None:
        txt = 'H_n'
        c_a = c.vector().array()
        model.assign_variable(c, c_a + a_n*p_n)
        print_min_max(c, 'c')
      else:
        txt = 'H0'
      
      H_n = assemble(self.adjoint_instance.H_lam)
      print_min_max(H_n, txt)
      
      return H_n

    def LS(p_n, c_n, cf):
      """
      Return the step length alpha_n.
      """
      s = '::: performing line-search for step length :::'
      print_text(s, self.color())

      a_n  = 1.0
      rho  = 5/10.
      c    = 10**(-4)
      H0   = H(cf)
      Hn   = H(cf, a_n, p_n)

      #while Hn > H0 or any(c_n - a_n*p_n < 0):
      while Hn > H0 + c*a_n*np.dot(p_n, p_n) or any(c_n + a_n*p_n < 0):
        a_n = rho*a_n
        Hn  = H(cf, a_n, p_n)
        model.assign_variable(cf, c_n)
      MPI.barrier(mpi_comm_world())
      return a_n 

    s  = '::: solving AdjointSolverNew :::'
    print_text(s, self.color())
    
    model       = self.model
    config      = self.config
    bounds_list = self.bounds_list
    c           = self.control
    maxfun      = self.maxfun
    H_lam       = self.adjoint_instance.H_lam
    dHdc        = self.adjoint_instance.dHdc[0]
    
    self.forward_model.solve()
    self.adjoint_instance.solve()
    
    #===========================================================================
    # begin the optimization :
    converged  = False
    atol, rtol = 1e-7, 1e-10           # abs/rel tolerances
    nIter      = 0                     # number of iterations
    residual   = 1                     # residual
    rel_res    = residual              # initial epsilon
    maxIter    = 100                   # max iterations
  
    while not converged and nIter < maxIter:
      nIter  += 1                                # increment interation
      p_n     = assemble(-dHdc)
      print_min_max(p_n, 'dH/dc')
      rel_res = p_n.norm('l2')                   # calculate norm
  
      # calculate residual :
      a = assemble(H_lam)
      #for bc in bcs_u:
      #  bc.apply(a)
      residual  = a
     
      converged = residual < atol or rel_res < rtol

      lmbda = LS(p_n.array(), c.vector().array(), c)
      
      #c.vector()[:] += lmbda*p_n                 # New control vector
      c_v = c.vector().array()
      p_v = p_n.array()
      model.assign_variable(c, c_v + lmbda*p_v)
      print_min_max(c, 'c')
  
      string = "::: Adjoint Newton iteration %d: r (abs) = %.3e (tol = %.3e) " \
               +"r (rel) = %.3e (tol = %.3e) :::"
      print_text(string % (nIter, residual, atol, rel_res, rtol), self.color())
    
      self.forward_model.solve()
      self.adjoint_instance.solve()

    ##==========================================================================
    ## begin the optimization :
    #r         = inf
    #counter   = 0
    #dHdb_norm = inf

    #while dHdb_norm > 1e-10 and counter < maxfun:
    #  
    #  self.forward_model.solve()
    #  self.adjoint_instance.solve()
    #  
    #  pf_a, p_a = p()
    #  c_a       = c()

    #  a_a = []
    #  for p_n, c_n, cf in zip(p_a, c_a, control):
    #    a_a.append(LS(p_n, c_n, cf))
    #  a_a = array(a_a)

    #  c_n = c_a - np.dot(a_a, p_a)

    #  dHdb_norm_a = []
    #  for pii in p_a:
    #    dHdb_norm_a.append(MPI.max(mpi_comm_world(), abs(pii).max()))
    #  dHdb_norm_a = array(dHdb_norm_a)
    #  if len(control) == 1:
    #    dHdb_norm = dHdb_norm_a[0]
    #  else:
    #    dHdb_norm = MPI.max(mpi_comm_world(), abs(dHdb_norm_a))
    #  print_min_max(dHdb_norm, '||dHdb||')

    #  counter += 1
    #  ## Calculate L_infinity norm
    #  ##u_new         = model.u.vector().array()
    #  ##diff          = (u_prev - u_new)
    #  ##inner_error_n = MPI.max(mpi_comm_world(), diff.max())
    #  ##u_prev        = u_new
    #  #inner_error_n = norm(project(U_prev - U))
    #  #U_prev        = U
    #  #if self.model.MPI_rank==0:
    #  #  s1    = 'Picard iteration %i (max %i) done: ' % (counter, max_iter)
    #  #  s2    = 'r0 = %.3e'  % inner_error
    #  #  s3    = ', '
    #  #  s4    = 'r = %.3e ' % inner_error_n
    #  #  s5    = '(tol %.3e)' % inner_tol
    #  #  text1 = colored(s1, 'blue')
    #  #  text2 = colored(s2, 'red', attrs=['bold'])
    #  #  text3 = colored(s3, 'blue')
    #  #  text4 = colored(s4, 'red', attrs=['bold'])
    #  #  text5 = colored(s5, 'blue')
    #  #  print text1 + text2 + text3 + text4 + text5
    #  #inner_error = inner_error_n

    #  #for ci, ci_n in zip(control, c_n):
    #  #  model.assign_variable(ci, ci_n)
    
    # if we've turned off the vert velocity, now we want it :
    if not config['velocity']['solve_vert_velocity']:
      s = '::: re-calculating velocity with vertical solve :::'
      print_text(s, self.color())
      config['velocity']['solve_vert_velocity'] = True
      self.forward_model.solve()

    # save the output :
    s = '::: saving control variable c.pvd file :::'
    print_text(s, self.color())
    File(config['output_path'] + 'c.pvd') << c


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
    
    config['mode']  = 'steady' # adjoint only solves steady-state
   
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
    if config['use_dukowicz']:
      self.adjoint_instance = AdjointDukowiczVelocity(model, config)
    else:
      self.adjoint_instance = AdjointVelocity(model, config)
        
  def set_target_velocity_from_surface(self, U):
    """ 
    Set target velocity u_ob, v_ob from surface magnitude <U>, going down 
    the surface gradient.
    """
    model  = self.model
    S      = model.S
    Q      = model.Q
    
    Smag   = project(sqrt(S.dx(0)**2 + S.dx(1)**2 + DOLFIN_EPS), Q)
    u_n    = project(-U * S.dx(0) / Smag, Q)
    v_n    = project(-U * S.dx(1) / Smag, Q)      

    model.assign_variable(model.u_ob, u_n)
    model.assign_variable(model.v_ob, v_n)

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
        \beta > 0
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
      This is the function to minimize, the cost function.

      Solve forward model with given control, calculate objective function
      """
      n = len(c_array)/len(control)
      for ii,c in enumerate(control):
        set_local_from_global(c, c_array[ii*n:(ii+1)*n])
      self.forward_model.solve()
      s = '::: calculating objective function value :::'
      print_text(s, self.color())
      # save the output :
      if config['adjoint']['objective_function'] == 'log_lin_hybrid':
        J1 = assemble(self.adjoint_instance.J1)
        J2 = assemble(self.adjoint_instance.J2)
        print_min_max(J1, 'J1')
        print_min_max(J2, 'J2')
      R = assemble(self.adjoint_instance.R)
      I = assemble(self.adjoint_instance.I)
      print_min_max(R, 'R')
      print_min_max(I, 'I')
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
      
      # calculate and print misfit : 
      #model.calc_misfit(config['adjoint']['surface_integral'])
      
      s = '::: calc. Gateaux derivative of the Hamiltonian w.r.t. the' + \
          ' control variable(s) :::'
      print_text(s, self.color())

      dHdcs = []
      for i,dHdci in enumerate(self.adjoint_instance.dHdc):
        dHdc = assemble(dHdci)
        print_min_max(dHdc, 'dH/dc%i' % i)
        dHdcs.extend(get_global(dHdc))
      dHdcs = array(dHdcs)
      return dHdcs

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
                               maxiter=maxfun-2, iprint=iprint)
    model.f_adj = f  # save the function value for later use 

    # if we've turned off the vert velocity, now we want it :
    if not config['velocity']['solve_vert_velocity']:
      s = '::: re-calculating velocity with vertical solve :::'
      print_text(s, self.color())
      config['velocity']['solve_vert_velocity'] = True
      self.forward_model.solve()

    n = len(mopt)/len(control)
    for ii,c in enumerate(control):
      set_local_from_global(c, mopt[ii*n:(ii+1)*n])
      
    # save the output :
    for i,c in enumerate(control):
      s = '::: saving control variable %sc%i.pvd file :::'
      print_text(s % (config['output_path'], i), self.color())
      File(config['output_path'] + 'c' + str(i) + '.pvd') << c


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
    
    if config['model_order'] == 'BP':
      self.stress_balance_instance = BP_Balance(model, config)
    elif config['model_order'] == 'SSA':
      self.stress_balance_instance = SSA_Balance(model, config)

  def solve(self):
    """ 
    """
    s    = "::: solving StokesBalanceSolver :::"
    print_text(s, self.color())
   
    model   = self.model
    config  = self.config
    
    # calculate ubar, vbar :
    self.stress_balance_instance.solve()
    if config['log']:
      U_s = as_vector([model.u_s, model.u_t])
      model.save_pvd(project(U_s), 'U_s')
    
    # solve for the stress balance given the appropriate vertically 
    # averaged velocities :
    self.stress_balance_instance.solve_component_stress()
    if config['log']: 
     
      if config['model_order'] == 'BP':
        memb_n   = as_vector([model.F_ii, model.F_ij, model.F_iz])
        memb_t   = as_vector([model.F_ji, model.F_jj, model.F_jz])
        memb_x   = model.F_ii + model.F_ij + model.F_iz
        memb_y   = model.F_ji + model.F_jj + model.F_jz
        membrane = as_vector([memb_x,     memb_y,     0.0])
        driving  = as_vector([model.F_id, model.F_jd, 0.0])
        basal    = as_vector([model.F_ib, model.F_jb, 0.0])
        pressure = as_vector([model.F_ip, model.F_jp, 0.0])
        
        total    = membrane + pressure - driving
        
        # attach the results to the model :
        s    = "::: projecting 'BPBalance' force terms onto vector space :::"
        print_text(s, self.color())
        
        memb_n   = project(memb_n)
        memb_t   = project(memb_t)
        membrane = project(membrane)
        driving  = project(driving)
        basal    = project(basal)
        pressure = project(pressure)
        total    = project(total)
        
        print_min_max(memb_n,    'F_memb_n')
        print_min_max(memb_t,    'F_memb_t')
        print_min_max(membrane,  'F_membrane')
        print_min_max(driving,   'F_driving')
        print_min_max(basal,     'F_basal')
        print_min_max(pressure,  'F_pressure')
        print_min_max(total,     'F_total')
        
        model.save_pvd(memb_n,   'F_memb_n')
        model.save_pvd(memb_t,   'F_memb_t')
        model.save_pvd(membrane, 'F_membrane')
        model.save_pvd(driving,  'F_driving')
        model.save_pvd(basal,    'F_basal')
        model.save_pvd(pressure, 'F_pressure')
        model.save_pvd(total,    'F_total')
        
        if config['stokes_balance']['vert_integrate']: 
          memb_n   = as_vector([model.tau_ii, model.tau_ij, model.tau_iz])
          memb_t   = as_vector([model.tau_ji, model.tau_jj, model.tau_jz])
          memb_x   = model.tau_ii + model.tau_ij + model.tau_iz
          memb_y   = model.tau_ji + model.tau_jj + model.tau_jz
          membrane = as_vector([memb_x,       memb_y,       0.0])
          driving  = as_vector([model.tau_id, model.tau_jd, 0.0])
          basal    = as_vector([model.tau_ib, model.tau_jb, 0.0])
          pressure = as_vector([model.tau_ip, model.tau_jp, 0.0])
          
          total    = membrane + pressure - driving
          
          # attach the results to the model :
          s    = "::: projecting 'BPBalance' stress terms onto vector space :::"
          print_text(s, self.color())
          
          memb_n   = project(memb_n)
          memb_t   = project(memb_t)
          membrane = project(membrane)
          driving  = project(driving)
          basal    = project(basal)
          pressure = project(pressure)
          total    = project(total)
          
          print_min_max(memb_n,   'tau_memb_n')
          print_min_max(memb_t,   'tau_memb_t')
          print_min_max(membrane, 'tau_membrane')
          print_min_max(driving,  'tau_driving')
          print_min_max(basal,    'tau_basal')
          print_min_max(pressure, 'tau_pressure')
          print_min_max(total,    'tau_total')
        
          model.save_pvd(memb_n,   'tau_memb_n')
          model.save_pvd(memb_t,   'tau_memb_t')
          model.save_pvd(membrane, 'tau_membrane')
          model.save_pvd(driving,  'tau_driving')
          model.save_pvd(basal,    'tau_basal')
          model.save_pvd(pressure, 'tau_pressure')
          model.save_pvd(total,    'tau_total')
        
      elif config['model_order'] == 'SSA':
        memb_n   = as_vector([model.tau_ii, model.tau_ij])
        memb_t   = as_vector([model.tau_ji, model.tau_jj])
        memb_x   = model.tau_ii + model.tau_ij
        memb_y   = model.tau_ji + model.tau_jj
        membrane = as_vector([memb_x,       memb_y      ])
        driving  = as_vector([model.tau_id, model.tau_jd])
        basal    = as_vector([model.tau_ib, model.tau_jb])
        
        total    = membrane + basal - driving
        
        # attach the results to the model :
        s    = "::: projecting 'BPBalance' stress terms onto vector space :::"
        print_text(s, self.color())
        
        memb_n   = project(memb_n)
        memb_t   = project(memb_t)
        membrane = project(membrane)
        driving  = project(driving)
        basal    = project(basal)
        total    = project(total)
        
        print_min_max(memb_n,   'tau_memb_n')
        print_min_max(memb_t,   'tau_memb_t')
        print_min_max(membrane, 'tau_membrane')
        print_min_max(driving,  'tau_driving')
        print_min_max(basal,    'tau_basal')
        print_min_max(total,    'tau_total')
        
        model.save_pvd(memb_n,   'tau_memb_n')
        model.save_pvd(memb_t,   'tau_memb_t')
        model.save_pvd(membrane, 'tau_membrane')
        model.save_pvd(driving,  'tau_driving')
        model.save_pvd(basal,    'tau_basal')
        model.save_pvd(total,    'tau_total')
 

class HybridTransientSolver(Solver):
  """
  This abstract class outlines the structure of a VarGlaS solver.
  """
  def __init__(self, model, config):
    """
    """
    s    = "::: INITIALIZING HYBRID TRANSIENT SOLVER :::"
    print_text(s, self.color())
    self.model          = model
    self.config         = config
    self.config['mode'] = 'transient'
    outpath             = config['output_path']

    # initialize velocity solver :
    if self.config['velocity']['on']:
      self.velocity_instance = VelocityHybrid(model, config)
    
    # initialized enthalpy solver : 
    if self.config['enthalpy']['on']:
      self.enthalpy_instance = EnergyHybrid(model, config)

    # initialize surface climate solver :
    if self.config['surface_climate']['on']:
      self.surface_climate_instance = SurfaceClimate(model, config)

    # initialize free surface solver :
    if config['free_surface']['on']:
      self.surface_instance = MassBalanceHybrid(model, config)

    # initialize stress balance solver :
    if config['stokes_balance']['on']:
      self.stokes_balance_instance = SSA_Balance(model, config)
    
    # balance velocity model :
    if config['balance_velocity']['on']:
      self.balance_velocity_instance = VelocityBalance(model, config)

    # Set up files for logging time dependent solutions to paraview files.
    if config['log']:
      self.file_Ubar = File(outpath + 'Ubar.pvd')
      self.file_U_s  = File(outpath + 'Us.pvd')
      self.file_U_b  = File(outpath + 'Ub.pvd')
      self.file_Ts   = File(outpath + 'Ts.pvd')
      self.file_Tb   = File(outpath + 'Tb.pvd')
      self.file_Mb   = File(outpath + 'Mb.pvd')
      self.file_H    = File(outpath + 'H.pvd')
      self.file_beta = File(outpath + 'beta.pvd')
      self.file_tau_ii = File(outpath + 'tau_ii.pvd')
      self.file_tau_ij = File(outpath + 'tau_ij.pvd')
      self.file_tau_jj = File(outpath + 'tau_jj.pvd')
      self.file_tau_ji = File(outpath + 'tau_ji.pvd')
      self.file_tau_id = File(outpath + 'tau_id.pvd')
      self.file_tau_jd = File(outpath + 'tau_jd.pvd')
      self.t_log     = []

    self.step_time = []
    self.M_prev    = 1.0

  def solve(self):
    """
    Performs the physics, evaluating and updating the enthalpy and age as 
    well as storing the velocity, temperature, and the age in vtk files.

    """
    s    = '::: solving HybridTransientSolver :::'
    print_text(s, self.color())
    model   = self.model
    config  = self.config
    outpath = config['output_path']
    
    t      = config['t_start']
    t_end  = config['t_end']
    dt     = config['time_step']
   
    # Loop over all times
    while t <= t_end:

      # start the timer :
      tic = time()
       
      # calculate velocity : 
      if config['velocity']['on']:
        self.velocity_instance.solve()
        if config['velocity']['log']:
          s    = '::: saving surface and bed velocity Us and Ub .pvd' + \
                 ' files to %s :::' % outpath
          print_text(s, self.color())
          U_s  = project(as_vector([model.u_s, model.v_s, model.w_s]))
          U_b  = project(as_vector([model.u_b, model.v_b, model.w_b]))
          if config['log_history']:
            self.file_U_s << U_s
            self.file_U_b << U_b
          else:
            File(outpath + 'Us.pvd')  << U_s
            File(outpath + 'Ub.pvd')  << U_b

      # calculate energy
      if config['enthalpy']['on']:
        #if 0 < t and t <= 100.0:
        #  s    = '::: updating surface temperature :::'
        #  print_text(s, self.color())
        #  T_v  = model.T_surface.vector().array()
        #  T_v += 5.0 / 100.0 * dt
        #  model.assign_variable(model.T_surface, T_v)
        #  print_min_max(model.T_surface, 'T_surface')
        self.enthalpy_instance.solve()
        if config['enthalpy']['log']:
          s    = '::: saving surface and bed temperature Ts, Tb, and Mb' + \
                 ' .pvd files to %s :::' % outpath
          print_text(s, self.color())
          if config['log_history']:
            self.file_Ts << model.Ts
            self.file_Tb << model.Tb
            self.file_Mb << model.Mb
          else:
            File(outpath + 'Ts.pvd')  << model.Ts
            File(outpath + 'Tb.pvd')  << model.Tb
            File(outpath + 'Mb.pvd')  << model.Mb
        model.T0_.interpolate(model.T_)  # update previous temp
      
      # calculate free surface :
      if config['free_surface']['on']:
        self.surface_instance.solve()
        if config['log']:
          s    = '::: saving thickness %sH.pvd file :::' % outpath
          print_text(s, self.color())
          if config['log_history']:
            self.file_H << model.H
          else:
            File(outpath + 'H.pvd') << model.H
        model.H0.interpolate(model.H)
    
      # calculate surface climate solver :
      if config['surface_climate']['on']:
        self.surface_climate_instance.solve()
    
      # balance velocity model :
      if config['balance_velocity']['on']:
        self.balance_velocity_instance.solve()
        if config['log']:
          s    = '::: saving balance velocity %sUbar.pvd file :::' % outpath
          print_text(s, self.color())
          if config['log_history']:
            self.file_Ubar << model.Ubar
          else:
            File(outpath + 'Ubar.pvd') << model.Ubar
     
      # solve the stress-balance :   
      if config['stokes_balance']['on']:
        self.stokes_balance_instance.solve()
        if config['log']:
          s    = '::: saving stress terms tau_ii, tau_ij,' \
                 ' tau_jj, tau_ji, tau_id, and tau_jd .pvd files to %s :::'
          print_text(s % outpath , self.color())
          if config['log_history']:
            self.file_tau_ii << model.tau_ii
            self.file_tau_ij << model.tau_ij
            self.file_tau_jj << model.tau_jj
            self.file_tau_ji << model.tau_ji
            self.file_tau_id << model.tau_id
            self.file_tau_jd << model.tau_jd
          else:
            File(outpath + 'tau_ii.pvd') << model.tau_ii
            File(outpath + 'tau_ij.pvd') << model.tau_ij
            File(outpath + 'tau_jj.pvd') << model.tau_jj
            File(outpath + 'tau_ji.pvd') << model.tau_ji
            File(outpath + 'tau_id.pvd') << model.tau_id
            File(outpath + 'tau_jd.pvd') << model.tau_jd
      
      # re-compute the friction field :
      if config['velocity']['transient_beta'] == 'stats':
        s    = "::: calculating new statistical beta :::"
        print_text(s, self.color())

        model.assign_variable(model.u, model.u_b)
        model.assign_variable(model.v, model.v_b)
        model.assign_variable(model.w, model.w_b)
        model.assign_variable(model.T, model.Tb)

        beta = project(model.beta_f, model.Q)
        print_min_max(beta, 'beta')
        s    = "::: removing negative values of beta :::"
        print_text(s, self.color())
        beta_v = beta.vector().array()
        #betaSIA_v = model.betaSIA.vector().array()
        #beta_v[beta_v < 10.0]   = betaSIA_v[beta_v < 10.0]
        beta_v[beta_v < sqrt(1e3)]  = sqrt(1e3)
        #beta_v[beta_v < 1e-2]  = 1e-2
        beta_v[beta_v > sqrt(1e9)]  = sqrt(1e9)
        model.assign_variable(model.beta, beta_v)
        #model.assign_variable(model.beta, np.sqrt(beta_v))
        print_min_max(model.beta, 'beta')

        Ubar_v = model.Ubar.vector().array()
        #Ubar_v[Ubar_v < 0]    = 0
        #Ubar_v[Ubar_v > 3000] = 3000
        model.assign_variable(model.Ubar, Ubar_v)

        # save beta : 
        if config['log']:
          s    = '::: saving stats %sbeta.pvd file :::' % outpath
          print_text(s, self.color())
          if config['log_history']:
            self.file_beta << model.beta
          else:
            File(outpath + 'beta.pvd') << model.beta
      
      elif config['velocity']['transient_beta'] == 'eismint_H':
        s    = "::: calculating new beta from pressure melting point :::"
        print_text(s, self.color())
        T_tol = 1.0
        beta = model.beta.vector()
        Tb   = model.Tb.vector()
        Tb_m = model.T_melt.vector()

        beta[Tb >  (Tb_m - T_tol)] = sqrt(1e3)
        beta[Tb <= (Tb_m - T_tol)] = sqrt(1e9)
        print_min_max(model.beta, 'beta')
        # save beta : 
        if config['log']:
          s    = '::: saving updated %sbeta.pvd file :::' % outpath
          print_text(s, self.color())
          if config['log_history']:
            self.file_beta << model.beta
          else:
            File(outpath + 'beta.pvd') << model.beta

      # store information : 
      if self.config['log']:
        self.t_log.append(t)

      if t % 100 == 0:
        model.save_xml(model.S,    'S_%i' % t)
        model.save_xml(model.H,    'H_%i' % t)
        model.save_xml(model.u_s,  'u_s_%i' % t)
        model.save_xml(model.v_s,  'v_s_%i' % t)
        model.save_xml(model.w_s,  'w_s_%i' % t)
        model.save_xml(model.beta, 'beta_%i' % t)

      # increment time step :
      if self.model.MPI_rank==0:
        s = '>>> Time: %i yr, CPU time for last dt: %.3f s <<<'
        text = colored(s, 'red', attrs=['bold'])
        print text % (t, time()-tic)

      t += dt
      self.step_time.append(time() - tic)



