from pylab          import *
from dolfin         import *
from physics        import *
from scipy.optimize import fmin_l_bfgs_b
from helper         import print_min_max

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
        print_min_max(U, 'U')

      # Solve enthalpy (temperature, water content)
      if config['enthalpy']['on']:
        self.enthalpy_instance.solve()
        if config['log']: 
          File(outpath + 'T.pvd')  << model.T   # save temperature
          File(outpath + 'Mb.pvd') << model.Mb  # save basal water content
        print_min_max(model.T, 'T')

      # Calculate L_infinity norm
      if config['coupled']['on']:
        u_new       = project(model.u, model.Q).vector().array()
        diff        = (u_prev - u_new)
        inner_error = diff.max()
        u_prev      = u_new
      
      counter += 1
      
      print 'Picard iteration %i (max %i) done: r = %.3e (tol %.3e)' \
            % (counter, max_iter, inner_error, inner_tol)


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
    if config['periodic_boundary_conditions']:
      #v2d = model.Q_non_periodic.dofmap().vertex_to_dof_map(model.flat_mesh)
      v2d = vertex_to_dof_map(model.Q_non_periodic)
    else:
      #v2d = model.Q.dofmap().vertex_to_dof_map(model.flat_mesh)
      v2d = vertex_to_dof_map(model.Q)
    model.S.vector().set_local(S[v2d])
    model.S.vector().apply('')
   
    if config['velocity']['on']:
      utemp = model.U.vector().get_local()
      utemp[:] = 0.0
      model.U.vector().set_local(utemp)
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

    S      = model.S
    B      = model.B
 
    smb.interpolate(config['free_surface']['observed_smb'])

    import time
    if config['periodic_boundary_conditions']:
      #v2d = model.Q_non_periodic.dofmap().vertex_to_dof_map(model.mesh)
      v2d = vertex_to_dof_map(model.Q_non_periodic)
      mhat_non = Function(model.Q_non_periodic)
    else:
      v2d = vertex_to_dof_map(model.Q)

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
      S.vector().apply('')

      f_1                     = self.rhs_func_explicit(t, S_1)
      S_2                     = 0.5*S_0 + 0.5*S_1 + 0.5*dt*f_1
      S_2[(S_2-B_a) < thklim] = thklim + B_a[(S_2-B_a) < thklim] 
      S.vector().set_local(S_2[v2d])
      S.vector().apply('')
     
      mesh.coordinates()[:, 2] = sigma.compute_vertex_values()*(S_2 - B_a) + B_a
      if config['periodic_boundary_conditions']:
        temp = (S_2[v2d] - S_0[v2d])/dt * sigma.vector().get_local()
        mhat_non.vector().set_local(temp)
        mhat_non.vector().apply('')
        m_temp = project(mhat_non,model.Q)
        mhat.vector().set_local(m_temp.vector().get_local())
        mhat.vector().apply('')
      else:
        temp = (S_2[v2d] - S_0[v2d])/dt * sigma.vector().get_local()
        model.mhat.vector().set_local(temp)
        model.mhat.vector().apply('')
      # Calculate enthalpy update
      if self.config['enthalpy']['on']:
        self.enthalpy_instance.solve(H0=model.H, Hhat=model.H, uhat=model.u, 
                                   vhat=model.v, what=model.w, mhat=model.mhat)

      # Calculate age update
      if self.config['age']['on']:
        self.age_instance.solve(A0=model.A, Ahat=model.A, uhat=model.u, 
                                vhat=model.v, what=model.w, mhat=model.mhat)

      # Store velocity, temperature, and age to vtk files
      if self.config['log']:
        U = project(as_vector([model.u, model.v, model.w]))
        self.file_U << (U, t)
        self.file_T << (model.T, t)
        self.file_S << (model.S, t)
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
      Smag   = project( sqrt(S.dx(0)**2 + S.dx(1)**2 + 1e-10), Q )
      model.U_o.interpolate(U)
      model.u_o = project( -model.U_o * S.dx(0) / Smag, Q )
      model.v_o = project( -model.U_o * S.dx(1) / Smag, Q )      

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

    def _I_fun(c_array, *args):
      """
      Solve forward model with given control, calculate objective function
      """
      n = len(c_array)/len(config['adjoint']['control_variable'])
      for ii,c in enumerate(config['adjoint']['control_variable']):
        set_local_from_global(c, c_array[ii*n:(ii+1)*n])
      self.forward_model.solve()
      I = assemble(self.adjoint_instance.I)
      return I
 
    def _J_fun(c_array, *args):
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
      # FIXME: project and extrude ruin the output for paraview
      #U    = project(as_vector([model.u, model.v, model.w]))
      #dSdt = project(- (model.u*model.S.dx(0) + model.v*model.S.dx(1)) \
      #               + model.w + model.adot)
      #file_b_pvd    << model.extrude(model.beta2, 3, 2)
      #file_u_pvd    << U
      #file_dSdt_pvd << dSdt
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
    m_global    = []
    for mm in config['adjoint']['control_variable']:
      m_global.extend(get_global(mm))
    m_global = array(m_global)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
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
    print bounds
    
    # minimize this stuff :
    mopt, f, d = fmin_l_bfgs_b(_I_fun, m_global, fprime=_J_fun, bounds=bounds,
                               maxfun=maxfun, iprint=iprint)

    n = len(mopt)/len(config['adjoint']['control_variable'])
    for ii,c in enumerate(config['adjoint']['control_variable']):
      set_local_from_global(c, mopt[ii*n:(ii+1)*n])
      
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
      
      print 'Picard iteration %i (max %i) done: r = %.3e (tol %.3e)' \
            % (counter, max_iter, inner_error, inner_tol)

  def component_stress_stokes(self):  
    """
    """
    print "solving 'stokes-balance' for stress terms :::" 
    model = self.model

    outpath = self.config['output_path']
    Q       = self.Q
    S       = model.S
    B       = model.B
    H       = S - B
    etabar  = model.etabar
    
    #===========================================================================
    # form the stokes equations in the normal direction (n) and tangential 
    # direction (t) in relation to the stress-tensor :
    u_s = project(model.ubar, Q)
    v_s = project(model.vbar, Q)
    
    U   = model.normalize_vector(as_vector([u_s, v_s]), Q)
    u_n = U[0]
    v_n = U[1]
    U   = as_vector([u_s, v_s, 0])
    U_n = as_vector([u_n, v_n, 0])
    U_t = as_vector([v_n,-u_n, 0])

    # directional derivatives :
    uhat     = dot(U, U_n)
    vhat     = dot(U, U_t)
    graduhat = grad(uhat)
    gradvhat = grad(vhat)
    dudn     = dot(graduhat, U_n)
    dvdn     = dot(gradvhat, U_n)
    dudt     = dot(graduhat, U_t)
    dvdt     = dot(gradvhat, U_t)

    # get driving stress and basal drag : 
    tau_d = model.tau_d
    tau_b = model.tau_b
    
    # trial and test functions for linear solve :
    phi   = TestFunction(Q)
    dtau  = TrialFunction(Q)
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # integration by parts directional derivative terms :
    gradphi = grad(phi)
    dphidn  = dot(gradphi, U_n)
    dphidt  = dot(gradphi, U_t)
    
    # stokes equation weak form in normal dir. (n) and tangent dir. (t) :
    tau_nn = - dphidn * H * etabar * (4*dudn + 2*dvdt) * dx
    tau_nt = - dphidt * H * etabar * (  dudt +   dvdn) * dx
    tau_tn = - dphidn * H * etabar * (  dudt +   dvdn) * dx
    tau_tt = - dphidt * H * etabar * (4*dvdt + 2*dudn) * dx
    
    # dot product of stress with the direction along (n) and across (t) flow :
    tau_bn = phi * dot(tau_b, U_n) * dx
    tau_dn = phi * dot(tau_d, U_n) * dx
    tau_bt = phi * dot(tau_b, U_t) * dx
    tau_dt = phi * dot(tau_d, U_t) * dx
    
    # the residuals :
    tau_totn = tau_nn + tau_tn - tau_bn - tau_dn
    tau_tott = tau_nt + tau_tt - tau_bt - tau_dt

    # assemble the vectors :
    tau_nn_v   = assemble(tau_nn)
    tau_nt_v   = assemble(tau_nt)
    tau_tn_v   = assemble(tau_tn)
    tau_tt_v   = assemble(tau_tt)
    tau_totn_v = assemble(tau_totn)
    tau_tott_v = assemble(tau_tott)
    
    # solution functions :
    tau_nn   = Function(Q)
    tau_nt   = Function(Q)
    tau_tn   = Function(Q)
    tau_tt   = Function(Q)
    tau_totn = Function(Q)
    tau_tott = Function(Q)
    
    # solve the linear system :
    solve(M, tau_nn.vector(),   tau_nn_v)
    solve(M, tau_nt.vector(),   tau_nt_v)
    solve(M, tau_tn.vector(),   tau_tn_v)
    solve(M, tau_tt.vector(),   tau_tt_v)
    solve(M, tau_totn.vector(), tau_totn_v)
    solve(M, tau_tott.vector(), tau_tott_v)

    # give the stress balance terms :
    tau_bn = project(dot(tau_b, U_n))
    tau_dn = project(dot(tau_d, U_n))

    # output the files to the specified directory :
    File(outpath + 'tau_dn.pvd')   << tau_dn
    File(outpath + 'tau_bn.pvd')   << tau_bn
    File(outpath + 'tau_nn.pvd')   << tau_nn
    File(outpath + 'tau_nt.pvd')   << tau_nt
    File(outpath + 'tau_totn.pvd') << tau_totn
    File(outpath + 'tau_tott.pvd') << tau_tott
    File(outpath + 'u_s.pvd')      << u_s
    File(outpath + 'v_s.pvd')      << v_s
   
    # return the functions for further analysis :
    return tau_nn, tau_nt, tau_bn, tau_dn, tau_totn, tau_tott, u_s, v_s



