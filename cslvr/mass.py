from cslvr.physics     import Physics 
from cslvr.inputoutput import print_text, print_min_max
from cslvr.d2model     import D2Model
from cslvr.d3model     import D3Model
from cslvr.helper      import VerticalBasis, VerticalFDBasis, \
                              VerticalIntegrator
from fenics            import *
from dolfin_adjoint    import *

class Mass(Physics):
  """
  Abstract class outlines the structure of a mass conservation.
  """

  def __new__(self, model, *args, **kwargs):
    """
    Creates and returns a new Mass object.
    """
    instance = Physics.__new__(self, model)
    return instance
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'white'
  
  def solve(self, annotate=True, params=None):
    """ 
    Solve the conservation of mass equation for a free-surface evolution.
    """
    raiseNotDefined()


class FreeSurface(Mass):
  """
  """ 
  # TODO: think about how to make this usable with D2Model.
  def __init_(self, model, thklim,
              static_boundary_conditions = False,
              use_shock_capturing        = False,
              lump_mass_matrix           = False):
    """
    """
    print_text("::: INITIALIZING FREE-SURFACE PHYSICS :::", self.color())
    
    if type(model) != D3Model:
      s = ">>> FreeSurface REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
   
    self.thklim                     = thklim 
    self.static_boundary_conditions = static_boundary_conditions
    self.use_shock_capturing        = use_shock_capturing
    self.lump_mass_matrix           = lump_mass_matrix

    # get the dof map :
    if periodic_boundary_conditions:
      self.d2v      = dof_to_vertex_map(model.Q_non_periodic)
      self.mhat_non = Function(model.Q_non_periodic)
    else:
      self.d2v      = dof_to_vertex_map(model.Q)

    Q       = model.Q
    Q_flat  = model.Q_flat
    ff_flat = model.ff_flat
    ds      = model.ds_flat
    dSrf    = model.dSrf
    dBed    = model.dBed
    
    S       = Function(Q_flat) # surface elevation
    adot    = Function(Q_flat) # accumulation
    u       = Function(Q_flat) # x-component of velocity
    v       = Function(Q_flat) # y-component of velocity
    w       = Function(Q_flat) # z-component of velocity
    m       = Function(Q_flat) # mesh velocity (can only be in the z direction)
    rho     = Function(self.Q) # for calculating the ice-sheet mass
    h       = CellSize(model.flat_mesh)

    # self.m when assembled gives the mass of the domain :
    rho.vector()[:] = model.rhoi
    self.M_tot      = rho * dx

    # for regions when constant surface, dSdt = 0 :    
    self.bcs = []
    self.bcs.append(DirichletBC(Q, 0.0, ff_flat, self.GAMMA_S_GND))  # grounded
    self.bcs.append(DirichletBC(Q, 0.0, ff_flat, self.GAMMA_S_FLT))  # shelves
    self.bcs.append(DirichletBC(Q, 0.0, ff_flat, self.GAMMA_U_GND))  # grounded
    self.bcs.append(DirichletBC(Q, 0.0, ff_flat, self.GAMMA_U_FLT))  # shelves

    # set up linear variational problem for unknown dSdt :
    phi         = TestFunction(Q_flat)
    psi         = TrialFunction(Q_flat)
    dSdt        = Function(Q_flat)

    # SUPG-modified trial function (artifical diffusion in direction of flow) :
    U           = as_vector([u, v, w])
    unorm       = sqrt(dot(U,U) + 1e-1)
    phihat      = phi + h / (2.0*unorm) * dot(U, grad(phi))

    # mass matrices :
    mass        = psi * phihat * dSrf
    lumped_mass = phi * dSrf

    # stiffness matrix :
    stiffness   = + adot * phi * dSrf \
                  + w * phi * dSrf \
                  - dot(U, grad(S)) * phihat * dSrf
   
    # shock-capturing steady-state residual dependent artificial diffusion :
    S_mag       = dot(grad(S), grad(S)) + 1e-1
    resid       = dot(U, grad(S)) - w - adot
    C           = 1e1 * h / (2*unorm) * S_mag * resid**2
    diffusion   = C * dot(grad(phi), grad(S)) * dSrf
    
    # set up the Galerkin/least-squares formulation of the Stokes functional :
    self.A_pro  = dSdt*phi*dSrf - psi*phi*dBed - psi*phi.dx(2)*dx

    self.stiffness    = stiffness
    self.diffusion    = diffusion
    self.mass         = mass
    self.lumped_mass  = lumped_mass
    self.Shat         = S
    self.ahat         = adot
    self.uhat         = u
    self.vhat         = v
    self.what         = w
    self.mhat         = m
    self.dSdt         = dSdt

  def solve(self):
    """
    """
    print_text("::: solving free-surface relation :::", self.color())
    
    model  = self.model
    config = self.config
  
    # assign the current values from the deformed mesh to the sigma-coord mesh :
    self.assign_variable(self.Shat, model.S) 
    self.assign_variable(self.ahat, model.adot) 
    self.assign_variable(self.uhat, model.u) 
    self.assign_variable(self.vhat, model.v) 
    self.assign_variable(self.what, model.w) 

    # assemple the stiffness matrix :
    K = assemble(self.stiffness, keep_diagonal=True)

    # apply static boundary conditions, if desired :
    if self.static_boundary_conditions:
      for bc in self.bcs: bc.apply(K)

    if self.use_shock_capturing:
      D = assemble(self.diffusion)
      K = K - D
      print_min_max(D, 'D')
      print_min_max(K, 'K')

    # calculate preliminary guess :
    if self.lump_mass_matrix:
      M               = assemble(self.lumped_mass)
      M_a             = M.get_local()
      M_a[M_a == 0.0] = 1.0
      self.assign_variable(self.dSdt, K.get_local() / M_a)
    else:
      M = assemble(self.mass, keep_diagonal=True)
      if self.static_boundary_conditions:
        for bc in self.bcs: bc.apply(M)
      M.ident_zeros()
      solve(M, self.dSdt.vector(), K, annotate=False)
   
    # calculate GLS system :
    A = assemble(lhs(self.A_pro))
    p = assemble(rhs(self.A_pro))
    q = Vector()
    solve(A, q, p, annotate=False)
    self.assign_variable(self.dSdt, q)

  def rhs_func_explicit(self, t, S, *f_args):
    """
    This function calculates the change in height of the surface of the
    ice sheet.
    
    :param t : Time
    :param S : Current height of the ice sheet
    :rtype   : Array containing rate of change of the ice surface values
    """
    # TODO: move this function into Model class timestepping function
    model   = self.model
    d2v     = self.d2v
    thklim  = self.thklim
    B       = model.B.compute_vertex_values()

    # impose the thickness limit :
    S[(S-B) < thklim] = thklim + B[(S-B) < thklim]
    model.assign_variable(model.S, S[d2v])
   
    if config['velocity']['on']:
      model.U.vector()[:] = 0.0
      self.velocity_instance.solve()
      print_min_max(U, 'U')

    if config['surface_climate']['on']:
      self.surface_climate_instance.solve()
   
    if config['free_surface']['on']:
      self.solve()
      print_min_max(model.S, 'S')
 
    return self.dSdt.compute_vertex_values()

  def really_solve(self):
    """
    Performs the physics, evaluating and updating the enthalpy and age as 
    well as storing the velocity, temperature, and the age in vtk files.
    """
    # TODO: move this function into Model class timestepping function
    print_text('::: solving TransientSolver :::', self.color())

    model  = self.model
    
    t      = config['t_start']
    t_end  = config['t_end']
    dt     = config['time_step'](0)
    thklim = config['free_surface']['thklim']
   
    mesh   = model.mesh 
    adot   = model.adot
    S      = model.S
    B      = model.B

    # history of the the total mass of the domain (iteration `k = 0`:
    m_tot_k           = assemble(self.M_tot) 
    self.mass_history = [m_tot_k]

    # Loop over all times
    while t <= t_end:

      # get time :
      t0 = time()

      # get nodal values :
      B_a                      = B.compute_vertex_values()
      S_a                      = S.compute_vertex_values()

      # Runga-Kutta method ?
      S_0                      = S_a
      f_0                      = self.rhs_func_explicit(t, S_0)
      S_1                      = S_0 + dt*f_0
      S_1[(S_1-B_a) < thklim]  = thklim + B_a[(S_1-B_a) < thklim]
      model.assign_variable(S, S_1[d2v])

      f_1                      = self.rhs_func_explicit(t, S_1)
      S_2                      = 0.5*S_0 + 0.5*S_1 + 0.5*dt*f_1
      S_2[(S_2-B_a) < thklim]  = thklim + B_a[(S_2-B_a) < thklim] 
      model.assign_variable(S, S_2[d2v])
    
      # adjust the z coordinate of the mesh vertices : 
      z_a                      = mesh.coordinates()[:, 2]
      S_a                      = S.compute_vertex_values()
      sigma                    = (z_a - B_a) / (S_a - B_a)
      
      mesh.coordinates()[:, 2] = sigma * (S_2 - B_a) + B_a

      # calculate mesh velocity :
      if self.periodic_boundary_conditions:
        temp = (S_2[d2v] - S_0[d2v]) / dt * sigma[d2v]
        model.assign_variable(self.mhat_non, temp)
        
        # FIXME: projection not required :
        m_temp = project(self.mhat_non, model.Q)
        model.assign_variable(model.mhat, m_temp)
      else:
        temp = (S_2[d2v] - S_0[d2v])/dt * sigma[d2v]
        model.assign_variable(model.mhat, temp)

      # calculate enthalpy update :
      if self.config['enthalpy']['on']:
        self.enthalpy_instance.solve(H0=model.H, Hhat=model.H, uhat=model.u, 
                                   vhat=model.v, what=model.w, mhat=model.mhat)
        print_min_max(model.H,  'H')
        print_min_max(model.T,  'T')
        print_min_max(model.Mb, 'Mb')
        print_min_max(model.W,  'W')

      # calculate age update :
      if self.config['age']['on']:
        self.age_instance.solve(A0=model.A, Ahat=model.A, uhat=model.u, 
                                vhat=model.v, what=model.w, mhat=model.mhat)
        print_min_max(model.age, 'age')

      # store information : 
      m_tot = assemble(self.M_tot)
      self.mass_history.append(m_tot)

      # print statistics :
      s = '>>> sim time: %g yr, CPU time: %g s, mass m_t / m_t-1: %g <<<' \
          % (t, time()-t0, m_tot / m_tot_k)
      print_text(s, 'red', 1)

      # increment :
      m_tot_k = m_tot
      t      += dt


class MassHybrid(Mass):
  """
  New 2D hybrid model.
  """
  def __init__(self, model, thklim=1.0, solve_params=None, isothermal=True):
    """
    """
    s = "::: INITIALIZING HYBRID MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D2Model:
      s = ">>> MassHybrid REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params
    
    # CONSTANTS
    year   = model.spy
    rho    = model.rhoi
    g      = model.g
    n      = model.n(0)
    
    Q      = model.Q
    B      = model.B
    beta   = model.beta
    adot   = model.adot
    ubar_c = model.ubar_c 
    vbar_c = model.vbar_c
    H      = model.H
    H0     = model.H0
    U      = model.UHV
    T_     = model.T_
    deltax = model.deltax
    sigmas = model.sigmas
    h      = model.h
    dt     = model.time_step
    S      = B + H
    coef   = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    T      = VerticalFDBasis(T_, deltax, coef, sigmas)
    
    Bc    = 3.61e-13*year
    Bw    = 1.73e3*year #model.a0 ice hardness
    Qc    = 6e4
    Qw    = model.Q0 # ice act. energy
    Rc    = model.R  # gas constant
   
    # function spaces : 
    dH  = TrialFunction(Q)
    xsi = TestFunction(Q)

    if isothermal:
      s = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      def A_v(T):
        return model.b**(-n)
    else:
      s = "    - using temperature-dependent rate-factor -"
      print_text(s, self.color())
      def A_v(T):
        return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))
    
    # SIA DIFFUSION COEFFICIENT INTEGRAL TERM.
    def sia_int(s):
      return A_v(T.eval(s)) * s**(n+1)
    
    vi = VerticalIntegrator(order=4)
    
    #D = 2 * (rho*g)**n * A/(n+2) * H**(n+2) \
    #      * dot(grad(S),grad(S))**((n-1)/2)
    D = + 2 * (rho*g)**n * H**(n+2) \
            * dot(grad(S),grad(S))**((n-1)/2) \
            * vi.intz(sia_int) \
        + rho * g * H**2 / beta
    
    ubar = U[0]
    vbar = U[1]
    
    ubar_si = -D/H*S.dx(0)
    vbar_si = -D/H*S.dx(1)
    
    self.ubar_proj = (ubar-ubar_si)*xsi*dx
    self.vbar_proj = (vbar-vbar_si)*xsi*dx

    # mass term :
    self.m  = dH*xsi*dx
    
    # residual :
    self.R_thick = + (H-H0) / dt * xsi * dx \
                   + D * dot(grad(S), grad(xsi)) * dx \
                   + (Dx(ubar_c*H,0) + Dx(vbar_c*H,1)) * xsi * dx \
                   - adot * xsi * dx

    # Jacobian :
    self.J_thick = derivative(self.R_thick, H, dH)

    self.bc = []#NOTE ? DirichletBC(Q, thklim, 'on_boundary') ? maybe ?
    self.bc = [DirichletBC(Q, thklim, 'on_boundary')]
   
    # create solver for the problem : 
    problem = NonlinearVariationalProblem(self.R_thick, model.H, 
                J=self.J_thick, bcs=self.bc,
                form_compiler_parameters=self.solve_params['ffc_params'])
    problem.set_bounds(model.H_min, model.H_max)
    self.solver = NonlinearVariationalSolver(problem)
    self.solver.parameters.update(self.solve_params['solver'])

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def default_ffc_options(self):
    """ 
    Returns a set of default ffc options that yield good performance
    """
    ffc_options = {"optimize"               : True,
                   "eliminate_zeros"        : True,
                   "precompute_basis_const" : True,
                   "precompute_ip_const"    : True}
    return ffc_options
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'nonlinear_solver' : 'snes',
               'snes_solver'      : {'method'                  : 'vinewtonrsls',
                                     'linear_solver'           : 'mumps',
                                     'relative_tolerance'      : 1e-6,
                                     'absolute_tolerance'      : 1e-6,
                                     'maximum_iterations'      : 20,
                                     'error_on_nonconvergence' : False,
                                     'report'                  : True}}
    m_params  = {'solver'      : nparams,
                 'ffc_params'  : self.default_ffc_options()}
    return m_params

  def solve(self, annotate=True):
    """
    Solves for hybrid conservation of mass.
    """
    model  = self.model
    params = self.solve_params

    # find corrective velocities :
    s    = "::: solving for corrective velocities :::"
    print_text(s, self.color())

    solve(self.m == self.ubar_proj, model.ubar_c,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=params['ffc_params'],
          annotate=annotate)

    solve(self.m == self.vbar_proj, model.vbar_c,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=params['ffc_params'],
          annotate=annotate)

    print_min_max(model.ubar_c, 'ubar_c')
    print_min_max(model.vbar_c, 'vbar_c')

    # SOLVE MASS CONSERVATION bounded by (H_max, H_min) :
    meth   = params['solver']['snes_solver']['method']
    maxit  = params['solver']['snes_solver']['maximum_iterations']
    s      = "::: solving 'MassTransportHybrid' using method '%s' with %i " + \
             "max iterations :::"
    print_text(s % (meth, maxit), self.color())
   
    # define variational solver for the mass problem :
    #p = NonlinearVariationalProblem(self.R_thick, model.H, J=self.J_thick,
    #      bcs=self.bc, form_compiler_parameters=params['ffc_params'])
    #p.set_bounds(model.H_min, model.H_max)
    #s = NonlinearVariationalSolver(p)
    #s.parameters.update(params['solver'])
    out = self.solver.solve(annotate=annotate)
    
    print_min_max(model.H, 'H')
    
    # update previous time step's H :
    model.assign_variable(model.H0, model.H)
    
    # update the surface :
    s    = "::: updating surface :::"
    print_text(s, self.color())
    B_v = model.B.vector().array()
    H_v = model.H.vector().array()
    S_v = B_v + H_v
    model.assign_variable(model.S, S_v)

    return out


class FirnMass(Mass):

  def __init__(self, model):
    """
    """
    s = "::: INITIALIZING FIRN MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D1Model:
      s = ">>> FirnMass REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

  def solve(self):
    """
    If conserving the mass of the firn column, calculate height of each 
    interval :
    """
    model  = self.model

    zOld   = model.z
    lnew   = append(0, model.lini) * model.rhoin / model.rhop
    zSum   = model.B
    zNew   = zeros(model.n)
    for i in range(model.n):
      zNew[i]  = zSum + lnew[i]
      zSum    += lnew[i]
    model.z    = zNew
    model.l    = lnew[1:]
    model.mp   = -(zNew - zOld) / model.dt
    model.lnew = lnew
    
    model.assign_variable(model.m_1, model.m)
    model.assign_variable(model.m,   model.mp)
    model.mesh.coordinates()[:,0][model.index] = model.z # update the mesh coor
    model.mesh.bounding_box_tree().build(model.mesh)     # rebuild the mesh tree



