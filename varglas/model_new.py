from fenics         import *
from dolfin_adjoint import *
from varglas.io     import print_text, get_text, print_min_max
import numpy        as np
import sys

class Model(object):
  """ 
  Instance of a 2D flowline ice model that contains geometric and scalar 
  parameters and supporting functions.  This class does not contain actual 
  physics but rather the interface to use physics in different simulation 
  types.
  """
    
  GAMMA_S_GND = 2   # grounded upper surface
  GAMMA_B_GND = 3   # grounded lower surface (bedrock)
  GAMMA_S_FLT = 6   # shelf upper surface
  GAMMA_B_FLT = 5   # shelf lower surface
  GAMMA_D     = 7   # basin divides
  GAMMA_T     = 4   # terminus
  GAMMA_U_GND = 8   # grounded surface with U observations
  GAMMA_U_FLT = 9   # shelf surface with U observations

  def __init__(self, mesh, out_dir='./results/', save_state=False, 
               state=None, use_periodic=False, **gfs_kwargs):
    """
    Create and instance of the model.
    """
    self.model_color = '148'
    
    s = "::: INITIALIZING BASE MODEL :::"
    print_text(s, self.model_color)
    
    parameters['form_compiler']['quadrature_degree']  = 2
    parameters["std_out_all_processes"]               = False
    parameters['form_compiler']['cpp_optimize']       = True

    PETScOptions.set("mat_mumps_icntl_14", 100.0)

    self.out_dir     = out_dir
    self.save_state  = save_state
    self.MPI_rank    = MPI.rank(mpi_comm_world())
    self.use_periodic_boundaries = use_periodic
    
    self.generate_constants()
    self.set_mesh(mesh)
    self.generate_function_spaces(use_periodic, **gfs_kwargs)
    self.initialize_variables()

    # create a new state called "state.h5" :
    if save_state and state == None:
      self.state = HDF5File(self.mesh.mpi_comm(), out_dir + 'state.h5', 'w')
    elif save_state and isinstance(state, dolfin.cpp.io.HDF5File):
      self.state = state

  def generate_constants(self):
    """
    Initializes important constants.
    """
    s = "::: generating constants :::"
    print_text(s, self.model_color)

    spy = 31556926.0
    ghf = 0.042 * spy  # W/m^2 = J/(s*m^2) = spy * J/(s*m^2)
    
    # Constants :
    self.kcHh    = Constant(3.7e-9)
    self.kcHh.rename('kcHh', 'creep coefficient high')

    self.kcLw    = Constant(9.2e-9)
    self.kcLw.rename('kcLw', 'creep coefficient low ')

    self.kg      = Constant(1.3e-7)
    self.kg.rename('kg', 'grain growth coefficient')

    self.Ec      = Constant(60e3)
    self.Ec.rename('Ec', 'act. energy for water in ice')

    self.Eg      = Constant(42.4e3)
    self.Eg.rename('Eg', 'act. energy for grain growth')

    self.thetasp = Constant(2009.0 * 273.15)
    self.thetasp.rename('thetasp', 'Internal energy of ice at Tw')
    
    self.etaw    = Constant(1.787e-3)
    self.etaw.rename('etaw', 'Dynamic viscosity of water at Tw')

    self.r       = Constant(0.0)
    self.r.rename('r', 'thickness exponent in sliding law')

    self.eps_reg = Constant(1e-15)
    self.eps_reg.rename('eps_reg', 'strain rate regularization parameter')

    self.n       = Constant(3.0)
    self.n.rename('n', 'viscosity nonlinearity parameter')

    self.spy     = Constant(spy)
    self.spy.rename('spy', 'seconds per year')

    self.A0      = Constant(1e-16)
    self.A0.rename('A0', 'flow rate factor')

    self.rhoi    = Constant(917.0)
    self.rhoi.rename('rhoi', 'ice density')

    self.rhow    = Constant(1000.0)
    self.rhow.rename('rhow', 'water density')
    
    self.rhom    = Constant(550.0)
    self.rhom.rename('rhom', 'firn pore close-off density')

    self.rhoc    = Constant(815.0)
    self.rhoc.rename('rhoc', 'firn density critical value')

    self.g       = Constant(9.80665)
    self.g.rename('g', 'gravitational acceleration')

    self.a0      = Constant(5.45e10)
    self.a0.rename('a0', 'ice hardness limit')

    self.Q0      = Constant(13.9e4)
    self.Q0.rename('Q0', 'ice activation energy')

    self.R       = Constant(8.3144621)
    self.R.rename('R', 'universal gas constant')

    self.ki      = Constant(2.1)
    self.ki.rename('ki', 'thermal conductivity of ice')

    self.kw      = Constant(0.561)
    self.kw.rename('kw', 'thermal conductivity of water')

    self.ci      = Constant(2009.0)
    self.ci.rename('ci', 'heat capacity of ice')
    
    self.cw      = Constant(4217.6)
    self.cw.rename('cw', 'Heat capacity of water at Tw')

    self.L       = Constant(3.3355e5)
    self.L.rename('L', 'latent heat of ice')

    self.ghf     = Constant(ghf)
    self.ghf.rename('ghf', 'geothermal heat flux')

    self.gamma   = Constant(9.8e-8)
    self.gamma.rename('gamma', 'pressure melting point depth dependence')

    self.nu      = Constant(3.5e3)
    self.nu.rename('nu', 'moisture diffusivity')

    self.T_w     = Constant(273.15)
    self.T_w.rename('T_w', 'Triple point of water')

  def init_state(self, fn):
    """
    set the self.state .h5 file for saving variables to <f>.h5 in self.out_dir.
    """
    self.state = HDF5File(mpi_comm_world(), self.out_dir + fn + '.h5', 'w')

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    raiseNotDefined()
    
  def set_mesh(self, f):
    """
    Sets the mesh to <f>, either a dolfin.Mesh or .h5 with a mesh file 
    saved with name 'mesh'.
    """
    s = "::: setting mesh :::"
    print_text(s, self.model_color)

    if isinstance(f, dolfin.cpp.io.HDF5File):
      self.mesh = Mesh()
      f.read(self.mesh, 'mesh', False)

    elif isinstance(f, dolfin.cpp.mesh.Mesh):
      self.mesh = f

    self.dim   = self.mesh.ufl_cell().topological_dimension()

  def calculate_boundaries(self, mask=None, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    raiseNotDefined()
  
  def set_out_dir(self, out_dir):
    """
    Set the output directory to something new.
    """
    self.out_dir = out_dir

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the finite-element function spaces used by all children of model.
    """
    s = "::: generating fundamental function spaces :::"
    print_text(s, self.model_color)

    if use_periodic:
      self.generate_pbc()
    else:
      self.pBC = None
    self.Q      = FunctionSpace(self.mesh,      "CG", 1, 
                                constrained_domain=self.pBC)
    self.Q2     = MixedFunctionSpace([self.Q]*2)
    self.Q3     = MixedFunctionSpace([self.Q]*3)
    self.Q4     = MixedFunctionSpace([self.Q]*4)
    self.Q_non_periodic = FunctionSpace(self.mesh, "CG", 1)
    self.V      = VectorFunctionSpace(self.mesh, "CG", 1)

    s = "    - fundamental function spaces created - "
    print_text(s, self.model_color)
  
  def init_S(self, S):
    """
    Set the Function for the surface <S>. 
    """
    s = "::: initializng surface topography :::"
    print_text(s, self.model_color)
    self.assign_variable(self.S, S)
    print_min_max(self.S, 'S')

  def init_B(self, B):
    """
    Set the Function for the bed <B>.
    """
    s = "::: initializng bed topography :::"
    print_text(s, self.model_color)
    self.assign_variable(self.B, B)
    print_min_max(self.B, 'B')
  
  def init_p(self, p):
    """
    """
    s = "::: initializing pressure :::"
    print_text(s, self.model_color)
    self.assign_variable(self.p, p)
    print_min_max(self.p, 'p')
  
  def init_theta(self, theta):
    """
    """
    s = "::: initializing internal energy :::"
    print_text(s, self.model_color)
    self.assign_variable(self.theta, theta)
    print_min_max(self.theta, 'theta')
  
  def init_T(self, T):
    """
    """
    s = "::: initializing temperature :::"
    print_text(s, self.model_color)
    self.assign_variable(self.T, T)
    print_min_max(self.T, 'T')
  
  def init_W(self, W):
    """
    """
    s = "::: initializing water content :::"
    print_text(s, self.model_color)
    self.assign_variable(self.W, W)
    print_min_max(self.W, 'W')
  
  def init_Mb(self, Mb):
    """
    """
    s = "::: initializing basal melt rate :::"
    print_text(s, self.model_color)
    self.assign_variable(self.Mb, Mb)
    print_min_max(self.Mb, 'Mb')
  
  def init_adot(self, adot):
    """
    """
    s = "::: initializing accumulation :::"
    print_text(s, self.model_color)
    self.assign_variable(self.adot, adot)
    print_min_max(self.adot, 'adot')
  
  def init_beta(self, beta):
    """
    """
    s = "::: initializing basal traction coefficient :::"
    print_text(s, self.model_color)
    self.assign_variable(self.beta, beta)
    print_min_max(self.beta, 'beta')
  
  def init_b(self, b):
    """
    """
    s = "::: initializing rate factor over grounded and shelves :::"
    print_text(s, self.model_color)
    self.assign_variable(self.b, b)
    print_min_max(self.b, 'b')
    self.init_b_shf(b)
    self.init_b_gnd(b)
  
  def init_b_shf(self, b_shf):
    """
    """
    s = "::: initializing rate factor over shelves :::"
    print_text(s, self.model_color)
    self.assign_variable(self.b_shf, b_shf)
    print_min_max(self.b_shf, 'b_shf')
  
  def init_b_gnd(self, b_gnd):
    """
    """
    s = "::: initializing rate factor over grounded ice :::"
    print_text(s, self.model_color)
    self.assign_variable(self.b_gnd, b_gnd)
    print_min_max(self.b_gnd, 'b_gnd')
    
  def init_E(self, E):
    """
    """
    s = "::: initializing enhancement factor over grounded and shelves :::"
    print_text(s, self.model_color)
    self.assign_variable(self.E, E)
    print_min_max(self.E, 'E')
    self.init_E_shf(E)
    self.init_E_gnd(E)
  
  def init_E_shf(self, E_shf):
    """
    """
    s = "::: initializing enhancement factor over shelves :::"
    print_text(s, self.model_color)
    self.assign_variable(self.E_shf, E_shf)
    print_min_max(self.E_shf, 'E_shf')
  
  def init_E_gnd(self, E_gnd):
    """
    """
    s = "::: initializing enhancement factor over grounded ice :::"
    print_text(s, self.model_color)
    self.assign_variable(self.E_gnd, E_gnd)
    print_min_max(self.E_gnd, 'E_gnd')
  
  def init_eta(self, eta):
    """
    """
    s = "::: initializing viscosity :::"
    print_text(s, self.model_color)
    self.assign_variable(self.eta, eta)
    print_min_max(self.eta, 'eta')
  
  def init_etabar(self, etabar):
    """
    """
    s = "::: initializing vertically averaged viscosity :::"
    print_text(s, self.model_color)
    self.assign_variable(self.etabar, etabar)
    print_min_max(self.etabar, 'etabar')
  
  def init_component_Ubar(self, ubar, vbar, wbar):
    """
    """
    s = "::: initializing vertically averaged velocity :::"
    print_text(s, self.model_color)
    self.assign_variable(self.ubar, ubar)
    self.assign_variable(self.vbar, vbar)
    self.assign_variable(self.wbar, wbar)
    print_min_max(self.ubar, 'ubar')
    print_min_max(self.vbar, 'vbar')
    print_min_max(self.wbar, 'wbar')
  
  def init_T_surface(self, T_s):
    """
    """
    s = "::: initializing surface temperature :::"
    print_text(s, self.model_color)
    self.assign_variable(self.T_surface, T_s)
    print_min_max(self.T_surface, 'T_surface')
  
  def init_q_geo(self, q_geo):
    """
    """
    s = "::: initializing geothermal heat flux :::"
    print_text(s, self.model_color)
    self.assign_variable(self.q_geo, q_geo)
    print_min_max(self.q_geo, 'q_geo')

  def init_U(self, u, v, w):
    """
    """
    s = "::: initializing velocity :::"
    print_text(s, self.model_color)
    u_t = Function(self.Q, name='u_t')
    v_t = Function(self.Q, name='v_t')
    w_t = Function(self.Q, name='w_t')
    self.assign_variable(u_t, u)
    self.assign_variable(v_t, v)
    self.assign_variable(w_t, w)
    self.assx.assign(self.u, u_t, annotate=False)
    self.assy.assign(self.v, v_t, annotate=False)
    self.assz.assign(self.w, w_t, annotate=False)
    u_v      = u_t.vector().array()
    v_v      = v_t.vector().array()
    w_v      = w_t.vector().array()
    U_mag_v  = np.sqrt(u_v**2 + v_v**2 + w_v**2 + 1e-16)
    self.assign_variable(self.U_mag, U_mag_v)
    print_min_max(u_t, 'u')
    print_min_max(v_t, 'v')
    print_min_max(w_t, 'w')
    print_min_max(self.U_mag, 'U_mag')
  
  def init_U_ob(self, u_ob, v_ob):
    """
    """
    s = "::: initializing surface velocity :::"
    print_text(s, self.model_color)
    self.assign_variable(self.u_ob, u_ob)
    self.assign_variable(self.v_ob, v_ob)
    u_v      = self.u_ob.vector().array()
    v_v      = self.v_ob.vector().array()
    U_mag_v  = np.sqrt(u_v**2 + v_v**2 + 1e-16)
    self.assign_variable(self.U_ob, U_mag_v)
    print_min_max(self.u_ob, 'u_ob')
    print_min_max(self.v_ob, 'v_ob')
    print_min_max(self.U_ob, 'U_ob')
  
  def init_Ubar(self, Ubar):
    """
    """
    s = "::: initializing balance velocity :::"
    print_text(s, self.model_color)
    self.assign_variable(self.Ubar, Ubar)
    print_min_max(self.Ubar, 'Ubar')
  
  def init_u_lat(self, u_lat):
    """
    """
    s = "::: initializing u lateral boundary condition :::"
    print_text(s, self.model_color)
    self.assign_variable(self.u_lat, u_lat)
    print_min_max(self.u_lat, 'u_lat')
  
  def init_v_lat(self, v_lat):
    """
    """
    s = "::: initializing v lateral boundary condition :::"
    print_text(s, self.model_color)
    self.assign_variable(self.v_lat, v_lat)
    print_min_max(self.v_lat, 'v_lat')
  
  def init_w_lat(self, w_lat):
    """
    """
    s = "::: initializing w lateral boundary condition :::"
    print_text(s, self.model_color)
    self.assign_variable(self.w_lat, w_lat)
    print_min_max(self.w_lat, 'w_lat')
  
  def init_mask(self, mask):
    """
    """
    s = "::: initializing shelf mask :::"
    print_text(s, self.model_color)
    self.assign_variable(self.mask, mask)
    print_min_max(self.mask, 'mask')
    self.shf_dofs = np.where(self.mask.vector().array() == 2.0)[0]
    self.gnd_dofs = np.where(self.mask.vector().array() == 1.0)[0]
  
  def init_U_mask(self, U_mask):
    """
    """
    s = "::: initializing velocity mask :::"
    print_text(s, self.model_color)
    self.assign_variable(self.U_mask, U_mask)
    print_min_max(self.U_mask, 'U_mask')
    self.Uob_dofs         = np.where(self.U_mask.vector().array() == 1.0)[0]
    self.Uob_missing_dofs = np.where(self.U_mask.vector().array() == 0.0)[0]
  
  def init_lat_mask(self, lat_mask):
    """
    """
    s = "::: initializing lateral boundary mask :::"
    print_text(s, self.model_color)
    self.assign_variable(self.lat_mask, lat_mask)
    print_min_max(self.lat_mask, 'lat_mask')
  
  def init_d_x(self, d_x):
    """
    """
    s = "::: initializing x-component-normalized-driving-stress direction :::"
    print_text(s, self.model_color)
    self.assign_variable(self.d_x, d_x)
    print_min_max(self.d_x, 'd_x')
  
  def init_d_y(self, d_y):
    """
    """
    s = "::: initializing y-component-normalized-driving-stress direction :::"
    print_text(s, self.model_color)
    self.assign_variable(self.d_y, d_y)
    print_min_max(self.d_y, 'd_y')
  
  def init_time_step(self, dt):
    """
    """
    s = "::: initializing time step :::"
    print_text(s, self.model_color)
    self.assign_variable(self.time_step, dt)
    print_min_max(self.time_step, 'time_step')

  def init_beta_SIA(self, U_mag=None, eps=0.5):
    r"""
    Init beta  :`\tau_b = \tau_d`, the shallow ice approximation, 
    using the observed surface velocity <U_mag> as approximate basal 
    velocity and <gradS> the projected surface gradient. i.e.,

    .. math::
    \beta \Vert U_b \Vert H^r = \rho g H \Vert \nabla S \Vert
    
    """
    s = "::: initializing beta from SIA :::"
    print_text(s, self.model_color)
    r        = self.r
    Q        = self.Q
    rhoi     = self.rhoi
    g        = self.g
    gradS    = grad(self.S)
    H        = self.S - self.B
    U_s      = Function(Q)
    if U_mag == None:
      U_v                        = self.U_ob.vector().array()
      Ubar_v                     = self.Ubar.vector().array()
      U_v[self.Uob_missing_dofs] = Ubar_v[self.Uob_missing_dofs]
    else:
      U_v = U_mag.vector().array()
    U_v[U_v < eps] = eps
    self.assign_variable(U_s, U_v, save=False)
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta_0   = project((rhoi*g*H*S_mag) / (H**r * U_s), Q, annotate=False)
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < 1e-2] = 1e-2
    self.betaSIA = Function(Q, name='betaSIA')
    self.assign_variable(self.betaSIA, beta_0_v)
    print_min_max(self.betaSIA, 'betaSIA')
    
    if self.dim == 3:
      #self.assign_variable(self.beta, DOLFIN_EPS, save=False)
      #bc_beta = DirichletBC(self.Q, self.betaSIA, self.ff, self.GAMMA_B_GND)
      #bc_beta.apply(self.beta.vector())
      self.assign_variable(self.beta, self.betaSIA)
    elif self.dim == 2:
      self.assign_variable(self.beta, self.betaSIA)
    print_min_max(self.beta, 'beta')
      
  def init_beta_SIA_new_slide(self, U_mag=None, eps=0.5):
    r"""
    Init beta  :`\tau_b = \tau_d`, the shallow ice approximation, 
    using the observed surface velocity <U_mag> as approximate basal 
    velocity and <gradS> the projected surface gradient. i.e.,

    .. math::
    \beta \Vert U_b \Vert H^r = \rho g H \Vert \nabla S \Vert
    
    """
    s = "::: initializing new sliding beta from SIA :::"
    print_text(s, self.model_color)
    r        = 0.0
    Q        = self.Q
    rhoi     = self.rhoi
    rhow     = self.rhow
    g        = self.g
    gradS    = self.gradS
    H        = self.S - self.B
    D        = self.D
    p        = -0.383
    q        = -0.349
    
    U_s      = Function(Q)
    if U_mag == None:
      U_v = self.U_ob.vector().array()
    else:
      U_v = U_mag.vector().array()
    U_v[U_v < eps] = eps
    self.assign_variable(U_s, U_v)
    
    Ne       = H + rhow/rhoi * D
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta     = U_s**(1/p) / ( rhoi * g * H * S_mag * Ne**(q/p) )
    beta_0   = project(beta, Q, annotate=False)
    
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    #self.assign_variable(beta_0, beta_0_v)
    print_min_max(beta_0, 'beta_0')

    #self.assign_variable(self.beta, beta_0)
    
    self.assign_variable(self.beta, DOLFIN_EPS)
    bc_beta = DirichletBC(self.Q, beta_0, self.ff, GAMMA_B_GND)
    bc_beta.apply(self.beta.vector())
    
    #self.betaSIA = Function(Q)
    #self.assign_variable(self.betaSIA, beta_0_v)
    
  def init_beta_stats(self, mdl='Ubar', use_temp=False, mode='steady'):
    """
    """
    s    = "::: initializing beta from stats :::"
    print_text(s, self.model_color)
    
    q_geo  = self.q_geo
    T_s    = self.T_surface
    adot   = self.adot
    Mb     = self.Mb
    Ubar   = self.Ubar
    Q      = self.Q
    B      = self.B
    S      = self.S
    T      = self.T
    T_s    = self.T_surface
    rho    = self.rhoi
    g      = self.g
    H      = S - B

    Ubar_v = Ubar.vector().array()
    Ubar_v[Ubar_v < 1e-10] = 1e-10
    self.assign_variable(Ubar, Ubar_v)
           
    D      = Function(Q)
    B_v    = B.vector().array()
    D_v    = D.vector().array()
    D_v[B_v < 0] = B_v[B_v < 0]
    self.assign_variable(D, D_v)

    gradS = as_vector([S.dx(0), S.dx(1), 0.0])
    gradB = as_vector([B.dx(0), B.dx(1), 0.0])
    gradH = as_vector([H.dx(0), H.dx(1), 0.0])

    nS   = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    nB   = sqrt(inner(gradB, gradB) + DOLFIN_EPS)
    nH   = sqrt(inner(gradH, gradH) + DOLFIN_EPS)
    
    #if mdl == 'Ubar':
    #  u_x    = -rho * g * H * S.dx(0)
    #  v_x    = -rho * g * H * S.dx(1)
    #  U_i    = as_vector([u_x,  v_x, 0.0])
    #  U_j    = as_vector([v_x, -u_x, 0.0])
    #elif mdl == 'U' or mdl == 'U_Ubar':
    #  U_i    = as_vector([self.u,  self.v, 0.0])
    #  U_j    = as_vector([self.v, -self.u, 0.0])
    U_i    = as_vector([self.u,  self.v, 0.0])
    U_j    = as_vector([self.v, -self.u, 0.0])
    Umag   = sqrt(inner(U_i,U_i) + DOLFIN_EPS)
    Uhat_i = U_i / Umag
    Uhat_j = U_j / Umag

    dBdi = dot(gradB, Uhat_i)
    dBdj = dot(gradB, Uhat_j)
    dSdi = dot(gradS, Uhat_i)
    dSdj = dot(gradS, Uhat_j)
    dHdi = dot(gradH, Uhat_i)
    dHdj = dot(gradH, Uhat_j)

    ini  = sqrt(rho * g * H * nS / (Umag + 0.1))

    x0   = S
    x1   = T_s
    x2   = nS
    x3   = D
    x4   = nB
    x5   = H
    x6   = q_geo
    x7   = adot
    x8   = T
    x9   = Mb
    x10  = self.u
    x11  = self.v
    x12  = self.w
    x13  = ln(Ubar + DOLFIN_EPS)
    x14  = ln(Umag + DOLFIN_EPS)
    x15  = ini
    x16  = dBdi
    x17  = dBdj
    x18  = dSdi
    x19  = dSdj
    x20  = nH
    x21  = self.tau_id
    x22  = self.tau_jd
    x23  = self.tau_ii
    x24  = self.tau_ij
    x25  = self.tau_ji
    x26  = self.tau_jj

    names = ['S', 'T_s', 'gradS', 'D', 'gradB', 'H', 'q_geo', 'adot', 'T',
             'Mb', 'u', 'v', 'w', 'ln(Ubar)', 'ln(Umag)', 'ini',
             'dBdi', 'dBdj', 'dSdi', 'dSdj', 'nablaH', 'tau_id', 'tau_jd',
             'tau_ii', 'tau_ij', 'tau_ji', 'tau_jj']
    names = np.array(names)

    if mdl == 'Ubar':
      if not use_temp:
        X    = [x0,x1,x5,x7,x13,x16,x18]
        idx  = [ 0, 1, 5, 7, 13, 16, 18]
        bhat = [ -1.01661102e+02,   6.59472291e-03,   8.34479667e-01,
                 -3.20751595e-04,  -1.86910058e+00,  -1.50122785e-01,
                 -1.61283407e+01,   3.42099244e+01,  -1.38190017e-07,
                 -2.42124307e-05,   5.28420031e-08,  -5.71485389e-05,
                 -3.75168897e-06,   6.62615357e-04,  -2.09616017e-03,
                 -1.63919106e-03,  -4.67468432e-07,   7.70150910e-03,
                 -1.06827565e-05,   5.82852747e-02,  -1.59176855e-01,
                  2.60703978e-08,   1.12176250e-04,  -9.96266233e-07,
                  1.54898171e-04,  -7.75201260e-03,  -3.97881378e-02,
                 -9.66212690e-04,  -6.88656946e-01,   2.86508703e+00,
                 -4.77406074e-03,   4.46234782e-03,  -9.93937326e-02,
                 -1.11058398e+01,   1.19703551e+01,  -3.46378138e+01]
        #bhat = [ -1.06707322e+02,   6.93681939e-03,   8.72090381e-01,
        #         -2.05377136e-04,  -1.68695225e+00,  -1.54427603e-01,
        #         -1.48494954e+01,   3.13320531e+01,  -1.46372911e-07,
        #         -2.54809386e-05,   5.58213888e-08,  -5.05686875e-05,
        #         -3.57485925e-06,   6.74423417e-04,  -1.90332998e-03,
        #         -1.70912922e-03,  -9.14015814e-07,   6.90894685e-03,
        #          5.38728829e-06,   5.52828014e-02,  -1.49677701e-01,
        #          2.10321794e-08,   1.26574205e-04,  -1.58804814e-06,
        #         -1.07066137e-04,  -6.59781673e-03,  -4.21221477e-02,
        #         -9.11842753e-04,  -5.91089434e-01,   2.37465616e+00,
        #         -4.79794725e-03,  -1.20787950e-03,  -8.37001425e-02,
        #         -1.35364012e+01,   2.01047113e+01,  -3.48057200e+01]
     
      else: 
        X    = [x0,x1,x5,x7,x8,x9,x13,x16,x18]
        idx  = [ 0, 1, 5, 7, 8, 9, 13, 16, 18]
        bhat = [  1.99093750e+01,  -9.37152784e-04,  -1.53849816e-03,
                 -2.72682710e-03,   3.11376629e+00,  -6.22550705e-02,
                 -4.78841821e+02,   1.18870083e-01,   1.46462501e+01,
                  4.73228083e+00,  -1.23039512e-05,   4.80948459e-08,
                 -1.75152253e-04,   1.57869882e-05,  -1.85979092e-03,
                 -5.31979350e-06,  -2.94994855e-04,  -2.88696470e-03,
                  9.87920894e-06,  -1.67014309e-02,   1.38310308e-05,
                  1.29911016e+00,   8.79462642e-06,   2.58486129e-02,
                  4.59079956e-01,  -1.62460133e-04,   8.39672735e-07,
                 -1.44977594e-02,   5.58957555e-07,   7.38625502e-04,
                 -9.92789432e-03,   6.02766800e-03,   2.74638935e-01,
                 -7.24036641e-05,  -4.63126335e-01,   2.92369712e+00,
                  5.07887934e-01,  -4.57929508e-04,  -8.33728342e-02,
                 -4.71625234e-01,  -5.85160316e-02,  -1.74723504e+01,
                 -1.83509536e+01,   5.35514345e-04,  -8.46507380e-02,
                 -1.60127263e+01]
    
    elif mdl == 'U':
      if not use_temp:
        X    = [x0,x1,x5,x7,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 14, 16, 18]
        bhat = [ -9.28289389e+01,   5.73687339e-03,   7.33526290e-01,
                  2.76998568e-03,  -1.08656857e-01,  -1.08545047e+00,
                 -1.50267782e+01,  -7.04864127e+01,  -7.76085391e-08,
                 -2.17802438e-05,  -4.99587467e-08,   5.87139196e-05,
                  1.64670170e-05,   1.06212966e-04,   7.11755177e-05,
                 -1.37677776e-03,  -9.08932836e-06,   3.60621065e-04,
                  2.97118032e-03,   5.50814766e-02,   2.21044611e-01,
                 -1.15497725e-07,   8.63993130e-05,  -2.12395318e-06,
                  7.21699958e-04,  -1.09346933e-02,  -3.12224072e-02,
                 -2.39690796e-02,  -2.95080157e-01,  -3.40502802e-01,
                 -2.62000881e-02,  -1.78157283e-02,   7.19763432e-02,
                 -1.94919730e+00,  -9.82413027e+00,  -7.61245200e+01]
      else:
        X    = [x0,x1,x5,x7,x8,x9,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 8, 9, 14, 16, 18]
        bhat = [  2.09623581e+01,   6.66919839e-04,  -7.02196170e-02,
                 -1.15080308e-03,   5.34783070e+00,  -7.11388758e-02,
                 -4.07361631e+01,   1.02018632e+00,  -1.86900651e+01,
                 -4.20181324e+01,  -9.26143019e-06,  -7.72058925e-08,
                 -4.15062408e-05,   7.02170069e-06,   2.70372865e-03,
                 -1.37333418e-05,   8.87920333e-05,   1.42938174e-03,
                  7.77557165e-06,  -2.35402146e-02,   3.04680358e-04,
                 -1.71597355e-01,   1.40252311e-04,   4.10097716e-02,
                  2.55567246e-01,  -1.33628767e-07,  -2.15459028e-06,
                  6.29599393e-05,  -4.11071912e-05,   1.28619782e-03,
                 -1.46657539e-02,   3.09279801e-03,  -2.27450062e-01,
                 -7.40025166e-03,  -5.06709113e-01,  -6.76120111e-01,
                  3.10802402e-01,  -5.34552872e-03,   2.19914707e-02,
                 -1.40943367e-01,   3.07890125e-01,  -9.03508676e+00,
                  8.27529346e+01,   6.60448755e-03,   2.42989633e+00,
                 -4.31461210e+01]
    
    elif mdl == 'U_Ubar':
      if not use_temp:
        X    = [x0,x1,x5,x7,x13,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 13, 14, 16, 18]
        bhat = [ -9.25221622e+01,   5.70295987e-03,   7.30768422e-01,
                  2.75877006e-03,   7.37861453e-02,  -2.93985236e-03,
                 -1.07390793e+00,  -1.45320123e+01,  -7.18521246e+01,
                 -7.86411913e-08,  -2.15769127e-05,  -4.80926515e-08,
                  5.56842889e-05,   1.28402687e-06,   1.12826733e-05,
                  9.07581727e-05,  -7.62357377e-05,  -1.37165484e-03,
                 -8.99331396e-06,  -3.36292037e-04,   4.24771193e-05,
                  2.97610385e-03,   5.34869351e-02,   2.28993842e-01,
                 -1.17987943e-07,   8.26468590e-05,   2.32815553e-06,
                 -6.66323072e-06,   6.73934903e-04,  -1.12192482e-02,
                 -3.22339742e-02,  -3.78492901e-04,  -2.38023512e-02,
                 -2.88687981e-01,  -4.11715791e-01,   3.06665249e-04,
                  3.29695662e-04,   4.96515338e-03,   1.28914720e-02,
                 -2.83133687e-02,  -3.08127082e-02,  -3.19074160e-02,
                 -1.60977763e+00,  -1.10451113e+01,  -7.66011531e+01]
      else:
        X    = [x0,x1,x5,x7,x8,x9,x13,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 8, 9, 13, 14, 16, 18]
        bhat = [  1.95228446e+01,   6.59477606e-04,  -6.45139002e-02,
                 -1.10071394e-03,   5.13699019e+00,  -6.45652015e-02,
                 -5.14739582e+01,  -3.68769001e-03,   9.57519905e-01,
                 -1.77507405e+01,  -4.37983921e+01,  -9.02491948e-06,
                 -7.61384926e-08,  -3.73066416e-05,   6.79516468e-06,
                  2.83564402e-03,  -4.68103812e-07,  -1.20747491e-05,
                  4.00845895e-05,   1.67755582e-03,   7.73371401e-06,
                 -2.23470170e-02,   2.78775317e-04,  -1.61211932e-01,
                  4.64633086e-05,   4.37335336e-04,   4.27466758e-02,
                  2.50573113e-01,  -4.81341231e-06,  -2.31708961e-06,
                 -1.68503900e-04,   3.54318161e-06,  -4.20165147e-05,
                  1.26878513e-03,  -1.54490818e-02,   2.66749014e-03,
                 -2.98194766e-01,  -2.92113296e-04,  -4.31378498e-03,
                 -4.83721711e-01,  -7.30055588e-01,   3.42250813e-01,
                 -3.22616161e-05,  -5.40195432e-03,   1.73408633e-02,
                 -1.31066469e-01,   9.73640123e-03,   2.61368301e-01,
                 -9.93273895e+00,   8.31773699e+01,  -5.74031885e-04,
                  9.54289863e-03,  -3.57353698e-02,   3.62295735e-03,
                  2.54399352e+00,  -4.21129483e+01]
    
    elif mdl == 'stress':
      X    = [x0,x1,x5,x7,x14,x16,x18,x21,x22,x24,x25,x26]
      idx  = [ 0, 1, 5, 7, 14, 16, 18, 21, 23, 24, 25, 26]
      bhat = [  5.47574225e+00,   9.14001489e-04,  -1.03229081e-03,
               -7.04987042e-04,   2.15686223e+00,  -1.52869679e+00,
               -1.74593819e+01,  -2.05459701e+01,  -1.23768850e-05,
                2.01460255e-05,   1.97622781e-05,   3.68067438e-05,
                6.63468606e-06,  -3.69046174e-06,  -4.47828887e-08,
               -3.67070759e-05,   2.53827543e-05,  -1.88069561e-05,
                2.05942231e-03,  -5.95566325e-10,   1.00881255e-09,
                6.11553989e-10,  -4.11737126e-10,   6.27370976e-10,
                3.42275389e-06,  -8.17017771e-03,   4.01803819e-03,
                6.78767571e-02,   4.29444354e-02,   4.45551518e-08,
               -8.23509210e-08,  -7.90182526e-08,  -1.48650850e-07,
               -2.36138203e-08,  -4.75130905e-05,  -1.81655894e-05,
                9.79852186e-04,  -1.49411705e-02,  -2.35701903e-10,
                2.32406866e-09,   1.48224703e-09,  -1.09016625e-09,
               -1.31162142e-09,   1.47593911e-02,  -1.84965301e-01,
               -1.62413731e-01,   2.38867744e-07,   2.09579112e-07,
                6.11572155e-07,   1.44891826e-06,  -4.94537953e-07,
               -3.30400642e-01,   7.93664407e-01,   7.76571489e-08,
               -1.64476914e-07,  -2.13414311e-07,   4.75810302e-07,
                2.55787543e-07,  -6.37972323e+00,  -3.77364196e-06,
                8.65062737e-08,   6.13207853e-06,   8.39233482e-07,
               -3.76402983e-06,  -2.02633500e-05,  -7.28788200e-06,
               -2.72030382e-05,  -1.33298507e-05,   1.11838930e-05,
                9.74762098e-14,  -2.37844072e-14,  -1.11310490e-13,
                8.91237008e-14,   1.16770903e-13,   5.77230478e-15,
               -4.87322338e-14,   9.62949381e-14,  -2.12122129e-13,
                1.55871983e-13]
   
    for xx,nam in zip(X, names[idx]):
      print_min_max(xx, nam)

    X_i  = []
    X_i.extend(X)
     
    for i,xx in enumerate(X):
      if mdl == 'Ubar' or mdl == 'U' and not use_temp:
        k = i
      else:
        k = i+1
      for yy in X[k:]:
        X_i.append(xx*yy)
    
    #self.beta_f = exp(Constant(bhat[0]))
    self.beta_f = Constant(bhat[0])
    
    for xx,bb in zip(X_i, bhat[1:]):
      self.beta_f += Constant(bb)*xx
      #self.beta_f *= exp(Constant(bb)*xx)
    self.beta_f = exp(self.beta_f)**2
    
    if mode == 'steady':
      beta0                   = project(self.beta_f, Q, annotate=False)
      beta0_v                 = beta0.vector().array()
      beta0_v[beta0_v < 1e-2] = 1e-2
      self.assign_variable(beta0, beta0_v)
    
      self.assign_variable(self.beta, 1e-2)
      bc_beta = DirichletBC(self.Q, beta0, self.ff, self.GAMMA_B_GND)
      bc_beta.apply(self.beta.vector())
    elif mode == 'transient':
      self.assign_variable(self.beta, 200.0)
    
    print_min_max(self.beta, 'beta0')
 
  def update_stats_beta(self):
    """
    Re-compute the statistical friction field and save into model.beta.
    """
    s    = "::: updating statistical beta :::"
    print_text(s, self.D3Model_color)
    beta   = project(self.beta_f, self.Q, annotate=False)
    beta_v = beta.vector().array()
    ##betaSIA_v = self.betaSIA.vector().array()
    ##beta_v[beta_v < 10.0]   = betaSIA_v[beta_v < 10.0]
    beta_v[beta_v < 0.0]    = 0.0
    #beta_v[beta_v > 2500.0] = 2500.0
    self.assign_variable(self.beta, beta_v)
    print_min_max(self.beta, 'beta')
     
  def init_b_SIA(self, b, U_ob, gradS):
    r"""
    Init rate-factor b from U_ob. 
    """
    s = "::: initializing b from U_ob :::"
    print_text(s, self.model_color)
   
    x      = self.x
    S      = self.S
    Q      = self.Q
    rhoi   = self.rhoi
    rhow   = self.rhow
    g      = self.g
    u      = U_ob[0]
    v      = U_ob[1]
    n      = 3.0
    
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = min(0, x[2])
    D = Depth(element = Q.ufl_element())
    
    N      = FacetNormal(self.mesh)
    dSde   = ds(4)
           
    b_f    = TrialFunction(Q)
    phi    = TestFunction(Q)

    epi    = self.strain_rate(U_ob)
    ep_xx  = epi[0,0]
    ep_yy  = epi[1,1]
    ep_xy  = epi[0,1]
    ep_xz  = epi[0,2]
    ep_yz  = epi[1,2]   
 
    epsdot = ep_xx**2 + ep_yy**2 + ep_xx*ep_yy + ep_xy**2 + ep_xz**2 + ep_yz**2
    eta    = 0.5 * b_f * (epsdot + 1e-10)**((1-n)/(2*n))

    f_w    = rhoi*g*(S - x[2]) + rhow*g*D

    epi_1  = as_vector([   2*u.dx(0) + v.dx(1), 
                        0.5*(u.dx(1) + v.dx(0)),
                        0.5* u.dx(2)            ])
    epi_2  = as_vector([0.5*(u.dx(1) + v.dx(0)),
                             u.dx(0) + 2*v.dx(1),
                        0.5* v.dx(2)            ])

    R  = - 2 * eta * dot(epi_1, grad(phi)) * dx \
         + rhoi * g * gradS[0] * phi * dx \
         #+ 2 * eta * dot(epi_2, grad(phi)) * dx \
         #+ rhoi * g * gradS[1] * phi * dx \
   
    b_f = Function(Q)
    solve(lhs(R) == rhs(R), b_f, annotate=False)
    self.assign_variable(b, b_f)
 
  def calc_eta(self):
    """
    Calculates viscosity, set to model.eta.
    """
    s     = "::: calculating viscosity :::"
    print_text(s, self.model_color)
    Q       = self.Q
    R       = self.R
    T       = self.T
    W       = self.W
    n       = self.n
    u       = self.u
    v       = self.v
    w       = self.w
    eps_reg = self.eps_reg
    E_shf   = self.E_shf
    E_gnd   = self.E_gnd
    E       = self.E
    U       = as_vector([u,v,w])
    
    epsdot = self.effective_strain_rate(U)

    # manually calculate a_T and Q_T to avoid oscillations with 'conditional' :
    a_T    = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
    Q_T    = conditional( lt(T, 263.15), 6e4,          13.9e4)
    #a_T     = Function(Q)
    #Q_T     = Function(Q)
    #T_v     = T.vector().array()
    #a_T_v   = a_T.vector().array()
    #Q_T_v   = Q_T.vector().array()
    #a_T_v[T_v  < 263.15] = 1.1384496e-5
    #a_T_v[T_v >= 263.15] = 5.45e10 
    #Q_T_v[T_v  < 263.15] = 6e4
    #Q_T_v[T_v >= 263.15] = 13.9e4 
    #self.assign_variable(a_T, a_T_v)
    #self.assign_variable(Q_T, Q_T_v)
   
    # unify the enhancement factor over shelves and grounded ice : 
    E   = Function(Q)
    E_v = E.vector().array()
    E_gnd_v = E_gnd.vector().array()
    E_shf_v = E_shf.vector().array()
    E_v[self.gnd_dofs] = E_gnd_v[self.gnd_dofs]
    E_v[self.shf_dofs] = E_shf_v[self.shf_dofs]
    self.assign_variable(E, E_v)

    # calculate viscosity :
    b       = ( E*a_T*(1 + 181.25*W)*exp(-Q_T/(R*T)) )**(-1/n)
    eta     = 0.5 * b * (epsdot + eps_reg)**((1-n)/(2*n))
    eta     = project(eta, Q, annotate=False)
    self.assign_variable(self.eta, eta)
    print_min_max(self.eta, 'eta')

  def calc_vert_average(self, u):
    """
    Calculates the vertical average of a given function space and function.  
    
    :param u: Function representing the model's function space
    :rtype:   Dolfin projection and Function of the vertical average
    """
    raiseNotDefined()

  def calc_misfit(self, integral):
    """
    Calculates the misfit of model and observations, 

      D = ||U - U_ob||

    over shelves or grounded depending on the paramter <integral>, then 
    updates model.misfit with D for plotting.
    """
    s   = "::: calculating misfit L-infty norm ||U - U_ob|| over '%s' :::"
    print_text(s % integral, self.model_color)

    U_s    = Function(self.Q2)
    U_ob_s = Function(self.Q2)
    U      = as_vector([self.u,    self.v])
    U_ob   = as_vector([self.u_ob, self.v_ob])

    if integral == 'shelves':
      bc_U    = DirichletBC(self.Q2, U,    self.ff, GAMMA_S_FLT)
      bc_U_ob = DirichletBC(self.Q2, U_ob, self.ff, GAMMA_S_FLT)
    elif integral == 'grounded':
      bc_U    = DirichletBC(self.Q2, U,    self.ff, GAMMA_S_GND)
      bc_U_ob = DirichletBC(self.Q2, U_ob, self.ff, GAMMA_S_GND)
    
    bc_U.apply(U_s.vector())
    bc_U_ob.apply(U_ob_s.vector())

    # calculate L_inf vector norm :
    U_s_v    = U_s.vector().array()
    U_ob_s_v = U_ob_s.vector().array()
    D_v      = U_s_v - U_ob_s_v
    D        = MPI.max(mpi_comm_world(), D_v.max())
    
    s    = "||U - U_ob|| : %.3E" % D
    print_text(s, '208', 1)
    self.misfit = D

  def get_theta(self):
    """
    Returns the angle in radians of the horizontal velocity vector from 
    the x-axis.
    """
    u_v     = self.u.vector().array()
    v_v     = self.v.vector().array()
    theta_v = np.arctan2(u_v, v_v)
    theta   = Function(self.Q)
    self.assign_variable(theta, theta_v)
    return theta

  def rotate(self, M, theta):
    """
    rotate the tensor <M> about the z axes by angle <theta>.
    """
    c  = cos(theta)
    s  = sin(theta)
    Rz = as_matrix([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])
    R  = dot(Rz, dot(M, Rz.T))
    return R

  def get_norm(self, U, type='l2'):
    """
    returns the norm of vector <U>.
    """
    # iterate through each component and convert to array :
    U_v = []
    for u in U:
      # convert to array and normailze the components of U :
      u_v = u.vector().array()
      U_v.append(u_v)
    U_v = np.array(U_v)

    # calculate the norm :
    if type == 'l2':
      norm_u = np.sqrt(np.sum(U_v**2,axis=0))
    elif type == 'linf':
      norm_u = np.amax(U_v,axis=0)
    
    return U_v, norm_u

  def normalize_vector(self, U, Q='self'):
    """
    Create a normalized vector of the UFL vector <U>.
    """
    if type(Q) != FunctionSpace:
      Q = self.Q

    U_v, norm_u = self.get_norm(U)

    norm_u[norm_u <= 0.0] = 1e-15
    
    # normalize the vector :
    U_v /= norm_u
    
    # convert back to fenics :
    U_f = []
    for u_v in U_v:
      u_f = Function(Q)
      self.assign_variable(u_f, u_v)
      U_f.append(u_f)

    # return a UFL vector :
    return as_vector(U_f)

  def assign_submesh_variable(self, u_to, u_from):
    """
    """
    lg = LagrangeInterpolator()
    lg.interpolate(u_to, u_from)

  def assign_variable(self, u, var, annotate=False, save=True):
    """
    Manually assign the values from <var> to Function <u>.  <var> may be an
    array, float, Expression, or Function.
    """
    if isinstance(var, float) or isinstance(var, int):
      if    isinstance(u, GenericVector) or isinstance(u, Function) \
         or isinstance(u, dolfin.functions.function.Function):
        u.vector()[:] = var
      elif  isinstance(u, Constant):
        u.assign(var, annotate=annotate)
    
    elif isinstance(var, np.ndarray):
      if var.dtype != np.float64:
        var = var.astype(np.float64)
      u.vector().set_local(var)
      u.vector().apply('insert')
    
    elif isinstance(var, Expression) \
      or isinstance(var, Constant)  \
      or isinstance(var, Function) \
      or isinstance(var, dolfin.functions.function.Function):
      u.interpolate(var, annotate=annotate)

    elif isinstance(var, GenericVector):
      self.assign_variable(u, var.array(), annotate=annotate)

    elif isinstance(var, str):
      File(var) >> u

    elif isinstance(var, HDF5File):
      var.read(u, u.name())

    else:
      s =  "*************************************************************\n" + \
           "assign_variable() function requires a Function, array, float,\n" + \
           " int, Vector, Expression, Constant, or string path to .xml,\n"   + \
           "not %s.  Replacing object entirely\n" + \
           "*************************************************************"
      print_text(s % type(var) , 'red', 1)
      u = var

    if self.save_state and save:
      if    isinstance(u, GenericVector) or isinstance(u, Function) \
         or isinstance(u, dolfin.functions.function.Function):
        s = "::: writing '%s' variable to '%sstate.h5' :::"
        print_text(s % (u.name(), self.out_dir), self.model_color)
        self.state.write(u, u.name())
        print_text("    - done -", self.model_color)

  def save_pvd(self, var, name, f_file=None):
    """
    Save a <name>.pvd file of the FEniCS Function <var> to this model's log 
    directory specified by the self.out_dir.
    """
    if f_file != None:
      s       = "::: saving %s.pvd file :::" % name
      print_text(s, self.model_color)
      f_file << var
    else:
      s       = "::: saving %s/pvd/%s.pvd file :::" % (self.out_dir, name)
      print_text(s, self.model_color)
      File(self.out_dir + 'pvd/' + name + '.pvd') << var

  def save_xml(self, var, name):
    """
    Save a <name>.xml file of the FEniCS Function <var> to this model's log 
    directory specified by model.out_dir.
    """
    s       = "::: saving %s/xml/%s.xml file :::" % (self.out_dir, name)
    print_text(s, self.model_color)
    File(self.out_dir + 'xml/' +  name + '.xml') << var
  
  def solve_hydrostatic_pressure(self, annotate=True):
    """
    Solve for the hydrostatic pressure 'p'.
    """
    # solve for vertical velocity :
    s  = "::: solving hydrostatic pressure :::"
    print_text(s, self.model_color)
    rhoi   = self.rhoi
    g      = self.g
    S      = self.S
    z      = self.x[2]
    p      = project(rhoi*g*(S - z), self.Q, annotate=annotate)
    self.assign_variable(self.p, p)
    print_min_max(self.p, 'p')
  
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    s = "::: initializing basic variables :::"
    print_text(s, self.model_color)

    # Coordinates of various types 
    self.x             = SpatialCoordinate(self.mesh)
    self.h             = CellSize(self.mesh)
    self.N             = FacetNormal(self.mesh)

    # time step :
    self.time_step = Constant(100.0)
    self.time_step.rename('time_step', 'time step')

    # shelf mask (2 if shelf) :
    self.mask          = Function(self.Q, name='mask')

    # lateral boundary mask (1 if on lateral boundary) :
    self.lat_mask      = Function(self.Q, name='lat_mask')

    # velocity mask (1 if velocity measurements present) :
    self.U_mask        = Function(self.Q, name='U_mask')

    # topography :
    self.S             = Function(self.Q_non_periodic, name='S')
    self.B             = Function(self.Q_non_periodic, name='B')
    
    # velocity observations :
    self.U_ob          = Function(self.Q, name='U_ob')
    self.u_ob          = Function(self.Q, name='u_ob')
    self.v_ob          = Function(self.Q, name='v_ob')
    
    # unified velocity :
    self.U_mag         = Function(self.Q,  name='U_mag')
    self.U3            = Function(self.Q3, name='U3')
    u,v,w              = self.U3.split()
    u.rename('u', '')
    v.rename('v', '')
    w.rename('w', '')
    self.u             = u
    self.v             = v
    self.w             = w
    
    self.assx          = FunctionAssigner(self.Q3.sub(0), self.Q)
    self.assy          = FunctionAssigner(self.Q3.sub(1), self.Q)
    self.assz          = FunctionAssigner(self.Q3.sub(2), self.Q)

    # momentum model :
    self.eta           = Function(self.Q, name='eta')
    self.p             = Function(self.Q, name='p')
    self.beta          = Function(self.Q, name='beta')
    self.E             = Function(self.Q, name='E')
    self.E_gnd         = Function(self.Q, name='E_gnd')
    self.E_shf         = Function(self.Q, name='E_shf')
    self.b             = Function(self.Q, name='b')
    self.b_gnd         = Function(self.Q, name='b_gnd')
    self.b_shf         = Function(self.Q, name='b_shf')
    self.u_lat         = Function(self.Q, name='u_lat')
    self.v_lat         = Function(self.Q, name='v_lat')
    self.w_lat         = Function(self.Q, name='w_lat')
    
    # energy model :
    self.T             = Function(self.Q, name='T')
    self.q_geo         = Function(self.Q, name='q_geo')
    self.W             = Function(self.Q, name='W')
    self.Mb            = Function(self.Q, name='Mb')
    self.T_melt        = Function(self.Q, name='T_melt')     # pressure-melting
    self.theta_melt    = Function(self.Q, name='theta_melt') # pressure-melting
    self.T_surface     = Function(self.Q, name='T_surface')
    
    # adjoint model :
    self.adj_f         = 0.0              # objective function value at end
    self.misfit        = 0.0              # ||U - U_ob||

    # balance Velocity model :
    self.adot          = Function(self.Q, name='adot')
    self.dSdx          = Function(self.Q, name='dSdx')
    self.dSdy          = Function(self.Q, name='dSdy')
    self.d_x           = Function(self.Q, name='d_x')
    self.d_y           = Function(self.Q, name='d_y')
    self.Ubar          = Function(self.Q, name='Ubar')
    self.Nx            = Function(self.Q, name='Nx')
    self.Ny            = Function(self.Q, name='Ny')
    
    # Stokes-balance model :
    self.u_s           = Function(self.Q, name='u_s')
    self.u_t           = Function(self.Q, name='u_t')
    self.tau_id        = Function(self.Q, name='tau_id')
    self.tau_jd        = Function(self.Q, name='tau_jd')
    self.tau_ib        = Function(self.Q, name='tau_ib')
    self.tau_jb        = Function(self.Q, name='tau_jb')
    self.tau_ip        = Function(self.Q, name='tau_ip')
    self.tau_jp        = Function(self.Q, name='tau_jp')
    self.tau_ii        = Function(self.Q, name='tau_ii')
    self.tau_ij        = Function(self.Q, name='tau_ij')
    self.tau_ji        = Function(self.Q, name='tau_ji')
    self.tau_jj        = Function(self.Q, name='tau_jj')

  def thermo_solve(self, momentum, energy, callback=None, 
                   rtol=1e-6, max_iter=15):
    """ 
    Perform thermo-mechanical coupling between momentum and energy.
    """
    s    = '::: performing thermo-mechanical coupling :::'
    print_text(s, self.model_color)
    
    from varglas.momentum import Momentum
    from varglas.energy   import Energy
    
    if momentum.__class__.__base__ != Momentum:
      s = ">>> thermo_solve REQUIRES A 'Momentum' INSTANCE, NOT %s <<<"
      print_text(s % type(momentum) , 'red', 1)
      sys.exit(1)
    
    if energy.__class__.__base__ != Energy:
      s = ">>> thermo_solve REQUIRES AN 'Energy' INSTANCE, NOT %s <<<"
      print_text(s % type(energy) , 'red', 1)
      sys.exit(1)

    t0   = time()

    # L_\infty norm in velocity between iterations :
    inner_error = np.inf
   
    # number of iterations
    counter     = 0
   
    # previous velocity for norm calculation
    U_prev      = self.U3.copy(True)

    # perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance :
    while inner_error > rtol and counter < max_iter:
     
      # need zero initial guess for Newton solve to converge : 
      self.assign_variable(momentum.get_U(),  DOLFIN_EPS, save=False)
      
      # solve velocity :
      momentum.solve(annotate=False)

      # solve energy (temperature, water content) :
      energy.solve(annotate=False)

      # calculate L_infinity norm, increment counter :
      counter       += 1
      inner_error_n  = norm(project(U_prev - self.U3, annotate=False))
      U_prev         = self.U3.copy(True)
      if self.MPI_rank==0:
        s0    = '>>> '
        s1    = 'Picard iteration %i (max %i) done: ' % (counter, max_iter)
        s2    = 'r0 = %.3e'  % inner_error
        s3    = ', '
        s4    = 'r = %.3e ' % inner_error_n
        s5    = '(tol %.3e)' % rtol
        s6    = ' <<<'
        text0 = get_text(s0, 'red', 1)
        text1 = get_text(s1, self.model_color)
        text2 = get_text(s2, 'red', 1)
        text3 = get_text(s3, self.model_color)
        text4 = get_text(s4, 'red', 1)
        text5 = get_text(s5, self.model_color)
        text6 = get_text(s6, 'red', 1)
        print text0 + text1 + text2 + text3 + text4 + text5 + text6
      inner_error = inner_error_n

      if callback != None:
        s    = '::: calling callback function :::'
        print_text(s, self.model_color)
        callback()

    # calculate total time to compute
    s = time() - t0
    m = s / 60.0
    h = m / 60.0
    s = s % 60
    m = m % 60
    text = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)

  def transient_solve(self, momentum, energy, mass, t_start, t_end, time_step,
                      annotate=False, callback=None):
    """
    """
    s    = '::: performing transient run :::'
    print_text(s, self.model_color)
    
    from varglas.momentum import Momentum
    from varglas.energy   import Energy
    from varglas.mass     import Mass
    
    if momentum.__class__.__base__ != Momentum:
      s = ">>> transient_solve REQUIRES A 'Momentum' INSTANCE, NOT %s <<<"
      print_text(s % type(momentum), 'red', 1)
      sys.exit(1)
    
    if energy.__class__.__base__ != Energy:
      s = ">>> transient_solve REQUIRES AN 'Energy' INSTANCE, NOT %s <<<"
      print_text(s % type(energy), 'red', 1)
      sys.exit(1)
    
    if mass.__class__.__base__ != Mass:
      s = ">>> transient_solve REQUIRES A 'Mass' INSTANCE, NOT %s <<<"
      print_text(s % type(mass), 'red', 1)
      sys.exit(1)
    
    self.init_time_step(time_step)
    self.step_time = []

    t0 = time()
    t  = t_start
   
    # Loop over all times
    while t <= t_end:

      # start the timer :
      tic = time()

      # solve energy :
      energy.solve(annotate=annotate)
      
      # solve velocity :
      momentum.solve(annotate=annotate)

      # solve mass :
      mass.solve(annotate=annotate)

      # update pressure-melting point :
      energy.calc_T_melt(annotate=annotate)

      if callback != None:
        s    = '::: calling callback function :::'
        print_text(s, self.model_color)
        callback()
       
      # increment time step :
      s = '>>> Time: %i yr, CPU time for last dt: %.3f s <<<'
      print_text(s % (t, time()-tic), 'red', 1)

      t += time_step
      self.step_time.append(time() - tic)

    # calculate total time to compute
    s = time() - t0
    m = s / 60.0
    h = m / 60.0
    s = s % 60
    m = m % 60
    text = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)



