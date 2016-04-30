from fenics            import *
from dolfin_adjoint    import *
from cslvr.io          import print_text, get_text, print_min_max
from cslvr.model       import Model
from scipy.interpolate import interp1d
import numpy               as np
import matplotlib          as plt
import sys


class D1Model(Model):
  """
  Data structure to hold firn model state data.
  """
  def __init__(self, mesh, out_dir='./results/', use_periodic=False):
    """
    Create and instance of a 1D model.
    """
    self.D1Model_color = '150'
    
    s = "::: INITIALIZING 1D MODEL :::"
    print_text(s, self.D1Model_color)
    
    Model.__init__(self, mesh, out_dir, use_periodic)

  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :mesh: :class:`~fenics.Mesh` mesh to be used.
    """
    super(D1Model, self).set_mesh(mesh)

    s = "::: setting 1D mesh :::"
    print_text(s, self.D1Model_color)
  
    self.mesh.init(0,1)
    if self.dim != 1:
      s = ">>> 1D MODEL REQUIRES A 1D MESH, EXITING <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    else:
      self.num_cells = self.mesh.size_global(1)
      self.dof       = self.mesh.size_global(0)
    s = "    - %iD mesh set, %i cells, %i vertices - " \
        % (self.dim, self.num_cells, self.dof)
    print_text(s, self.D1Model_color)

    # set the geometry from the mesh :
    zb = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,0].min())
    zs = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,0].max())
    self.init_S_bc(zs)
    self.init_B_bc(zb)
  
  def refine_mesh(self, divs, i, k,  m=1):
    r"""
    splits ``self.mesh`` *divs* times.
  
    Args:
      :divs: number of times to split mesh
      :i:    fraction of the mesh from the surface to split
      :k:    multiple to decrease i by each step to reduce the distance from the
             surface to split
      :m:    counter used to keep track of calls
  
    """
    if m==1:
      s = "::: entering recursive mesh refinement function :::"
      print_text(s, self.D1Model_color)
    
    mesh  = self.mesh
    S     = self.S_bc
    B     = self.B_bc
  
    if m < divs :
      cell_markers = CellFunction("bool", mesh)
      cell_markers.set_all(False)
      origin = Point(S)
      for cell in cells(mesh):
        p  = cell.midpoint()
        if p.distance(origin) < (S - B) * i:
          cell_markers[cell] = True
      mesh = refine(mesh, cell_markers)
      self.set_mesh(mesh)
      return self.refine_mesh(divs, k/i, k, m=m+1)
    else:
      s = "::: refinement finished, redefining function spaces :::"
      print_text(s, self.D1Model_color)
      self.generate_function_spaces(self.use_periodic_boundaries)
      self.initialize_variables()

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D1Model, self).generate_function_spaces(use_periodic)
    
  def init_S_bc(self, S_bc):
    """
    Set the scalar-value for the surface. 
    """
    s = "::: initializng surface boundary condition :::"
    print_text(s, self.D1Model_color)
    self.S_bc = S_bc
    print_min_max(self.S_bc, 'S_bc')

  def init_B_bc(self, B_bc):
    """
    Set the scalar-value for bed.
    """
    s = "::: initializng bed boundary condition :::"
    print_text(s, self.D1Model_color)
    self.B_bc = B_bc
    print_min_max(self.B_bc, 'B_bc')
  
  def init_theta_surface(self, theta_s):
    """
    """
    s = "::: initializing surface energy :::"
    print_text(s, self.D1Model_color)
    self.theta_surface = theta_s
  
  def init_rho_surface(self, rho_s):
    """
    """
    s = "::: initializing surface density :::"
    print_text(s, self.D1Model_color)
    self.rho_surface = rho_s
  
  def init_w_surface(self, w_s):
    """
    """
    s = "::: initializing surface velocity :::"
    print_text(s, self.D1Model_color)
    self.w_surface = w_s
  
  def init_r_surface(self, r_s):
    """
    """
    s = "::: initializing surface grain-size :::"
    print_text(s, self.D1Model_color)
    self.r_surface = r_s
  
  def init_sigma_surface(self, sigma_s):
    """
    """
    s = "::: initializing surface stress :::"
    print_text(s, self.D1Model_color)
    self.sigma_surface = sigma_s
    
  def init_rho(self, rho):
    """
    """
    s = "::: initializing density :::"
    print_text(s, self.D1Model_color)
    self.assign_variable(self.rho, rho)
    print_min_max(self.rho, 'rho')
 
  def init_adot(self, adot):
    """
    """
    s = "::: initializing accumulation :::"
    print_text(s, self.D1Model_color)
    self.assign_variable(self.adot, adot)
    self.assign_variable(self.bdot, self.rhoi(0) * adot / self.spy(0))
    print_min_max(self.adot, 'adot')
    print_min_max(self.bdot, 'bdot')

  def init_r(self, r):
    """
    """
    s = "::: initializing grain-size :::"
    print_text(s, self.D1Model_color)
    self.assign_variable(self.r, r)
    print_min_max(self.r, 'r^2')
    
  def set_boundary_conditions(self, theta_exp, rho_exp, w_exp, r_exp):
    """
    """
    # enthalpy surface condition :
    self.theta_S     = theta_exp
    
    # density surface condition :
    self.rho_S   = rho_exp

    # velocity surface condition :
    self.w_S     = w_exp

    # grain radius surface condition :
    self.r_S     = r_exp

    # age surface condition (always zero at surface) :
    self.age_S   = Constant(0.0)

    # sigma suface condition (always zero at surface) :
    self.sigma_S = Constant(0.0)
    
    L       = self.L(0) 
    thetasp = self.thetasp(0)
    Tw      = self.Tw(0)
    rhoi    = self.rhoi(0)
    rhow    = self.rhow(0)
    g       = self.g(0)
    etaw    = self.etaw(0)

    # water percentage on the surface :
    class BCW(Expression):
      def __init__(self, thetas, cps, rhos):
        self.thetas   = thetas
        self.cps  = cps
        self.rhos = rhos
      def eval(self, values, x):
        #psis  = 1 - self.rhos/rhoi
        #Wmi   = 0.0057 / (1 - psis) + 0.017         # irr. water content
        #if self.thetas > thetasp:
        #  values[0] = Wmi + (self.thetas - self.cps*Tw) / L
        #else:
        #  values[0] = Wmi
        values[0] = 0.08
    
    # water flux at the surface :
    class BCWFlux(Expression):
      def __init__(self, rs, rhos, thetas, cps):
        self.rs     = rs
        self.rhos   = rhos
        self.thetas     = thetas
        self.cps    = cps
      def eval(self, values, x):
        rhos  = self.rhos
        rs    = self.rs
        #ks    = 0.077 * (1.0/100)**2 * rs * exp(-7.8*rhos/rhow)
        ks    = 0.0602 * exp(-0.00957 * rhos)
        psis  = 1 - rhos/rhoi
        Wmi = 0.0057 / (1 - psis) + 0.017         # irr. water content
        if self.thetas > thetasp:
          omg_s = (self.thetas - self.cps*Tw) / L
        else:
          omg_s = Wmi
        Wes   = (omg_s - Wmi) / (psis - Wmi)
        kws   = ks * Wes**3.0
        Ks    = kws * rhow * g / etaw
        print "::::::::::::::::::::::::KS", Ks, rs, rhos, omg_s
        values[0] = Ks
    self.W_S = BCW(0.0, 0.0, 0.0)
    #self.W_S = BCWFlux(0.0, 0.0, 0.0, 0.0)

  def calculate_boundaries(self):
    """
    Determines the boundaries of the current model mesh
    """
    # this function contains markers which may be applied to facets of the mesh
    self.ff = FacetFunction('size_t', self.mesh)
    tol     = 1e-3

    S = self.S_bc
    B = self.B_bc
   
    # iterate through the facets and mark each if on a boundary :
    #
    #   0 = surface
    #   1 = base
    class Surface(SubDomain):
      def inside(self, x, on_boundary):
        return on_boundary and x[0] == S
    
    class Base(SubDomain):
      def inside(self, x, on_boundary):
        return on_boundary and x[0] == B

    S = Surface()
    B = Base()
    S.mark(self.ff, 0)
    B.mark(self.ff, 1)
    self.ds = Measure('ds')[self.ff]
    self.dx = Measure('dx')(self.mesh)

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D1Model, self).initialize_variables()
    
    s = "::: initializing 1D model variables :::"
    print_text(s, self.model_color)

    self.z     = self.mesh.coordinates()[:,0]
    self.index = np.argsort(self.z)[::-1]
    self.z     = self.z[self.index]
    self.l     = np.diff(self.z)
   
    # surface Dirichlet boundary :
    def surface(x, on_boundary):
      return on_boundary and x[0] == self.S_bc
    
    # base Dirichlet boundary :
    def base(x, on_boundary):
      return on_boundary and x[0] == self.B_bc

    self.surface = surface
    self.base    = base

    #===========================================================================
    # Define variational problem spaces :

    self.m       = Function(self.Q, name='m')
    self.m0      = Function(self.Q, name='m0')
    self.W       = Function(self.Q, name='W')
    self.W0      = Function(self.Q, name='W0')
    self.dW      = Function(self.Q, name='dW')
    self.drhodt  = Function(self.Q, name='drhodt')
    self.theta   = Function(self.Q, name='theta')
    self.theta0  = Function(self.Q, name='theta0')
    self.rho     = Function(self.Q, name='rho')
    self.rhoCoef = Function(self.Q, name='rhoCoef')
    self.bdot    = Function(self.Q, name='bdot')
    self.w       = Function(self.Q, name='w')
    self.w0      = Function(self.Q, name='w0')
    self.age     = Function(self.Q, name='age')
    self.age0    = Function(self.Q, name='age0')
    self.sigma   = Function(self.Q, name='sigma')
    self.r       = Function(self.Q, name='r')
    self.p       = Function(self.Q, name='p')
    self.u       = Function(self.Q, name='u')
    self.ql      = Function(self.Q, name='ql')
    self.Smi     = Function(self.Q, name='Smi')
    self.cif     = Function(self.Q, name='cif')
    self.adot    = Function(self.Q, name='adot')

    self.assign_variable(self.rhoCoef, self.kcHh)
    
    self.lini    = self.l                    # initial height vector
    self.lnew    = self.l.copy()             # previous height vector
    self.t       = 0.0                       # initialize time
    
    self.S_1     = self.S_bc                 # previous time-step surface  
    self.zo      = self.S_bc                 # z-coordinate of initial surface
    self.ht      = [self.S_bc]               # list of surface heights
    self.origHt  = [self.zo]                 # list of initial surface heights

  def update_wBc(self):
    """
    Adjust the velocity at the surface.
    """
    self.w_surface.t    = self.t
    self.w_surface.rhos = self.rhop[0]
    bdotNew             = (self.w_surface.adot * self.rhoi(0)) / self.spy(0)
    self.assign_variable(self.bdot, bdotNew)
  
  def update_rhoBc(self):
    """
    Adjust the density at the surface.
    """
    #dW_s = self.dW[self.index][-1]
    #if self.Ts > self.Tw:
    #  if dW_s > 0:
    #    if self.rho_surface.rhon < self.rhoi(0):
    #      self.rho_surface.rhon += dW_s*self.rhow(0)
    #  else:
    #    self.rho_surface.rhon += dW_s*self.rhow(0)#83.0
    #else:
    #  self.rho_surface.rhon = self.rhos
    self.rho_surface.t = self.t

  def vert_integrate(self, u):
    """
    Integrate <u> from the surface to the bed.
    """
    s    = '::: vertically integrating function :::'
    print_text(s, self.D1Model_color)

    ff  = self.ff
    Q   = self.Q
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    
    # surface Dirichlet boundary :
    def surface(x, on_boundary):
      return on_boundary and x[0] == self.S_bc
    
    # integral is zero on surface
    bcs = DirichletBC(Q, 0.0, surface)
    a      = v.dx(0) * phi * dx
    L      = u * phi * dx
    v      = Function(Q)
    solve(a == L, v, bcs, annotate=False)
    print_min_max(v, 'vertically integrated function')
    #print_min_max(v, 'vertically integrated %s' % u.name())
    return v

  def update_height_history(self):
    """
    track the current height of the firn :
    """
    self.ht.append(self.z[0])

    # calculate the new height of original surface by interpolating the 
    # vertical speed from w and keeping the ratio intact :
    wp      = self.w.vector().array()
    interp  = interp1d(self.z, wp, bounds_error=False, fill_value=wp[0])
    wzo     = interp(self.zo)
    dt      = self.time_step(0)
    zs      = self.z[0]
    zb      = self.z[-1]
    zs_1    = self.S_1
    zo      = self.zo
    #self.zo = zo * (zs - zb) / (zs_1 - zb) + wzo * dt
    self.zo = zo + (zs - zs_1) + wzo * dt
    
    # track original height :
    if self.zo > zb:
      self.origHt.append(self.zo)
    
    # update the previous time steps' surface height :
    self.S_1  = self.z[0]

  def set_ini_conv(self, ex):
    """
    sets the firn model's initial state based on files in data/enthalpy folder.
    """
    ex = str(ex)

    self.rhoin = genfromtxt("data/fmic/initial/initial" + ex + "/rho.txt")
    self.rho   = self.rhoin
    self.w     = genfromtxt("data/fmic/initial/initial" + ex + "/w.txt")
    self.z     = genfromtxt("data/fmic/initial/initial" + ex + "/z.txt")
    self.a     = genfromtxt("data/fmic/initial/initial" + ex + "/a.txt")
    self.theta = genfromtxt("data/fmic/initial/initial" + ex + "/theta.txt")
    self.lin   = genfromtxt("data/fmic/initial/initial" + ex + "/l.txt")
    
    self.S_1    = self.z[0]                # previous time-step surface  
    self.zo     = self.z[0]                # z-coordinate of initial surface
    self.ht     = [self.z[0]]              # list of surface heights
    self.origHt = [self.z[0]]              # list of initial surface heights
    self.Ts     = self.theta[0] / self.c[0]    # temperature of surface
  
    self.assign_variable(self.rho_i,   self.rho)
    self.assign_variable(self.theta_i, self.theta)
    self.assign_variable(self.w_i,     self.w)
    self.assign_variable(self.aF,      self.a)
    self.assign_variable(self.a0,      self.a)
  
  def transient_solve(self, momentum, energy, t_start, t_mid, t_end, time_step,
                      dt_list=None, annotate=False, callback=None):
    """
    """
    s    = '::: performing transient run :::'
    print_text(s, self.D1Model_color)
    
    from cslvr.momentum import Momentum
    from cslvr.energy   import Energy
      
    if momentum.__class__.__base__ != Momentum:
      s = ">>> transient_solve REQUIRES A 'Momentum' CHILD INSTANCE, NOT %s <<<"
      print_text(s % momentum.__class__.__base__, 'red', 1)
      sys.exit(1)
    
    if energy.__class__.__base__ != Energy:
      s = ">>> transient_solve REQUIRES AN 'Energy' CHILD INSTANCE, NOT %s <<<"
      print_text(s % energy.__class__.__base__, 'red', 1)
      sys.exit(1)
   
    # form time steps : 
    if dt_list != None:
      numt1   = (t_mid - t_start) / dt_list[0] + 1  # number of time steps
      numt2   = (t_end - t_mid) / dt_list[1] + 1    # number of time steps
      times1  = np.linspace(t_start, t_mid, numt1)  # array of times in seconds
      times2  = np.linspace(t_mid,   t_end, numt2)  # array of times in seconds
      dt1     = dt_list[0] * np.ones(len(times1))
      dt2     = dt_list[1] * np.ones(len(times2))
      times   = np.hstack((times1, times2))
      dts     = np.hstack((dt1, dt2))
    
    else: 
      numt   = (t_end - t_start) / time_step + 1    # number of time steps
      times  = np.linspace(t_start, t_end, numt)    # array of times in seconds
      dts    = time_step * np.ones(len(times))
   
    self.t     = t_start
    self.times = times
    self.dts   = dts
    
    t0 = time()

    for t,dt in zip(times[1:], dts[1:]):

      # start the timer :
      tic = time()
      
      # update timestep :
      self.init_time_step(dt)

      # solve momentum :
      momentum.solve(annotate=annotate)

      # solve energy :
      energy.solve(annotate=annotate)

      # solve mass :
      #mass.solve(annotate=annotate)
      
      # update model parameters :
      if t != times[-1]:
         self.t = t
         momentum.U0.assign(momentum.U, annotate=annotate)
         self.theta0.assign(self.theta, annotate=annotate)
         self.W0.assign(self.W, annotate=annotate)
         self.w0.assign(self.w, annotate=annotate)
         self.age0.assign(self.age, annotate=annotate)
         self.m0.assign(self.m, annotate=annotate)
      
      if callback != None:
        s    = '::: calling callback function :::'
        print_text(s, self.D1Model_color)
        callback()
        
      # increment time step :
      s = '>>> Time: %g yr, CPU time for last dt: %.3f s <<<'
      print_text(s % (t / self.spy(0), time()-tic), 'red', 1)
    
    # calculate total time to compute
    s = time() - t0
    m = s / 60.0
    h = m / 60.0
    s = s % 60
    m = m % 60
    text = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)



