from fenics            import *
from dolfin_adjoint    import *
from varglas.io        import print_text, get_text, print_min_max
from varglas.model_new import Model
from scipy.interpolate import interp1d
import numpy               as np
import matplotlib          as plt
import sys


class D1Model(Model):
  """
  Data structure to hold firn model state data.
  """
  def __init__(self, out_dir='./results/'):
    """
    Create and instance of a 1D model.
    """
    Model.__init__(self, out_dir)
    self.D1Model_color = '150'

  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    s = "::: setting 1D mesh :::"
    print_text(s, self.D1Model_color)
    self.mesh = mesh
    self.dim  = self.mesh.ufl_cell().topological_dimension()
  
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
    """
    splits the mesh a <divs> times.
  
    INPUTS:
      divs - number of times to split mesh
      i    - fraction of the mesh from the surface to split
      k    - multiple to decrease i by each step to reduce the distance from the
             surface to split
      m    - counter used to keep track of calls
  
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

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D1Model, self).generate_function_spaces(use_periodic)
    self.initialize_variables()
    
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
  
  def init_H_surface(self, H_s):
    """
    """
    s = "::: initializing surface energy :::"
    print_text(s, self.D1Model_color)
    self.H_surface = H_s
  
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
    self.adot = adot
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
    
  def set_boundary_conditions(self, H_exp, rho_exp, w_exp, r_exp):
    """
    """
    # enthalpy surface condition :
    self.H_S     = H_exp
    
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
    
    L    = self.L(0) 
    Hsp  = self.Hsp(0)
    Tw   = self.Tw(0)
    rhoi = self.rhoi(0)
    rhow = self.rhow(0)
    g    = self.g(0)
    etaw = self.etaw(0)

    # water percentage on the surface :
    class BComega(Expression):
      def __init__(self, Hs, cps, rhos):
        self.Hs   = Hs
        self.cps  = cps
        self.rhos = rhos
      def eval(self, values, x):
        #psis  = 1 - self.rhos/rhoi
        #Wmi   = 0.0057 / (1 - psis) + 0.017         # irr. water content
        #if self.Hs > Hsp:
        #  values[0] = Wmi + (self.Hs - self.cps*Tw) / L
        #else:
        #  values[0] = Wmi
        values[0] = 0.08
    
    # water flux at the surface :
    class BComegaFlux(Expression):
      def __init__(self, rs, rhos, Hs, cps):
        self.rs     = rs
        self.rhos   = rhos
        self.Hs     = Hs
        self.cps    = cps
      def eval(self, values, x):
        rhos  = self.rhos
        rs    = self.rs
        #ks    = 0.077 * (1.0/100)**2 * rs * exp(-7.8*rhos/rhow)
        ks    = 0.0602 * exp(-0.00957 * rhos)
        psis  = 1 - rhos/rhoi
        Wmi = 0.0057 / (1 - psis) + 0.017         # irr. water content
        if self.Hs > Hsp:
          omg_s = (self.Hs - self.cps*Tw) / L
        else:
          omg_s = Wmi
        Wes   = (omg_s - Wmi) / (psis - Wmi)
        kws   = ks * Wes**3.0
        Ks    = kws * rhow * g / etaw
        print "::::::::::::::::::::::::KS", Ks, rs, rhos, omg_s
        values[0] = Ks
    self.omega_S = BComega(0.0, 0.0, 0.0)
    #self.omega_S = BComegaFlux(0.0, 0.0, 0.0, 0.0)

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
    self.ds = ds[self.ff]

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D1Model, self).initialize_variables()

    self.z     = self.mesh.coordinates()[:,0]
    self.index = np.argsort(self.z)[::-1]
    self.z     = self.z[self.index]
    self.l     = np.diff(self.z)
    self.x     = SpatialCoordinate(self.mesh)[0]
   
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

    self.m       = Function(self.Q)
    self.m_1     = Function(self.Q)
    self.T       = Function(self.Q)
    self.omega   = Function(self.Q)
    self.omega_1 = Function(self.Q)
    self.domega  = Function(self.Q)
    self.drhodt  = Function(self.Q)
    self.Kcoef   = Function(self.Q)
    self.H       = Function(self.Q)
    self.H_1     = Function(self.Q)
    self.rho     = Function(self.Q)
    self.rho_1   = Function(self.Q)
    self.rhoCoef = Function(self.Q)
    self.bdot    = Function(self.Q)
    self.w       = Function(self.Q)
    self.w_1     = Function(self.Q)
    self.a       = Function(self.Q)
    self.a_1     = Function(self.Q)
    self.sigma   = Function(self.Q)
    self.sigma_1 = Function(self.Q)
    self.r       = Function(self.Q)
    self.r_1     = Function(self.Q)
    self.p       = Function(self.Q)
    self.u       = Function(self.Q)
    self.ql      = Function(self.Q)
    self.Smi     = Function(self.Q)

    self.assign_variable(self.Kcoef,   1.0)
    self.assign_variable(self.rhoCoef, self.kcHh)
    
    self.lini    = self.l                    # initial height vector
    self.lnew    = self.l.copy()             # previous height vector
    self.t       = 0.0                       # initialize time
    
    self.Hp      = self.H.vector().array()
    self.Tp      = self.T.vector().array()
    self.omegap  = self.omega.vector().array()
    self.rhop    = self.rho.vector().array()
    self.drhodtp = self.drhodt.vector().array()
    self.ap      = self.a.vector().array()
    self.wp      = self.w.vector().array()
    self.kp      = 2.1*(self.rhop / self.rhoi(0))**2
    self.cp      = self.ci(0) * np.ones(self.dof)
    self.rp      = self.r.vector().array()
    self.rhoinp  = self.rhop
    self.agep    = np.zeros(self.dof)
    self.pp      = np.zeros(self.dof)
    self.up      = np.zeros(self.dof)
    self.Smip    = np.zeros(self.dof)
    
    self.S_1     = self.S_bc                 # previous time-step surface  
    self.zo      = self.S_bc                 # z-coordinate of initial surface
    self.ht      = [self.S_bc]               # list of surface heights
    self.origHt  = [self.zo]                 # list of initial surface heights
    self.Ts      = self.Hp[0] / self.cp[0]   # temperature of surface

  def update_Hbc(self): 
    """
    Adjust the enthalpy at the surface.
    """
    self.H_surface.t = self.t
    self.H_surface.c = self.cp[0]
  
  def update_Tbc(self): 
    """
    Adjust the enthalpy at the surface.
    """
    self.T_surface.t = self.t
 
  def update_omegaBc(self): 
    """
    Adjust the water-content at the surface.
    """
    self.omega_surface.Hs   = self.Hp[0]
    self.omega_surface.cps  = self.cp[0]
    self.omega_surface.rs   = self.rp[0]
    self.omega_surface.rhos = self.rhop[0]
      
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
    #domega_s = self.domega[self.index][-1]
    #if self.Ts > self.Tw:
    #  if domega_s > 0:
    #    if self.rho_surface.rhon < self.rhoi(0):
    #      self.rho_surface.rhon += domega_s*self.rhow(0)
    #  else:
    #    self.rho_surface.rhon += domega_s*self.rhow(0)#83.0
    #else:
    #  self.rho_surface.rhon = self.rhos
    self.rho_surface.t = self.t

  def update_vars(self, t):
    """
    Project the variables onto the space V and update firn object.
    """
    self.t       = t
    self.Hp      = self.H.vector().array()
    self.rhop    = self.rho.vector().array()
    self.wp      = self.w.vector().array()
    self.ap      = self.a.vector().array()
    self.Tp      = self.T.vector().array()
    self.omegap  = self.omega.vector().array()
    self.rp      = self.r.vector().array()
    self.pp      = self.p.vector().array()
    self.up      = self.u.vector().array()
    self.Smip    = self.Smi.vector().array()
    self.Ts      = self.Hp[0] / self.cp[0]
  
  def vert_integrate(self, u):
    """
    Integrate <u> from the surface to the bed.
    """
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
    solve(a == L, v, bcs)
    return v

  def update_height_history(self):
    """
    track the current height of the firn :
    """
    self.ht.append(self.z[-1])

    # calculate the new height of original surface by interpolating the 
    # vertical speed from w and keeping the ratio intact :
    interp  = interp1d(self.z, self.wp,
                       bounds_error=False,
                       fill_value=self.wp[0])
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
    self.H     = genfromtxt("data/fmic/initial/initial" + ex + "/H.txt")
    self.lin   = genfromtxt("data/fmic/initial/initial" + ex + "/l.txt")
    
    self.S_1    = self.z[0]                # previous time-step surface  
    self.zo     = self.z[0]                # z-coordinate of initial surface
    self.ht     = [self.z[0]]              # list of surface heights
    self.origHt = [self.z[0]]              # list of initial surface heights
    self.Ts     = self.H[0] / self.c[0]    # temperature of surface
  
    self.assign_variable(self.rho_i, self.rho)
    self.assign_variable(self.H_i,   self.H)
    self.assign_variable(self.w_i,   self.w)
    self.assign_variable(self.aF,    self.a)
    self.assign_variable(self.a_1,   self.a)
  
  def transient_solve(self, momentum, energy, t_start, t_mid, t_end, time_step,
                      dt_list=None, annotate=False, callback=None):
    """
    """
    s    = '::: performing transient run :::'
    print_text(s, self.D1Model_color)
    
    from varglas.momentum import Momentum
    from varglas.energy   import Energy
      
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
      
      # update firn object :
      self.update_vars(t)
      self.update_height_history()
      #if config['free_surface']['on']:
      #  if dt_list != None:
      #    if t > tm+dt:
      #      self.update_height()
      #  else:
      #    self.update_height()
      
      # update model parameters :
      if t != times[-1]:
         momentum.U_1.assign(momentum.U)
         self.H_1.assign(self.H)
         self.omega_1.assign(self.omega)
         self.w_1.assign(self.w)
         self.a_1.assign(self.a)
         self.m_1.assign(self.m)
      
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



