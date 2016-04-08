from fenics            import *
from dolfin_adjoint    import *
from cslvr.io          import print_text, get_text, print_min_max
from cslvr.model       import Model
from cslvr.helper      import VerticalBasis
from pylab             import inf, np
import sys

class HybridModel(Model):
  """ 
  """

  def __init__(self, mesh, out_dir='./results/', save_state=False, 
               state=None, use_periodic=False, **gfs_kwargs):
    """
    Create and instance of a 2D model.
    """
    s = "::: INITIALIZING HYBRID MODEL :::"
    print_text(s, cls=self)
    
    Model.__init__(self, mesh, out_dir, use_periodic)
  
  def color(self):
    return '150'

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    s = "    - using 2D periodic boundaries -"
    print_text(s, cls=self)

    xmin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,0].min())
    xmax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,0].max())
    ymin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,1].min())
    ymax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,1].max())
    
    self.use_periodic_boundaries = True
    
    class PeriodicBoundary(SubDomain):
      
      def inside(self, x, on_boundary):
        """
        Return True if on left or bottom boundary AND NOT on one 
        of the two corners (0, 1) and (1, 0).
        """
        return bool((near(x[0], xmin) or near(x[1], ymin)) and \
                    (not ((near(x[0], xmin) and near(x[1], ymax)) \
                     or (near(x[0], xmax) and near(x[1], ymin)))) \
                     and on_boundary)

      def map(self, x, y):
        """
        Remap the values on the top and right sides to the bottom and left
        sides.
        """
        if near(x[0], xmax) and near(x[1], ymax):
          y[0] = x[0] - xmax
          y[1] = x[1] - ymax
        elif near(x[0], xmax):
          y[0] = x[0] - xmax
          y[1] = x[1]
        elif near(x[1], ymax):
          y[0] = x[0]
          y[1] = x[1] - ymax
        else:
          y[0] = x[0]
          y[1] = x[1]

    self.pBC = PeriodicBoundary()
  
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    super(HybridModel, self).set_mesh(mesh)
    
    s = "::: setting 2D mesh :::"
    print_text(s, cls=self)
    
    if self.dim != 2:
      s = ">>> 2D MODEL REQUIRES A 2D MESH, EXITING <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    else:
      self.num_facets = self.mesh.size_global(1)
      self.num_cells  = self.mesh.size_global(2)
      self.dof        = self.mesh.size_global(0)
    s = "    - %iD mesh set, %i cells, %i facets, %i vertices - " \
        % (self.dim, self.num_cells, self.num_facets, self.dof)
    print_text(s, cls=self)
  
  def calculate_boundaries(self, mask=None, lat_mask=None,
                           adot=None, U_mask=None, mark_divide=False):
    """
    Determines the boundaries of the current model mesh
    """
    s = "::: calculating boundaries :::"
    print_text(s, cls=self)

    if lat_mask == None and mark_divide:
      s = ">>> IF PARAMETER <mark_divide> OF calculate_boundaries() IS " + \
          "TRUE, PARAMETER <lat_mask> MUST BE AN EXPRESSION FOR THE LATERAL" + \
          " BOUNDARIES <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
     
    # this function contains markers which may be applied to facets of the mesh
    self.ff      = FacetFunction('size_t', self.mesh)
    self.ff_acc  = FacetFunction('size_t', self.mesh)
    self.cf      = CellFunction('size_t',  self.mesh)
    dofmap       = self.Q.dofmap()

    S = self.S
    B = self.B
    
    # default to all grounded ice :
    if mask == None:
      mask = Expression('1.0', element=self.Q.ufl_element())
    
    # default to all positive accumulation :
    if adot == None:
      adot = Expression('1.0', element=self.Q.ufl_element())
    
    # default to U observations everywhere :
    if U_mask == None:
      U_mask = Expression('1.0', element=self.Q.ufl_element())

    self.init_adot(adot)
    self.init_mask(mask)
    self.init_U_mask(U_mask)

    if mark_divide:
      s = "    - marking the interior facets for incomplete meshes -"
      print_text(s, cls=self)
      self.init_lat_mask(lat_mask)

    self.S.set_allow_extrapolation(True)
    self.B.set_allow_extrapolation(True)
    self.mask.set_allow_extrapolation(True)
    self.adot.set_allow_extrapolation(True)
    self.U_mask.set_allow_extrapolation(True)
    self.lat_mask.set_allow_extrapolation(True)
    
    tol = 1e-6
    
    s = "    - marking boundaries - "
    print_text(s, cls=self)

    class GAMMA_GND(SubDomain):
      def inside(self, x, on_boundary):
        return mask(x[0], x[1]) <= 1.0
    gamma_gnd = GAMMA_GND()

    class GAMMA_FLT(SubDomain):
      def inside(self, x, on_boundary):
        return mask(x[0], x[1]) > 1.0
    gamma_flt = GAMMA_FLT()

    gamma_flt.mark(self.cf, 1)
    gamma_gnd.mark(self.cf, 0)
    
    # mark the divide if desired :  
    if mark_divide:
      class GAMMA_L_DVD(SubDomain):
        def inside(self, x, on_boundary):
          return lat_mask(x[0], x[1]) <= 0.0 and on_boundary
      gamma_l_dvd = GAMMA_L_DVD()
      gamma_l_dvd.mark(self.ff, self.GAMMA_L_DVD)
    
    s = "    - done - "
    print_text(s, cls=self)
    
    self.ds      = Measure('ds')[self.ff]
    self.dx      = Measure('dx')[self.cf]
    
    self.dx_g    = self.dx(0)                # internal above grounded
    self.dx_f    = self.dx(1)                # internal above floating
    self.dBed_g  = self.ds(3)                # grounded bed
    self.dBed_f  = self.ds(5)                # floating bed
    self.dBed    = self.ds(3) + self.ds(5)   # bed
    self.dSrf_gu = self.ds(8)                # grounded with U observations
    self.dSrf_fu = self.ds(9)                # floating with U observations
    self.dSrf_u  = self.ds(8) + self.ds(9)   # surface with U observations
    self.dSrf_g  = self.ds(2) + self.ds(8)   # surface of grounded ice
    self.dSrf_f  = self.ds(6) + self.ds(9)   # surface of floating ice
    self.dSrf    =   self.ds(6) + self.ds(2) \
                   + self.ds(8) + self.ds(9) # surface
    self.dLat_d  = self.ds(7)                # lateral divide
    self.dLat_to = self.ds(4)                # lateral terminus overwater
    self.dLat_tu = self.ds(10)               # lateral terminus underwater
    self.dLat_t  = self.ds(4) + self.ds(10)  # lateral terminus
    self.dLat    =   self.ds(4) + self.ds(7) \
                   + self.ds(10)             # lateral

    if self.save_state:
      s = "::: writing 'ff' FacetFunction to '%sstate.h5' :::"
      print_text(s % self.out_dir, cls=self)
      self.state.write(self.ff,     'ff')

      s = "::: writing 'ff_acc' FacetFunction to '%sstate.h5' :::"
      print_text(s % self.out_dir, cls=self)
      self.state.write(self.ff_acc, 'ff_acc')

      s = "::: writing 'cf' CellFunction to '%sstate.h5' :::"
      print_text(s % self.out_dir, cls=self)
      self.state.write(self.cf,     'cf')

  def generate_function_spaces(self, use_periodic=False, **kwargs):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(HybridModel, self).generate_function_spaces(use_periodic)
   
    # default values if not provided : 
    self.poly_deg = kwargs.get('poly_deg', 2)
    self.N_T      = kwargs.get('N_T',      8)

    s = "::: generating 2D function spaces :::"
    print_text(s, cls=self)
    
    self.HV     = MixedFunctionSpace([self.Q]*2*self.poly_deg) # VELOCITY
    self.Z      = MixedFunctionSpace([self.Q]*self.N_T)        # TEMPERATURE
    
    s = "    - 2D function spaces created - "
    print_text(s, cls=self)
  
  def init_T_T0(self, T, cls=None):
    """
    """
    if cls is None:
      cls = self
    s = "::: initializing temperature in model.Q space to model.Z space :::"
    print_text(s, cls=cls)
    T = project(as_vector([T]*self.N_T), self.Z, annotate=False)
    self.assign_variable(self.T_,  T, cls=cls)
    self.assign_variable(self.T0_, T, cls=cls)
    
  def init_H_H0(self, H, cls=None):
    """
    """
    if cls is None:
      cls = self
    s = "::: initializing thickness :::"
    print_text(s, cls=cls)
    self.assign_variable(self.H,  H, cls=cls)
    self.assign_variable(self.H0, H, cls=cls)

  def init_H_bounds(self, H_min, H_max, cls=None):
    """
    """
    if cls is None:
      cls = self
    s = "::: initializing bounds on thickness :::"
    print_text(s, cls=cls)
    self.assign_variable(self.H_min, H_min, cls=cls)
    self.assign_variable(self.H_max, H_max, cls=cls)

  def init_U(self, U, cls=None):
    """
    """
    # NOTE: this overides model.init_U
    if cls is None:
      cls = self.this
    s = "::: initializing velocity :::"
    print_text(s, cls=cls)
    self.assign_variable(self.U3, U, cls=cls)
  
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(HybridModel, self).initialize_variables()

    s = "::: initializing 2D variables :::"
    print_text(s, cls=self)
    
    # hybrid mass-balance :
    self.H             = Function(self.Q, name='H')
    self.H0            = Function(self.Q, name='H0')
    self.ubar_c        = Function(self.Q, name='ubar_c')
    self.vbar_c        = Function(self.Q, name='vbar_c')
    self.H_min         = Function(self.Q, name='H_min')
    self.H_max         = Function(self.Q, name='H_max')

    # hybrid energy-balance :
    self.deltax        = 1.0 / (self.N_T - 1.0)
    self.sigmas        = np.linspace(0, 1, self.N_T, endpoint=True)
    self.Tm            = Function(self.Z, name='Tm')
    self.T_            = Function(self.Z, name='T_')
    self.T0_           = Function(self.Z, name='T0_')
    self.Ts            = Function(self.Q, name='Ts')
    self.Tb            = Function(self.Q, name='Tb')
    
    # horizontal velocity :
    self.U3            = Function(self.HV, name='U3')
    u_                 = [self.U3[0],   self.U3[2]]
    v_                 = [self.U3[1],   self.U3[3]]
    coef               = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    dcoef              = [lambda s:0.0, lambda s:5*s**3]
    self.u             = VerticalBasis(u_,  coef, dcoef)
    self.v             = VerticalBasis(v_,  coef, dcoef)
    
    # basal velocity :
    self.U3_b          = Function(self.Q3, name='U3_b')
    u_b, v_b, w_b      = self.U3_b.split()
    u_b.rename('u_b', '')
    v_b.rename('v_b', '')
    w_b.rename('w_b', '')
    self.u_b           = u_b
    self.v_b           = v_b
    self.w_b           = w_b
    
    # surface velocity :
    self.U3_s          = Function(self.Q3, name='U3_s')
    u_s, v_s, w_s      = self.U3_s.split()
    u_s.rename('u_b', '')
    v_s.rename('v_b', '')
    w_s.rename('w_b', '')
    self.u_s           = u_s
    self.v_s           = v_s
    self.w_s           = w_s
    
    # SSA-balance : 
    self.U   = Function(self.Q2)
    self.dU  = TrialFunction(self.Q2)
    self.Phi = TestFunction(self.Q2)
    self.Lam = Function(self.Q2)

    # SSA stress-balance variables :
    self.etabar        = Function(self.Q, name='etabar')
    self.ubar          = Function(self.Q, name='ubar')
    self.vbar          = Function(self.Q, name='vbar')
    self.wbar          = Function(self.Q, name='wbar')


