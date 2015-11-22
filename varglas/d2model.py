from fenics            import *
from dolfin_adjoint    import *
from varglas.io        import print_text, get_text, print_min_max
from varglas.model_new import Model
from pylab             import inf, np
import sys

class D2Model(Model):
  """ 
  """

  def __init__(self, mesh, out_dir='./results/', save_state=False, 
               state=None, use_periodic=False, **gfs_kwargs):
    """
    Create and instance of a 2D model.
    """
    self.D2Model_color = '150'
    
    s = "::: INITIALIZING 2D MODEL :::"
    print_text(s, self.D2Model_color)
    
    Model.__init__(self, mesh, out_dir, save_state, state, 
                   use_periodic, **gfs_kwargs)

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    s = "    - using 2D periodic boundaries -"
    print_text(s, self.D2Model_color)

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
    super(D2Model, self).set_mesh(mesh)
    
    s = "::: setting 2D mesh :::"
    print_text(s, self.D2Model_color)
    
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
    print_text(s, self.D2Model_color)

  def generate_function_spaces(self, use_periodic=False, **kwargs):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D2Model, self).generate_function_spaces(use_periodic)
   
    # default values if not provided : 
    self.poly_deg = kwargs.get('poly_deg', 2)
    self.N_T      = kwargs.get('N_T',      8)

    s = "::: generating 2D function spaces :::"
    print_text(s, self.D2Model_color)
    
    self.HV     = MixedFunctionSpace([self.Q]*2*self.poly_deg) # VELOCITY
    self.Z      = MixedFunctionSpace([self.Q]*self.N_T)        # TEMPERATURE
    
    s = "    - 2D function spaces created - "
    print_text(s, self.D2Model_color)
  
  def init_T_T0(self, T):
    """
    """
    s = "::: initializing temperature :::"
    print_text(s, self.D2Model_color)
    self.assign_variable(self.T_,  T)
    self.assign_variable(self.T0_, T)
    print_min_max(self.T_,  'T_')
    print_min_max(self.T0_, 'T0_')
    
  def init_H_H0(self, H):
    """
    """
    s = "::: initializing thickness :::"
    print_text(s, self.D2Model_color)
    self.assign_variable(self.H,  H)
    self.assign_variable(self.H0, H)
    print_min_max(self.H,  'H')
    print_min_max(self.H0, 'H0')

  def init_H_bounds(self, H_min, H_max):
    """
    """
    s = "::: initializing bounds on thickness :::"
    print_text(s, self.D2Model_color)
    self.assign_variable(self.H_min, H_min)
    self.assign_variable(self.H_max, H_max)
    print_min_max(self.H_min, 'H_min')
    print_min_max(self.H_max, 'H_max')

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D2Model, self).initialize_variables()

    s = "::: initializing 2D variables :::"
    print_text(s, self.D2Model_color)
    
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
    
    # unified basal velocity :
    self.UHV           = Function(self.HV, name='UHV')
    self.U3_b          = Function(self.Q3, name='U3_b')
    u_b, v_b, w_b      = self.U3_b.split()
    u_b.rename('u_b', '')
    v_b.rename('v_b', '')
    w_b.rename('w_b', '')
    self.u_b           = u_b
    self.v_b           = v_b
    self.w_b           = w_b
    
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


