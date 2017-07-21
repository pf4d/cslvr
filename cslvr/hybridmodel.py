from fenics            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import print_text, get_text, print_min_max
from cslvr.model       import Model
from cslvr.helper      import VerticalBasis
from pylab             import inf, np
import sys

class HybridModel(Model):
  """ 
  """
  #FIXME:  THis class should be removed and D2Model used instead.  Also,
  #        the hybrid model is not currently working.


  def __init__(self, mesh, out_dir='./results/', order=1, 
               use_periodic=False):
    """
    Create and instance of a 2D 'hybrid' model.
    """
    s = "::: INITIALIZING HYBRID MODEL :::"
    print_text(s, cls=self)
    Model.__init__(self, mesh, out_dir, order, use_periodic)
  
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
  
  def generate_function_spaces(self, order=1, use_periodic=False, **kwargs):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(HybridModel, self).generate_function_spaces(order, use_periodic)
   
    # default values if not provided : 
    self.poly_deg = kwargs.get('poly_deg', 2)
    self.N_T      = kwargs.get('N_T',      8)

    s = "::: generating 2D function spaces :::"
    print_text(s, cls=self)
    
    # velocity space :
    self.HV = FunctionSpace(self.mesh, MixedElement([self.Q1e]*2*self.poly_deg))

    # temperature space :
    self.Z  = FunctionSpace(self.mesh, MixedElement([self.Q1e]*self.N_T))   
    
    s = "    - 2D function spaces created - "
    print_text(s, cls=self)
  
  def init_T_T0(self, T):
    """
    """
    s = "::: initializing temperature in model.Q space to model.Z space :::"
    print_text(s, cls=self)
    T = project(as_vector([T]*self.N_T), self.Z, annotate=False)
    self.assign_variable(self.T_,  T)
    self.assign_variable(self.T0_, T)
    
  def init_H_H0(self, H):
    """
    """
    s = "::: initializing thickness :::"
    print_text(s, cls=self)
    self.assign_variable(self.H,  H)
    self.assign_variable(self.H0, H)

  def init_H_bounds(self, H_min, H_max):
    """
    """
    s = "::: initializing bounds on thickness :::"
    print_text(s, cls=self)
    self.assign_variable(self.H_min, H_min)
    self.assign_variable(self.H_max, H_max)

  def init_U(self, U):
    """
    """
    # NOTE: this overides model.init_U
    s = "::: initializing velocity :::"
    print_text(s, cls=self)
    self.assign_variable(self.U3, U)
  
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
 


