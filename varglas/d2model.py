from fenics         import *
from dolfin_adjoint import *
from io             import print_text, get_text, print_min_max
from model_new      import Model
from pylab          import inf, np
import sys

class D2Model(Model):
  """ 
  """

  def __init__(self, out_dir='./results/'):
    """
    Create and instance of a 2D model.
    """
    Model.__init__(self, out_dir)
    self.D2Model_color = '150'
  
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    s = "::: setting 2D mesh :::"
    print_text(s, self.model_color)
    self.mesh       = mesh
    self.flat_mesh  = Mesh(mesh)
    self.dim        = self.mesh.ufl_cell().topological_dimension()
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

  def generate_function_spaces(self, poly_deg=2, N_T=8, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D2Model, self).generate_function_spaces(use_periodic)

    self.poly_deg = poly_deg
    self.N_T      = N_T

    s = "::: generating 2D function spaces :::"
    print_text(s, self.D2Model_color)
    
    self.HV     = MixedFunctionSpace([self.Q]*2*poly_deg) # VELOCITY
    self.Z      = MixedFunctionSpace([self.Q]*N_T)        # TEMPERATURE
    
    s = "    - 2D function spaces created - "
    print_text(s, self.D2Model_color)
    self.initialize_variables()

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
    self.T_            = Function(self.Z, name='T_')
    self.T0_           = Function(self.Z, name='T0_')
    self.Ts            = Function(self.Q, name='Ts')
    self.Tb            = Function(self.Q, name='Tb')
    
    # unified basal velocity :
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


