from fenics            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import print_text, get_text, print_min_max
from cslvr.model       import Model
from pylab             import inf
import sys

class D2Model(Model):
  """ 
  """
  OMEGA_GND   = 0   # internal cells over bedrock
  OMEGA_FLT   = 1   # internal cells over water
  OMEGA_U_GND = 8   # grounded surface with U observations
  OMEGA_U_FLT = 9   # shelf surface with U observations
  GAMMA_L_DVD = 7   # basin divides
  
  # external boundaries :
  ext_boundaries = {GAMMA_L_DVD : 'basin divides'}

  # internal boundaries :
  int_boundaries = {OMEGA_GND   : 'over bedrock',
                    OMEGA_FLT   : 'over water',
                    OMEGA_U_FLT : 'over water with U observations',
                    OMEGA_U_GND : 'over bedrock with U observations'}
  # union :
  boundaries = dict(ext_boundaries, **int_boundaries)

  def __init__(self, mesh, out_dir='./results/', order=1, use_periodic=False):
    """
    Create and instance of a 2D model.
    """
    s = "::: INITIALIZING 2D MODEL :::"
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
    super(D2Model, self).set_mesh(mesh)
    
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

  def generate_function_spaces(self, order=1, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D2Model, self).generate_function_spaces(order, use_periodic)

    s = "::: generating 2D function spaces :::"
    print_text(s, cls=self)
    
    s = "    - 2D function spaces created - "
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

    s = "    - not using a lateral surface mesh - "
    print_text(s, cls=self)

    class OMEGA_GND(SubDomain):
      def inside(self, x, on_boundary):
        return mask(x[0], x[1]) <= 1.0
    gamma_gnd = OMEGA_GND()

    class OMEGA_FLT(SubDomain):
      def inside(self, x, on_boundary):
        return mask(x[0], x[1]) > 1.0
    gamma_flt = OMEGA_FLT()

    class OMEGA_U_GND(SubDomain):
      def inside(self, x, on_boundary):
        return     mask(x[0], x[1]) <= 1.0 and U_mask(x[0], x[1]) <= 0.0
    gamma_u_gnd = OMEGA_U_GND()

    class OMEGA_U_FLT(SubDomain):
      def inside(self, x, on_boundary):
        return     mask(x[0], x[1]) >  1.0 and U_mask(x[0], x[1]) <= 0.0 
    gamma_u_flt = OMEGA_U_FLT()

    gamma_gnd.mark(self.cf,   self.OMEGA_GND) # grounded, no U obs.
    gamma_flt.mark(self.cf,   self.OMEGA_FLT) # floating, no U obs.
    gamma_u_gnd.mark(self.cf, self.OMEGA_U_GND) # grounded, with U obs.
    gamma_u_flt.mark(self.cf, self.OMEGA_U_FLT) # floating, with U obs.
    
    # mark the divide if desired :  
    if mark_divide:
      class GAMMA_L_DVD(SubDomain):
        def inside(self, x, on_boundary):
          return lat_mask(x[0], x[1]) <= 0.0 and on_boundary
      gamma_l_dvd = GAMMA_L_DVD()
      gamma_l_dvd.mark(self.ff, self.GAMMA_L_DVD)
    
    self.N_OMEGA_GND   = sum(self.cf.array() == self.OMEGA_GND)
    self.N_OMEGA_FLT   = sum(self.cf.array() == self.OMEGA_FLT)
    self.N_OMEGA_U_GND = sum(self.cf.array() == self.OMEGA_U_GND)
    self.N_OMEGA_U_FLT = sum(self.cf.array() == self.OMEGA_U_FLT)
    self.N_GAMMA_L_DVD = sum(self.ff.array() == self.GAMMA_L_DVD)
    
    s = "    - done - "
    print_text(s, cls=self)
    
    #s = "    - iterating through %i cells - " % self.num_cells
    #print_text(s, cls=self)
    #for c in cells(self.mesh):
    #  x_m     = c.midpoint().x()
    #  y_m     = c.midpoint().y()
    #  z_m     = c.midpoint().z()
    #  mask_xy = mask(x_m, y_m, z_m)

    #  if mask_xy > 1:
    #    self.cf[c] = 1
    #  else:
    #    self.cf[c] = 0

    #s = "    - done - "
    #print_text(s, cls=self)

    self.set_measures()

  def set_measures(self):
    """
    set the new measure space for facets ``self.ds`` and cells ``self.dx`` for
    the boundaries marked by FacetFunction ``self.ff`` and CellFunction 
    ``self.cf``, respectively.

    Also, the number of cells or facets marked by 
    :func:`calculate_boundaries` :

    * ``self.N_OMEGA_GND``   -- number of cells marked ``self.OMEGA_GND``  
    * ``self.N_OMEGA_FLT``   -- number of cells marked ``self.OMEGA_FLT``  
    * ``self.N_OMEGA_U_GND`` -- number of cells marked ``self.OMEGA_U_GND``
    * ``self.N_OMEGA_U_FLT`` -- number of cells marked ``self.OMEGA_U_FLT``
    * ``self.N_GAMMA_L_DVD`` -- number of facets marked ``self.GAMMA_L_DVD``

    The subdomain corresponding to FacetFunction ``self.ff`` is :

    * ``self.dLat_d``  --  lateral divide

    The subdomains corresponding to CellFunction ``self.cf`` are :

    * ``self.dx_g``    --  internal above grounded
    * ``self.dx_f``    --  internal above floating
    * ``self.dx_u_G``  --  grounded with U observations
    * ``self.dx_u_f``  --  floating with U observations
    """
    # calculate the number of cells and facets that are of a certain type
    # for determining Dirichlet boundaries :
    self.N_OMEGA_GND   = sum(self.cf.array() == self.OMEGA_GND)
    self.N_OMEGA_FLT   = sum(self.cf.array() == self.OMEGA_FLT)
    self.N_OMEGA_U_GND = sum(self.cf.array() == self.OMEGA_U_GND)
    self.N_OMEGA_U_FLT = sum(self.cf.array() == self.OMEGA_U_FLT)
    self.N_GAMMA_L_DVD = sum(self.ff.array() == self.GAMMA_L_DVD)

    # create new measures of integration :
    self.ds      = Measure('ds', subdomain_data=self.ff)
    self.dx      = Measure('dx', subdomain_data=self.cf)
    
    self.dx_g    = self.dx(self.OMEGA_GND)   # above grounded
    self.dx_f    = self.dx(self.OMEGA_FLT)   # above floating
    self.dx_u_g  = self.dx(self.OMEGA_U_GND) # above grounded with U
    self.dx_u_f  = self.dx(self.OMEGA_U_FLT) # above floating with U
    self.dLat_d  = self.ds(self.GAMMA_L_DVD) # lateral divide
    
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D2Model, self).initialize_variables()

    s = "::: initializing 2D variables :::"
    print_text(s, cls=self)

    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = abs(min(0, x[2]))
    self.D = Depth(element=self.Q.ufl_element())
    
    # Enthalpy model
    self.theta0        = Function(self.Q, name='theta0')
    self.W0            = Function(self.Q, name='W0')
    self.thetahat      = Function(self.Q, name='thetahat')
    self.uhat          = Function(self.Q, name='uhat')
    self.vhat          = Function(self.Q, name='vhat')
    self.what          = Function(self.Q, name='what')
    self.mhat          = Function(self.Q, name='mhat')

    # Surface climate model
    self.precip        = Function(self.Q, name='precip')



