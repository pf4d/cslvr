from fenics            import *
from dolfin_adjoint    import *
from cslvr.io          import print_text, get_text, print_min_max
from cslvr.model       import Model
from pylab             import inf
import sys

class LatModel(Model):
  """ 
  """

  def __init__(self, mesh, out_dir='./results/', use_periodic=False):
    """
    Create and instance of a 2D model.
    """
    s = "::: INITIALIZING LATERAL MODEL :::"
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
    zmin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,1].min())
    zmax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,1].max())
    
    self.use_periodic_boundaries = True
    
    class PeriodicBoundary(SubDomain):
      
      def inside(self, x, on_boundary):
        """
        Return True if on left or bottom boundary AND NOT on one 
        of the two corners (0, 1) and (1, 0).
        """
        return bool((near(x[0], xmin) or near(x[1], zmin)) and \
                    (not ((near(x[0], xmin) and near(x[1], zmax)) \
                     or (near(x[0], xmax) and near(x[1], zmin)))) \
                     and on_boundary)

      def map(self, x, y):
        """
        Remap the values on the top and right sides to the bottom and left
        sides.
        """
        if near(x[0], xmax) and near(x[1], zmax):
          y[0] = x[0] - xmax
          y[1] = x[1] - zmax
        elif near(x[0], xmax):
          y[0] = x[0] - xmax
          y[1] = x[1]
        elif near(x[1], zmax):
          y[0] = x[0]
          y[1] = x[1] - zmax
        else:
          y[0] = x[0]
          y[1] = x[1]

    self.pBC = PeriodicBoundary()
  
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    super(LatModel, self).set_mesh(mesh)
    
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

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(LatModel, self).generate_function_spaces(use_periodic)

    s = "::: generating 2D function spaces :::"
    print_text(s, cls=self)
    
    ## mini elements :
    #self.Bub    = FunctionSpace(self.mesh, "B", 4, 
    #                            constrained_domain=self.pBC)
    #self.MQ     = self.Q + self.Bub
    #M3          = MixedFunctionSpace([self.MQ]*3)
    #self.MV     = MixedFunctionSpace([M3,self.Q])
    
    ## Taylor-Hood elements :
    #V           = VectorFunctionSpace(self.mesh, "CG", 2,
    #                                  constrained_domain=self.pBC)
    #self.MV     = V * self.Q
    
    s = "    - 2D function spaces created - "
    print_text(s, cls=self)

  def generate_stokes_function_spaces(self, kind='mini'):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.

    If <kind> == 'mini', use enriched mini elements.
    If <kind> == 'th',   use Taylor-Hood elements.
    """
    s = "::: generating Stokes function spaces :::"
    print_text(s, cls=self)
        
    # mini elements :
    if kind == 'mini':
      self.Bub    = FunctionSpace(self.mesh, "B", 4, 
                                  constrained_domain=self.pBC)
      self.MQ     = self.Q + self.Bub
      M3          = MixedFunctionSpace([self.MQ]*3)
      self.Q4     = MixedFunctionSpace([M3, self.Q])
      self.Q5     = MixedFunctionSpace([M3, self.Q, self.Q])

    # Taylor-Hood elements :
    elif kind == 'th':
      V           = VectorFunctionSpace(self.mesh, "CG", 2,
                                        constrained_domain=self.pBC)
      self.Q4     = V * self.Q
      self.Q5     = V * self.Q * self.Q
    
    else:
      s = ">>> METHOD generate_stokes_function_spaces <kind> FIELD <<<\n" + \
          ">>> MAY BE 'mini' OR 'th', NOT '%s'. <<<" % kind
      print_text(s, 'red', 1)
      sys.exit(1)

    s = "    - Stokes function spaces created - "
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
    
    # iterate through the facets and mark each if on a boundary :
    #
    #   2 = high slope, upward facing ................ grounded surface
    #   3 = grounded high slope, downward facing ..... grounded base
    #   4 = low slope, upward or downward facing ..... sides
    #   5 = floating ................................. floating base
    #   6 = floating ................................. floating surface
    #   7 = floating sides
    #
    # facet for accumulation :
    #
    #   1 = high slope, upward facing ................ positive adot
    s = "    - iterating through %i facets - " % self.num_facets
    print_text(s, cls=self)
    for f in facets(self.mesh):
      n        = f.normal()
      x_m      = f.midpoint().x()
      y_m      = f.midpoint().y()
      mask_xy  = mask(x_m, y_m)
      
      if   n.y() >=  tol and f.exterior():
        adot_xy   = adot(x_m, y_m)
        U_mask_xy = U_mask(x_m, y_m)
        if adot_xy > 0:
          self.ff_acc[f] = 1
        if mask_xy > 1:
          if U_mask_xy > 0:
            self.ff[f] = 9
          else:
            self.ff[f] = 6
        else:
          if U_mask_xy > 0:
            self.ff[f] = 8
          else:
            self.ff[f] = 2
    
      elif n.y() <= -tol and f.exterior():
        if mask_xy > 1:
          self.ff[f] = 5
        else:
          self.ff[f] = 3
      
      elif n.y() >  -tol and n.y() < tol and f.exterior():
        # if we want to use a basin, we need to mark the interior facets :
        if mark_divide:
          lat_mask_xy = lat_mask(x_m, y_m)
          if lat_mask_xy > 0:
            if y_m > 0:
              self.ff[f] = 4
            else:
              self.ff[f] = 10
          else:
            self.ff[f] = 7
        # otherwise just mark for over (4) and under (10) water :
        else:
          if y_m > 0:
            self.ff[f] = 4
          else:
            self.ff[f] = 10
    
    s = "    - iterating through %i cells - " % self.num_cells
    print_text(s, cls=self)
    for c in cells(self.mesh):
      x_m     = c.midpoint().x()
      y_m     = c.midpoint().y()
      mask_xy = mask(x_m, y_m)

      if mask_xy > 1:
        self.cf[c] = 1
      else:
        self.cf[c] = 0

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
    
  def set_subdomains(self, f):
    """
    Set the facet subdomains FacetFunction self.ff, cell subdomains
    CellFunction self.cf, and accumulation FacetFunction self.ff_acc from
    MeshFunctions saved in an .h5 file <f>.
    """
    s = "::: setting 2D subdomains :::"
    print_text(s, cls=self)

    self.ff     = MeshFunction('size_t', self.mesh)
    self.cf     = MeshFunction('size_t', self.mesh)
    self.ff_acc = MeshFunction('size_t', self.mesh)
    f.read(self.ff,     'ff')
    f.read(self.cf,     'cf')
    f.read(self.ff_acc, 'ff_acc')
    
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

  def deform_mesh_to_geometry(self, S, B):
    """
    Deforms the 2D mesh to the geometry from FEniCS Expressions for the 
    surface <S> and bed <B>.
    """
    s = "::: deforming mesh to geometry :::"
    print_text(s, cls=self)

    self.init_S(S)
    self.init_B(B)
    
    # transform z :
    # thickness = surface - base, z = thickness + base
    # Get the height of the mesh, assumes that the base is at z=0
    max_height  = self.mesh.coordinates()[:,1].max()
    min_height  = self.mesh.coordinates()[:,1].min()
    mesh_height = max_height - min_height
    
    s = "    - iterating through %i vertices - " % self.dof
    print_text(s, cls=self)
    
    for x in self.mesh.coordinates():
      x[1] = (x[1] / mesh_height) * ( + S(x[0],x[1]) \
                                      - B(x[0],x[1]) )
      x[1] = x[1] + B(x[0], x[1])
    s = "    - done - "
    print_text(s, cls=self)
    
  def calc_thickness(self):
    """
    Calculate the continuous thickness field which increases from 0 at the 
    surface to the actual thickness at the bed.
    """
    s = "::: calculating z-varying thickness :::"
    print_text(s, cls=self)
    #H = project(self.S - self.x[2], self.Q, annotate=False)
    H          = self.vert_integrate(Constant(1.0), d='down')
    Hv         = H.vector()
    Hv[Hv < 0] = 0.0
    print_min_max(H, 'H', cls=self)
    return H
  
  def solve_hydrostatic_pressure(self, annotate=False, cls=None):
    """
    Solve for the hydrostatic pressure 'p'.
    """
    if cls is None:
      cls = self
    # solve for vertical velocity :
    s  = "::: solving hydrostatic pressure :::"
    print_text(s, cls=cls)
    rhoi   = self.rhoi
    g      = self.g
    #S      = self.S
    #z      = self.x[2]
    #p      = project(rhoi*g*(S - z), self.Q, annotate=annotate)
    p      = self.vert_integrate(rhoi*g, d='down')
    pv     = p.vector()
    pv[pv < 0] = 0.0
    self.assign_variable(self.p, p, cls=cls)
  
  def vert_extrude(self, u, d='up', Q='self'):
    r"""
    This extrudes a function *u* vertically in the direction *d* = 'up' or
    'down'.  It does this by solving a variational problem:
  
    .. math::
       
       \frac{\partial v}{\partial z} = 0 \hspace{10mm}
       v|_b = u

    """
    s = "::: extruding function %s :::" % d
    print_text(s, cls=self)
    if type(Q) != FunctionSpace:
      Q  = self.Q
    ff   = self.ff
    phi  = TestFunction(Q)
    v    = TrialFunction(Q)
    a    = v.dx(1) * phi * dx
    L    = DOLFIN_EPS * phi * dx
    bcs  = []
    # extrude bed (ff = 3,5) 
    if d == 'up':
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_GND))  # grounded
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_FLT))  # shelves
    # extrude surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_GND))  # grounded
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_FLT))  # shelves
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_U_GND))  # grounded
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_U_FLT))  # shelves
    try:
      name = '%s extruded %s' % (u.name(), d)
    except AttributeError:
      name = 'extruded'
    v    = Function(Q, name=name)
    solve(a == L, v, bcs, annotate=False)
    print_min_max(v, 'extruded function', cls=self)
    return v
  
  def vert_integrate(self, u, d='up', Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
    s = "::: vertically integrating function :::"
    print_text(s, cls=self)

    if type(Q) != FunctionSpace:
      Q = self.Q
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    bcs = []
    # integral is zero on bed (ff = 3,5) 
    if d == 'up':
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_FLT))  # shelves
      a      = v.dx(1) * phi * dx
    # integral is zero on surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_FLT))  # shelves
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_FLT))  # shelves
      a      = -v.dx(1) * phi * dx
    L      = u * phi * dx
    name   = 'value integrated %s' % d 
    v      = Function(Q, name=name)
    solve(a == L, v, bcs, annotate=False)
    print_min_max(v, 'vertically integrated function', cls=self)
    return v

  def calc_vert_average(self, u):
    """
    Calculates the vertical average of a given function space and function.  
    
    :param u: Function representing the model's function space
    :rtype:   Dolfin projection and Function of the vertical average
    """
    H    = self.S - self.B
    uhat = self.vert_integrate(u, d='up')
    s = "::: calculating vertical average :::"
    print_text(s, cls=self)
    ubar = project(uhat/H, self.Q, annotate=False)
    print_min_max(ubar, 'ubar', cls=self)
    name = "vertical average of %s" % u.name()
    ubar.rename(name, '')
    ubar = self.vert_extrude(ubar, d='down')
    return ubar

  def save_subdomain_data(self, h5File):
    """
    save all the subdomain data to hd5f file <h5File>.
    """
    s = "::: writing 'ff' FacetFunction to supplied hdf5 file :::"
    print_text(s, cls=self)
    h5File.write(self.ff,     'ff')

    s = "::: writing 'ff_acc' FacetFunction to supplied hdf5 file :::"
    print_text(s, cls=self)
    h5File.write(self.ff_acc, 'ff_acc')

    s = "::: writing 'cf' CellFunction to supplied hdf5 file :::"
    print_text(s, cls=self)
    h5File.write(self.cf,     'cf')
  
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(LatModel, self).initialize_variables()

    s = "::: initializing 2D variables :::"
    print_text(s, cls=self)

    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = abs(min(0, x[1]))
    self.D = Depth(element=self.Q.ufl_element())
    
    # Enthalpy model
    self.theta_surface = Function(self.Q, name='theta_surface')
    self.theta_float   = Function(self.Q, name='theta_float')
    self.theta_app     = Function(self.Q, name='theta_app')
    self.theta         = Function(self.Q, name='theta')
    self.theta0        = Function(self.Q, name='theta0')
    self.W0            = Function(self.Q, name='W0')
    self.thetahat      = Function(self.Q, name='thetahat')
    self.uhat          = Function(self.Q, name='uhat')
    self.vhat          = Function(self.Q, name='vhat')
    self.what          = Function(self.Q, name='what')
    self.mhat          = Function(self.Q, name='mhat')
    self.rho_b         = Function(self.Q, name='rho_b')

    # Age model   
    self.age           = Function(self.Q, name='age')
    self.a0            = Function(self.Q, name='a0')

    # Surface climate model
    self.precip        = Function(self.Q, name='precip')



