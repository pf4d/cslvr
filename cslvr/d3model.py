from fenics            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import print_text, get_text, print_min_max
from cslvr.model       import Model
from cslvr.helper      import Boundary
from pylab             import inf
import sys

class D3Model(Model):
  """ 
  """
  OMEGA_GND   = 0   # internal cells over bedrock
  OMEGA_FLT   = 1   # internal cells over water
  GAMMA_S_GND = 2   # grounded upper surface
  GAMMA_B_GND = 3   # grounded lower surface (bedrock)
  GAMMA_S_FLT = 6   # shelf upper surface
  GAMMA_B_FLT = 5   # shelf lower surface
  GAMMA_L_DVD = 7   # basin divides
  GAMMA_L_OVR = 4   # terminus over water
  GAMMA_L_UDR = 10  # terminus under water
  GAMMA_U_GND = 8   # grounded surface with U observations
  GAMMA_U_FLT = 9   # shelf surface with U observations
  GAMMA_ACC   = 1   # areas with positive surface accumulation

  # external boundaries :
  ext_boundaries = {GAMMA_S_GND : 'grounded upper surface',
                    GAMMA_B_GND : 'grounded lower surface (bedrock)',
                    GAMMA_S_FLT : 'shelf upper surface',
                    GAMMA_B_FLT : 'shelf lower surface',
                    GAMMA_L_DVD : 'basin divides',
                    GAMMA_L_OVR : 'terminus over water',
                    GAMMA_L_UDR : 'terminus under water',
                    GAMMA_U_GND : 'grounded upper surface with U observations',
                    GAMMA_U_FLT : 'shelf upper surface with U observations',
                    GAMMA_ACC   : 'upper surface with accumulation'}

  # internal boundaries :
  int_boundaries = {OMEGA_GND   : 'internal cells located over bedrock',
                    OMEGA_FLT   : 'internal cells located over water'}

  # union :
  boundaries = {'OMEGA' : int_boundaries,
                'GAMMA' : ext_boundaries}

  def __init__(self, mesh, out_dir='./results/', order=1, 
               use_periodic=False):
    """
    Create and instance of a 3D model.
    """
    s = "::: INITIALIZING 3D MODEL :::"
    print_text(s, cls=self)
    
    Model.__init__(self, mesh, out_dir, order, use_periodic)
  
  def color(self):
    return '130'

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    s = "    - using 3D periodic boundaries -"
    print_text(s, cls=self)

    xmin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,0].min())
    xmax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,0].max())
    ymin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,1].min())
    ymax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,1].max())
    
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
          y[2] = x[2]
        elif near(x[0], xmax):
          y[0] = x[0] - xmax
          y[1] = x[1]
          y[2] = x[2]
        elif near(x[1], ymax):
          y[0] = x[0]
          y[1] = x[1] - ymax
          y[2] = x[2]
        else:
          y[0] = x[0]
          y[1] = x[1]
          y[2] = x[2]

    self.pBC = PeriodicBoundary()
  
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    super(D3Model, self).set_mesh(mesh)
    
    s = "::: setting 3D mesh :::"
    print_text(s, cls=self)
    
    self.mesh.init(1,2)
    if self.dim != 3:
      s = ">>> 3D MODEL REQUIRES A 3D MESH, EXITING <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    else:
      self.num_facets   = self.mesh.num_facets()
      self.num_cells    = self.mesh.num_cells()
      self.num_vertices = self.mesh.num_vertices()
    s = "    - %iD mesh set, %i cells, %i facets, %i vertices - " \
        % (self.dim, self.num_cells, self.num_facets, self.num_vertices)
    print_text(s, cls=self)
  
  def set_flat_mesh(self, flat_mesh):
    """
    Sets the flat_mesh for 3D free-surface.
    
    :param flat_mesh : Dolfin mesh to be written
    """
    s = "::: setting 3D ``flat'' mesh :::"
    print_text(s, cls=self)

    self.flat_mesh = flat_mesh
    self.flat_dim  = self.flat_mesh.ufl_cell().topological_dimension()
    if self.flat_dim != 3:
      s = ">>> 3D MODEL REQUIRES A 3D FLAT_MESH, EXITING <<<"
      print_text(s, 'red', 1)
      sys.exit(1)

  def set_srf_mesh(self, srfmesh):
    """
    Set the surface boundary mesh.
    """
    s = "::: setting surface boundary mesh :::"
    print_text(s, cls=self)

    if isinstance(srfmesh, dolfin.cpp.io.HDF5File):
      self.srfmesh = Mesh()
      srfmesh.read(self.srfmesh, 'srfmesh', False)

    elif isinstance(srfmesh, dolfin.cpp.mesh.Mesh):
      self.srfmesh = srfmesh

  def set_bed_mesh(self, bedmesh):
    """
    Set the basal boundary mesh.
    """
    s = "::: setting basal boundary mesh :::"
    print_text(s, cls=self)

    if isinstance(bedmesh, dolfin.cpp.io.HDF5File):
      self.bedmesh = Mesh()
      bedmesh.read(self.bedmesh, 'bedmesh', False)

    elif isinstance(bedmesh, dolfin.cpp.mesh.Mesh):
      self.bedmesh = bedmesh

  def set_lat_mesh(self, latmesh):
    """
    Set the lateral boundary mesh.
    """
    s = "::: setting lateral boundary mesh :::"
    print_text(s, cls=self)

    if isinstance(latmesh, dolfin.cpp.io.HDF5File):
      self.latmesh = Mesh()
      latmesh.read(self.latmesh, 'latmesh', False)

    elif isinstance(latmesh, dolfin.cpp.mesh.Mesh):
      self.latmesh = latmesh

    self.Q_lat = FunctionSpace(self.latmesh, 'CG', 1)

  def set_dvd_mesh(self, dvdmesh):
    """
    Set the lateral divide boundary mesh.
    """
    s = "::: setting lateral divide boundary mesh :::"
    print_text(s, cls=self)

    if isinstance(dvdmesh, dolfin.cpp.io.HDF5File):
      self.dvdmesh = Mesh()
      dvdmesh.read(self.dvdmesh, 'dvdmesh', False)

    elif isinstance(dvdmesh, dolfin.cpp.mesh.Mesh):
      self.dvdmesh = dvdmesh

    self.Q_dvd = FunctionSpace(self.dvdmesh, 'CG', 1)

  def generate_function_spaces(self):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D3Model, self).generate_function_spaces()

    s = "::: generating 3D function spaces :::"
    print_text(s, cls=self)
    
    if self.use_periodic:
      self.Q4    = FunctionSpace(self.mesh, self.QM4e,
                                 constrained_domain=self.pBC)
      self.QTH3  = FunctionSpace(self.mesh, self.QTH3e,
                                 constrained_domain=self.pBC)
    
    s = "    - 3D function spaces created - "
    print_text(s, cls=self)

  def calculate_boundaries(self, mask=None, lat_mask=None, adot=None,
                           U_mask=None, mark_divide=False):
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
    self.ff      = MeshFunction('size_t', self.mesh, 2, 0)
    self.ff_acc  = MeshFunction('size_t', self.mesh, 2, 0)
    self.cf      = MeshFunction('size_t', self.mesh, 3, 0)
    dofmap       = self.Q.dofmap()
    
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
      z_m      = f.midpoint().z()
      mask_xy  = mask(x_m, y_m, z_m)
      
      if   n.z() >=  tol and f.exterior():
        adot_xy   = adot(x_m, y_m, z_m)
        U_mask_xy = U_mask(x_m, y_m, z_m)
        if adot_xy > 0:
          self.ff_acc[f] = self.GAMMA_ACC
        if mask_xy > 1:
          if U_mask_xy > 0:
            self.ff[f] = self.GAMMA_U_FLT
          else:
            self.ff[f] = self.GAMMA_S_FLT
        else:
          if U_mask_xy > 0:
            self.ff[f] = self.GAMMA_U_GND
          else:
            self.ff[f] = self.GAMMA_S_GND
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 1:
          self.ff[f] = self.GAMMA_B_FLT
        else:
          self.ff[f] = self.GAMMA_B_GND
      
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        # if we want to use a basin, we need to mark the interior facets :
        if mark_divide:
          lat_mask_xy = lat_mask(x_m, y_m, z_m)
          if lat_mask_xy > 0:
            if z_m > 0:
              self.ff[f] = self.GAMMA_L_OVR
            else:
              self.ff[f] = self.GAMMA_L_UDR
          else:
            self.ff[f] = self.GAMMA_L_DVD
        # otherwise just mark for over (4) and under (10) water :
        else:
          if z_m > 0:
            self.ff[f] = self.GAMMA_L_OVR
          else:
            self.ff[f] = self.GAMMA_L_UDR
    
    s = "    - done - "
    print_text(s, cls=self)
    
    s = "    - iterating through %i cells - " % self.num_cells
    print_text(s, cls=self)
    for c in cells(self.mesh):
      x_m     = c.midpoint().x()
      y_m     = c.midpoint().y()
      z_m     = c.midpoint().z()
      mask_xy = mask(x_m, y_m, z_m)

      if mask_xy > 1:
        self.cf[c] = self.OMEGA_FLT
      else:
        self.cf[c] = self.OMEGA_GND
    
    s = "    - done - "
    print_text(s, cls=self)
  
    self.set_measures()

  def set_measures(self):
    """
    set the new measure space for facets ``self.ds`` and cells ``self.dx`` for
    the boundaries marked by FacetFunction ``self.ff`` and CellFunction 
    ``self.cf``, respectively.

    Also, the number of facets marked by 
    :func:`calculate_boundaries` :

    * ``self.N_OMEGA_GND``   -- number of cells marked ``self.OMEGA_GND``  
    * ``self.N_OMEGA_FLT``   -- number of cells marked ``self.OMEGA_FLT``  
    * ``self.N_GAMMA_S_GND`` -- number of facets marked ``self.GAMMA_S_GND``
    * ``self.N_GAMMA_B_GND`` -- number of facets marked ``self.GAMMA_B_GND``
    * ``self.N_GAMMA_S_FLT`` -- number of facets marked ``self.GAMMA_S_FLT``
    * ``self.N_GAMMA_B_FLT`` -- number of facets marked ``self.GAMMA_B_FLT``
    * ``self.N_GAMMA_L_DVD`` -- number of facets marked ``self.GAMMA_L_DVD``
    * ``self.N_GAMMA_L_OVR`` -- number of facets marked ``self.GAMMA_L_OVR``
    * ``self.N_GAMMA_L_UDR`` -- number of facets marked ``self.GAMMA_L_UDR``
    * ``self.N_GAMMA_U_GND`` -- number of facets marked ``self.GAMMA_U_GND``
    * ``self.N_GAMMA_U_FLT`` -- number of facets marked ``self.GAMMA_U_FLT``

    The subdomains corresponding to FacetFunction ``self.ff`` are :

    * ``self.dBed_g``  --  grounded bed
    * ``self.dBed_f``  --  floating bed
    * ``self.dBed``    --  bed
    * ``self.dSrf_gu`` --  grounded with U observations
    * ``self.dSrf_fu`` --  floating with U observations
    * ``self.dSrf_u``  --  surface with U observations
    * ``self.dSrf_g``  --  surface of grounded ice
    * ``self.dSrf_f``  --  surface of floating ice
    * ``self.dSrf``    --  surface
    * ``self.dLat_d``  --  lateral divide
    * ``self.dLat_to`` --  lateral terminus overwater
    * ``self.dLat_tu`` --  lateral terminus underwater
    * ``self.dLat_t``  --  lateral terminus
    * ``self.dLat``    --  lateral

    The subdomains corresponding to CellFunction ``self.cf`` are :

    * ``self.dx_g``    --  internal above grounded
    * ``self.dx_f``    --  internal above floating
    """
    # calculate the number of cells and facets that are of a certain type
    # for determining Dirichlet boundaries :
    self.N_OMEGA_GND   = sum(self.cf.array()     == self.OMEGA_GND)
    self.N_OMEGA_FLT   = sum(self.cf.array()     == self.OMEGA_FLT)
    self.N_GAMMA_S_GND = sum(self.ff.array()     == self.GAMMA_S_GND)
    self.N_GAMMA_B_GND = sum(self.ff.array()     == self.GAMMA_B_GND)
    self.N_GAMMA_S_FLT = sum(self.ff.array()     == self.GAMMA_S_FLT)
    self.N_GAMMA_B_FLT = sum(self.ff.array()     == self.GAMMA_B_FLT)
    self.N_GAMMA_L_DVD = sum(self.ff.array()     == self.GAMMA_L_DVD)
    self.N_GAMMA_L_OVR = sum(self.ff.array()     == self.GAMMA_L_OVR)
    self.N_GAMMA_L_UDR = sum(self.ff.array()     == self.GAMMA_L_UDR)
    self.N_GAMMA_U_GND = sum(self.ff.array()     == self.GAMMA_U_GND)
    self.N_GAMMA_U_FLT = sum(self.ff.array()     == self.GAMMA_U_FLT)
    self.N_GAMMA_ACC   = sum(self.ff_acc.array() == self.GAMMA_ACC)

    # create new measures of integration :
    self.ds      = Measure('ds', subdomain_data=self.ff)
    self.dx      = Measure('dx', subdomain_data=self.cf)
    
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
    
    self.dOmega     = Boundary(self.dx, [0,1],
                      'entire interior')
    self.dOmega_g   = Boundary(self.dx, [0],
                      'interior above grounded ice')
    self.dOmega_w   = Boundary(self.dx, [1],
                      'interior above floating ice')
    self.dGamma     = Boundary(self.ds, [2,3,4,5,6,7,8,9,2,8,9,10],
                      'entire exterior')
    self.dGamma_bg  = Boundary(self.ds, [3],
                      'grounded basal surface')
    self.dGamma_bw  = Boundary(self.ds, [5],
                      'floating basal surface')
    self.dGamma_b   = Boundary(self.ds, [3,5],
                      'entire basal surface')
    self.dGamma_sgu = Boundary(self.ds, [8],
                      'upper surface with U observations above grounded ice')
    self.dGamma_swu = Boundary(self.ds, [9],
                      'upper surface with U observations above floating ice')
    self.dGamma_su  = Boundary(self.ds, [8,9],
                      'entire upper surface with U observations')
    self.dGamma_sg  = Boundary(self.ds, [2,8],
                      'upper surface above grounded ice')
    self.dGamma_sw  = Boundary(self.ds, [6,9],
                      'upper surface above floating ice')
    self.dGamma_s   = Boundary(self.ds, [6,2,8,9],
                      'entire upper surface')
    self.dGamma_ld  = Boundary(self.ds, [7],
                      'lateral interior surface')
    self.dGamma_lto = Boundary(self.ds, [4],
                      'exterior lateral surface above water')
    self.dGamma_ltu = Boundary(self.ds, [10],
                      'exterior lateral surface below= water')
    self.dGamma_lt  = Boundary(self.ds, [4,10],
                      'entire exterior lateral surface')
    self.dGamma_l   = Boundary(self.ds, [4,7,10],
                      'entire exterior and interior lateral surface')

  def deform_mesh_to_geometry(self, S, B):
    """
    Deforms the 3D mesh to the geometry from FEniCS Expressions for the 
    surface <S> and bed <B>.
    """
    s = "::: deforming mesh to geometry :::"
    print_text(s, cls=self)

    self.init_S(S)
    self.init_B(B)
    
    # transform z :
    # thickness = surface - base, z = thickness + base
    # Get the height of the mesh, assumes that the base is at z=0
    max_height  = self.mesh.coordinates()[:,2].max()
    min_height  = self.mesh.coordinates()[:,2].min()
    mesh_height = max_height - min_height
    
    s = "    - iterating through %i vertices - " % self.num_vertices
    print_text(s, cls=self)
    
    for x in self.mesh.coordinates():
      x[2] = (x[2] / mesh_height) * ( + S(x[0],x[1],x[2]) \
                                      - B(x[0],x[1],x[2]) )
      x[2] = x[2] + B(x[0], x[1], x[2])
    s = "    - done - "
    print_text(s, cls=self)

  def form_srf_mesh(self):
    """
    sets self.srfmesh, the surface boundary mesh for this model instance.
    """
    s = "::: extracting surface mesh :::"
    print_text(s, cls=self)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = MeshFunction("size_t", bmesh, 2, 0)
    for c in cells(bmesh):
      if Facet(self.mesh, cellmap[c.index()]).normal().z() > 1e-3:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    self.srfmesh = submesh

  def form_bed_mesh(self):
    """
    sets self.bedmesh, the basal boundary mesh for this model instance.
    """
    s = "::: extracting bed mesh :::"
    print_text(s, cls=self)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = MeshFunction("size_t", bmesh, 2, 0)
    for c in cells(bmesh):
      if Facet(self.mesh, cellmap[c.index()]).normal().z() < -1e-3:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    self.bedmesh = submesh

  def form_lat_mesh(self):
    """
    sets self.latmesh, the lateral boundary mesh for this model instance.
    """
    s = "::: extracting lateral mesh :::"
    print_text(s, cls=self)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = CellFunction("size_t", bmesh, 0)
    for c in cells(bmesh):
      if abs(Facet(self.mesh, cellmap[c.index()]).normal().z()) < 1e-3:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    self.latmesh = submesh

  def form_dvd_mesh(self):
    """
    sets self.dvdmesh, the lateral divide boundary mesh for this model instance.
    """
    s = "::: extracting lateral divide mesh :::"
    print_text(s, cls=self)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = CellFunction("size_t", bmesh, 0)
    self.lat_mask.set_allow_extrapolation(True)
    for c in cells(bmesh):
      f       = Facet(self.mesh, cellmap[c.index()])
      n       = f.normal()
      x_m     = f.midpoint().x()
      y_m     = f.midpoint().y()
      z_m     = f.midpoint().z()
      if abs(n.z()) < 1e-3 and self.lat_mask(x_m, y_m, z_m) <= 0:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    self.dvdmesh = submesh
      
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
  
  def solve_hydrostatic_pressure(self, annotate=True, cls=None):
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
    p      = self.vert_integrate(rhoi*g, d='down')
    pv     = p.vector()
    pv[pv < 0] = 0.0
    self.assign_variable(self.p, p)
  
  def vert_extrude(self, u, d='up', Q='self'):
    r"""
    This extrudes a function *u* vertically in the direction *d* = 'up' or
    'down'.  It does this by solving a variational problem:
  
    .. math::
       
       \frac{\partial v}{\partial z} = 0 \hspace{10mm}
       v|_b = u

    """
    s = "::: extruding function %swards :::" % d
    print_text(s, cls=self)
    if type(Q) != FunctionSpace:
      #Q  = self.Q_non_periodic
      Q = u.function_space()
    ff   = self.ff
    phi  = TestFunction(Q)
    v    = TrialFunction(Q)
    a    = v.dx(2) * phi * dx
    L    = DOLFIN_EPS * phi * dx
    bcs  = []
    # extrude bed (ff = 3,5) 
    if d == 'up':
      #if self.N_GAMMA_B_GND != 0:
      #  bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_GND))  # grounded
      #if self.N_GAMMA_B_FLT != 0:
      #  bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_FLT))  # shelves
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_GND))  # grounded
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_FLT))  # shelves
    # extrude surface (ff = 2,6) 
    elif d == 'down':
      #if self.N_GAMMA_S_GND != 0:
      #  bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_GND))  # grounded
      #if self.N_GAMMA_S_FLT != 0:
      #  bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_FLT))  # shelves
      #if self.N_GAMMA_U_GND != 0:
      #  bcs.append(DirichletBC(Q, u, ff, self.GAMMA_U_GND))  # grounded
      #if self.N_GAMMA_U_FLT != 0:
      #  bcs.append(DirichletBC(Q, u, ff, self.GAMMA_U_FLT))  # shelves
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
    print_min_max(v, 'extruded function')
    return v
  
  def vert_integrate(self, u, d='up', Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
    s = "::: vertically integrating function :::"
    print_text(s, cls=self)

    if type(Q) != FunctionSpace:
      Q  = self.Q_non_periodic
      #Q = u.function_space()
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    bcs = []
    # integral is zero on bed (ff = 3,5) 
    if d == 'up':
      #if self.N_GAMMA_B_GND != 0:
      #  bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_GND))  # grounded
      #if self.N_GAMMA_B_FLT != 0:
      #  bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_FLT))  # shelves
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_FLT))  # shelves
      a      = v.dx(2) * phi * dx
    # integral is zero on surface (ff = 2,6) 
    elif d == 'down':
      #if self.N_GAMMA_S_GND != 0:
      #  bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_GND))  # grounded
      #if self.N_GAMMA_S_FLT != 0:
      #  bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_FLT))  # shelves
      #if self.N_GAMMA_U_GND != 0:
      #  bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_GND))  # grounded
      #if self.N_GAMMA_U_FLT != 0:
      #  bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_FLT))  # shelves
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_FLT))  # shelves
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_FLT))  # shelves
      a      = -v.dx(2) * phi * dx
    L      = u * phi * dx
    name   = 'value integrated %s' % d 
    v      = Function(Q, name=name)
    solve(a == L, v, bcs, annotate=False)
    print_min_max(v, 'vertically integrated function')
    return v

  def calc_vert_average(self, u):
    """
    Calculates the vertical average of a given function *u*.  
    
    :param u: Function to avergage vertically
    :rtype:   the vertical average of *u*
    """
    s = "::: calculating vertical average :::"
    print_text(s, cls=self)

    # vertically integrate the function up and then extrude that down :
    ubar = self.vert_integrate(u,  d='up')
    ubar = self.vert_extrude(ubar, d='down')
    
    try:
      name = 'vertical average of %s' % u.name()
    except AttributeError:
      name = 'vertical average'
    ubar.rename(name, '')
   
    # divide by the thickness for vertical average : 
    ubar_v = ubar.vector().array()
    H_v    = self.S.vector().array() - self.B.vector().array() + DOLFIN_EPS
    self.assign_variable(ubar, ubar_v / H_v)
    return ubar
 
  def save_bed_mesh(self, h5File): 
    """
    save the basal boundary mesh to hdf5 file <h5File>.
    """
    s = "::: writing 'bedmesh' to supplied hdf5 file :::"
    print_text(s, cls=self.this)
    h5File.write(self.bedmesh, 'bedmesh')

  def save_srf_mesh(self, h5File): 
    """
    save the surface boundary mesh to hdf5 file <h5File>.
    """
    s = "::: writing 'srfmesh' to supplied hdf5 file :::"
    print_text(s, cls=self.this)
    h5File.write(self.srfmesh, 'srfmesh')

  def save_lat_mesh(self, h5File): 
    """
    save the lateral boundary mesh to hdf5 file <h5File>.
    """
    s = "::: writing 'latmesh' to supplied hdf5 file :::"
    print_text(s, cls=self.this)
    h5File.write(self.latmesh, 'latmesh')

  def save_dvd_mesh(self, h5File): 
    """
    save the divide boundary mesh to hdf5 file <h5File>.
    """
    s = "::: writing 'dvdmesh' to supplied hdf5 file :::"
    print_text(s, cls=self.this)
    h5File.write(self.dvdmesh, 'dvdmesh')

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D3Model, self).initialize_variables()

    s = "::: initializing 3D variables :::"
    print_text(s, cls=self)

    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = abs(min(0, x[2]))
    self.D = Depth(element=self.Q.ufl_element())
   
    self.init_mask(1.0) # default to all grounded ice 
    self.init_E(1.0)    # always use no enhancement on rate-factor A 
    
    # Age model   
    self.age           = Function(self.Q, name='age')
    self.a0            = Function(self.Q, name='a0')

    # Surface climate model
    self.precip        = Function(self.Q, name='precip')


