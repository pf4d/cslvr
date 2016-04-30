from fenics            import *
from dolfin_adjoint    import *
from cslvr.io          import print_text, get_text, print_min_max
from cslvr.model       import Model
from pylab             import inf
import sys

class D3Model(Model):
  """ 
  """

  def __init__(self, mesh, out_dir='./results/', 
               use_periodic=False):
    """
    Create and instance of a 3D model.
    """
    s = "::: INITIALIZING 3D MODEL :::"
    print_text(s, cls=self)
    
    Model.__init__(self, mesh, out_dir, use_periodic)
  
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
      self.num_facets = self.mesh.size_global(2)
      self.num_cells  = self.mesh.size_global(3)
      self.dof        = self.mesh.size_global(0)
    s = "    - %iD mesh set, %i cells, %i facets, %i vertices - " \
        % (self.dim, self.num_cells, self.num_facets, self.dof)
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

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D3Model, self).generate_function_spaces(use_periodic)

    s = "::: generating 3D function spaces :::"
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
    
    s = "    - 3D function spaces created - "
    print_text(s, cls=self)

  def generate_stokes_function_spaces(self, kind='mini'):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.

    If <kind> == 'mini', use enriched mini elements.
    If <kind> == 'th',   use Taylor-Hood elements.
    """
    # mini elements :
    if kind == 'mini':
      s = "::: generating 'mini' Stokes function spaces :::"
        
      self.Bub    = FunctionSpace(self.mesh, "B", 4, 
                                  constrained_domain=self.pBC)
      self.MQ     = self.Q + self.Bub
      M3          = MixedFunctionSpace([self.MQ]*3)
      self.Q4     = MixedFunctionSpace([M3, self.Q])
      self.Q5     = MixedFunctionSpace([M3, self.Q, self.Q])

    # Taylor-Hood elements :
    elif kind == 'th':
      s = "::: generating 'Taylor-Hood' Stokes function spaces :::"
      V           = VectorFunctionSpace(self.mesh, "CG", 2,
                                        constrained_domain=self.pBC)
      self.Q4     = V * self.Q
      self.Q5     = V * self.Q * self.Q
    
    else:
      s = ">>> METHOD generate_stokes_function_spaces <kind> FIELD <<<\n" + \
          ">>> MAY BE 'mini' OR 'th', NOT '%s'. <<<" % kind
      print_text(s, 'red', 1)
      sys.exit(1)

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
    self.ff      = FacetFunction('size_t', self.mesh, 0)
    self.ff_acc  = FacetFunction('size_t', self.mesh, 0)
    self.cf      = CellFunction('size_t',  self.mesh, 0)
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
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 1:
          self.ff[f] = 5
        else:
          self.ff[f] = 3
      
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        # if we want to use a basin, we need to mark the interior facets :
        if mark_divide:
          lat_mask_xy = lat_mask(x_m, y_m, z_m)
          if lat_mask_xy > 0:
            if z_m > 0:
              self.ff[f] = 4
            else:
              self.ff[f] = 10
          else:
            self.ff[f] = 7
        # otherwise just mark for over (4) and under (10) water :
        else:
          if z_m > 0:
            self.ff[f] = 4
          else:
            self.ff[f] = 10
    
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

  def calculate_flat_mesh_boundaries(self, mask=None, adot=None,
                                     mark_divide=False):
    """
    Determines the boundaries of the current model mesh
    """
    s = "::: calculating flat_mesh boundaries :::"
    print_text(s, cls=self)

    self.Q_flat = FunctionSpace(self.flat_mesh, "CG", 1, 
                                constrained_domain=self.pBC)
    
    # this function contains markers which may be applied to facets of the mesh
    self.ff_flat = FacetFunction('size_t', self.flat_mesh, 0)
    
    # default to all grounded ice :
    if mask == None:
      mask = Expression('1.0', element=self.Q.ufl_element())
    
    # default to all positive accumulation :
    if adot == None:
      adot = Expression('1.0', element=self.Q.ufl_element())
    
    tol = 1e-6
    
    s = "    - iterating through %i facets of flat_mesh - " % self.num_facets
    print_text(s, cls=self)
    for f in facets(self.flat_mesh):
      n       = f.normal()
      x_m     = f.midpoint().x()
      y_m     = f.midpoint().y()
      z_m     = f.midpoint().z()
      mask_xy = mask(x_m, y_m, z_m)
    
      if   n.z() >=  tol and f.exterior():
        if mask_xy > 1:
          self.ff_flat[f] = 6
        else:
          self.ff_flat[f] = 2
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 1:
          self.ff_flat[f] = 5
        else:
          self.ff_flat[f] = 3
    
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        if mark_divide:
          if mask_xy > 1:
            self.ff_flat[f] = 4
          else:
            self.ff_flat[f] = 7
        else:
          self.ff_flat[f] = 4
    
    s = "    - done - "
    print_text(s, cls=self)
    
    self.ds_flat = Measure('ds')[self.ff_flat]
  
  def set_subdomains_3(self, ff, cf, ff_acc):
    """
    Set the facet subdomains to FacetFunction <ff>, and set the cell subdomains 
    to CellFunction <cf>, and accumulation FacetFunction to <ff_acc>.
    """
    s = "::: setting 3D subdomains :::"
    print_text(s, cls=self)

    self.ff     = ff
    self.cf     = cf
    self.ff_acc = ff_acc
    self.ds     = Measure('ds')[self.ff]
    self.dx     = Measure('dx')[self.cf]
    
    self.dx_g    = self.dx(0)                # internal above grounded
    self.dx_f    = self.dx(1)                # internal above floating
    self.dBed_g  = self.ds(3)                # grounded bed
    self.dBed_f  = self.ds(5)                # floating bed
    self.dBed    = self.ds(3) + self.ds(5)   # bed
    self.dSrf_g  = self.ds(2)                # surface of grounded ice
    self.dSrf_f  = self.ds(6)                # surface of floating ice
    self.dSrf    = self.ds(6) + self.ds(2)   # surface
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
    s = "::: setting 3D subdomains :::"
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
    
    s = "    - iterating through %i vertices - " % self.dof
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
    pb      = CellFunction("size_t", bmesh, 0)
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
    pb      = CellFunction("size_t", bmesh, 0)
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
    self.assign_variable(self.p, p, cls=cls)
  
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
      Q  = self.Q
    ff   = self.ff
    phi  = TestFunction(Q)
    v    = TrialFunction(Q)
    a    = v.dx(2) * phi * dx
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
      a      = v.dx(2) * phi * dx
    # integral is zero on surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_FLT))  # shelves
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_GND))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_FLT))  # shelves
      a      = -v.dx(2) * phi * dx
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
    s = "::: calculating vertical average :::"
    print_text(s, cls=self)

    ubar = self.vert_integrate(u, d='up')
    ubar = self.vert_extrude(ubar, d='down')
    
    try:
      name = 'vertical average of %s' % u.name()
    except AttributeError:
      name = 'vertical average'
    ubar.rename(name, '')
    
    ubar_v = ubar.vector().array()
    H_v    = self.S.vector().array() - self.B.vector().array() + DOLFIN_EPS
    self.assign_variable(ubar, ubar_v / H_v, cls=self)
    return ubar

  def strain_rate_tensor(self):
    """
    return the strain-rate tensor of <U>.
    """
    U = self.U3
    return 0.5 * (grad(U) + grad(U).T)

  def effective_strain_rate(self):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor()
    ep_xx  = epi[0,0]
    ep_yy  = epi[1,1]
    ep_zz  = epi[2,2]
    ep_xy  = epi[0,1]
    ep_xz  = epi[0,2]
    ep_yz  = epi[1,2]
    
    # Second invariant of the strain rate tensor squared
    epsdot = 0.5 * (+ ep_xx**2 + ep_yy**2 + ep_zz**2) \
                    + ep_xy**2 + ep_xz**2 + ep_yz**2
    return epsdot
    
  def stress_tensor(self):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the Cauchy stress tensor :::"
    print_text(s, self.color)
    epi = self.strain_rate_tensor(self.U3)
    I   = Identity(3)

    sigma = 2*self.eta*epi - self.p*I
    return sigma
    
  def deviatoric_stress_tensor(self):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
    print_text(s, self.color)
    epi = self.strain_rate_tensor()
    tau = 2*self.eta*epi
    return tau
  
  def effective_stress(self):
    """
    return the effective stress squared.
    """
    tau    = self.deviatoric_stress_tensor()
    tu_xx  = tau[0,0]
    tu_yy  = tau[1,1]
    tu_zz  = tau[2,2]
    tu_xy  = tau[0,1]
    tu_xz  = tau[0,2]
    tu_yz  = tau[1,2]
    
    # Second invariant of the strain rate tensor squared
    taudot = 0.5 * (+ tu_xx**2 + tu_yy**2 + tu_zz**2) \
                    + tu_xy**2 + tu_xz**2 + tu_yz**2
    return taudot

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
    super(D3Model, self).initialize_variables()

    s = "::: initializing 3D variables :::"
    print_text(s, cls=self)

    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = abs(min(0, x[2]))
    self.D = Depth(element=self.Q.ufl_element())
    
    # Enthalpy model
    self.theta_surface = Function(self.Q, name='theta_surface')
    self.theta_float   = Function(self.Q, name='theta_float')
    self.theta_app     = Function(self.Q, name='theta_app')
    self.theta0        = Function(self.Q, name='theta0')
    self.W0            = Function(self.Q, name='W0')
    self.thetahat      = Function(self.Q, name='thetahat')
    self.uhat          = Function(self.Q, name='uhat')
    self.vhat          = Function(self.Q, name='vhat')
    self.what          = Function(self.Q, name='what')
    self.mhat          = Function(self.Q, name='mhat')

    # Age model   
    self.age           = Function(self.Q, name='age')
    self.a0            = Function(self.Q, name='a0')

    # Surface climate model
    self.precip        = Function(self.Q, name='precip')

    # Stokes-balance model :
    self.u_s           = Function(self.Q, name='u_s')
    self.u_t           = Function(self.Q, name='u_t')
    self.F_id          = Function(self.Q, name='F_id')
    self.F_jd          = Function(self.Q, name='F_jd')
    self.F_ib          = Function(self.Q, name='F_ib')
    self.F_jb          = Function(self.Q, name='F_jb')
    self.F_ip          = Function(self.Q, name='F_ip')
    self.F_jp          = Function(self.Q, name='F_jp')
    self.F_ii          = Function(self.Q, name='F_ii')
    self.F_ij          = Function(self.Q, name='F_ij')
    self.F_iz          = Function(self.Q, name='F_iz')
    self.F_ji          = Function(self.Q, name='F_ji')
    self.F_jj          = Function(self.Q, name='F_jj')
    self.F_jz          = Function(self.Q, name='F_jz')
    self.tau_iz        = Function(self.Q, name='tau_iz')
    self.tau_jz        = Function(self.Q, name='tau_jz')



