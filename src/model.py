from fenics       import *
from ufl.indexed  import Indexed
from abc          import ABCMeta, abstractmethod
from physics      import Physics
from solvers      import Solver
from helper       import default_config
from io           import print_text, print_min_max
import numpy              as np
import physical_constants as pc
import sys

class Model(object):
  """ 
  Instance of a 2D flowline ice model that contains geometric and scalar 
  parameters and supporting functions.  This class does not contain actual 
  physics but rather the interface to use physics in different simulation 
  types.
  """

  def __init__(self, config=None):
    """
    Create and instance of the model.
    """
    PETScOptions.set("mat_mumps_icntl_14", 100.0)
    if config == None:
      self.config = default_config()
    else:
      self.config = config
    self.MPI_rank = MPI.rank(mpi_comm_world())
    self.color    = '148'#'purple_1a'

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    s = "    - using periodic boundaries -"
    print_text(s, self.color)

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
    s = "::: setting mesh :::"
    print_text(s, self.color)
    self.mesh       = mesh
    self.flat_mesh  = Mesh(mesh)
    self.dim        = self.mesh.ufl_cell().topological_dimension()
    self.mesh.init(1,2)
    if self.dim == 3:
      self.num_facets = self.mesh.size_global(2)
      self.num_cells  = self.mesh.size_global(3)
      self.dof        = self.mesh.size_global(0)
    elif self.dim == 2:
      self.num_facets = self.mesh.size_global(1)
      self.num_cells  = self.mesh.size_global(2)
      self.dof        = self.mesh.size_global(0)
    s = "    - %iD mesh set, %i cells, %i facets, %i vertices - " \
        % (self.dim, self.num_cells, self.num_facets, self.dof)
    print_text(s, self.color)
    self.generate_function_spaces()

  def generate_function_spaces(self):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    s = "::: generating %s function spaces :::" % self.config['model_order']
    print_text(s, self.color)
    if self.config['periodic_boundary_conditions']:
      self.generate_pbc()
    else:
      self.pBC = None
    self.Q      = FunctionSpace(self.mesh,      "CG", 1, 
                                constrained_domain=self.pBC)
    if     self.config['model_order'] != 'L1L2' \
       and self.config['model_order'] != 'SSA':
      self.Q_flat = FunctionSpace(self.flat_mesh, "CG", 1, 
                                  constrained_domain=self.pBC)
      if self.config['model_order'] == 'BP':
        if self.config['velocity']['full_BP']:
          self.Q3     = MixedFunctionSpace([self.Q]*3)
        else:
          self.Q2     = MixedFunctionSpace([self.Q]*2)
      elif self.config['model_order'] == 'stokes':
        self.Q4     = MixedFunctionSpace([self.Q]*4)
        # mini elements :
        self.Bub    = FunctionSpace(self.mesh, "B", 4, 
                                    constrained_domain=self.pBC)
        self.MQ     = self.Q + self.Bub
        M3          = MixedFunctionSpace([self.MQ]*3)
        self.MV     = MixedFunctionSpace([M3,self.Q])
        # Taylor-Hood elements :
        #V           = VectorFunctionSpace(self.mesh, "CG", 2,
        #                                  constrained_domain=self.pBC)
        #self.MV     = V * self.Q
      self.V      = VectorFunctionSpace(self.mesh, "CG", 1)
    else:
      poly_degree = self.config['velocity']['poly_degree']
      N_T         = self.config['enthalpy']['N_T']
      self.HV     = MixedFunctionSpace([self.Q]*2*poly_degree) # VELOCITY
      self.Z      = MixedFunctionSpace([self.Q]*N_T)           # TEMPERATURE
      self.Q2     = MixedFunctionSpace([self.Q]*2)
    
    self.Q_non_periodic = FunctionSpace(self.mesh, "CG", 1)

    s = "    - function spaces created - "
    print_text(s, self.color)
    
  def set_surface_and_bed(self, S, B):
    """
    Set the Functions for the surface <S> and bed <B>.
    """
    s = "::: setting the surface and bed functions :::"
    print_text(s, self.color)
    self.S = S
    self.B = B
    print_min_max(S, 'S')
    print_min_max(B, 'B')
  
  def set_geometry(self, surface, bed, deform=True):
    """
    Sets the geometry of the surface and bed of the ice sheet.
    
    :param surface : Expression representing the surface of the mesh
    :param bed     : Expression representing the base of the mesh
    :param mask    : Expression representing a mask of grounded (0) and 
                     floating (1) areas of the ice.
    """
    s = "::: setting geometry :::"
    print_text(s, self.color)

    self.S_ex = surface
    self.B_ex = bed
    
    if deform:
      self.deform_mesh_to_geometry()
    
    self.S = interpolate(self.S_ex, self.Q_non_periodic)
    self.B = interpolate(self.B_ex, self.Q_non_periodic)
    print_min_max(self.S, 'S')
    print_min_max(self.B, 'B')

  def deform_mesh_to_geometry(self):
    """
    Deforms the mesh to the geometry.
    """
    s = "::: deforming mesh to geometry :::"
    print_text(s, self.color)
    
    # transform z :
    # thickness = surface - base, z = thickness + base
    # Get the height of the mesh, assumes that the base is at z=0
    max_height  = self.mesh.coordinates()[:,2].max()
    min_height  = self.mesh.coordinates()[:,2].min()
    mesh_height = max_height - min_height
    
    s = "    - iterating through %i vertices - " % self.dof
    print_text(s, self.color)
    
    for x in self.mesh.coordinates():
      x[2] = (x[2] / mesh_height) * ( + self.S_ex(x[0],x[1],x[2]) \
                                      - self.B_ex(x[0],x[1],x[2]) )
      x[2] = x[2] + self.B_ex(x[0], x[1], x[2])

  def set_subdomains(self, ff, cf, ff_acc):
    """
    Set the facet subdomains to FacetFunction <ff>, and set the cell subdomains 
    to CellFunction <cf>, and accumulation FacetFunction to <ff_acc>.
    """
    s = "::: setting subdomains :::"
    print_text(s, self.color)
    self.ff     = ff
    self.cf     = cf
    self.ff_acc = ff_acc
    self.ds     = Measure('ds')[self.ff]
    self.dx     = Measure('dx')[self.cf]

  def calculate_boundaries(self, mask=None, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    # default to all grounded ice :
    if mask == None:
      mask = Expression('0.0', element=self.Q.ufl_element())
    
    # default to all positive accumulation :
    if adot == None:
      adot = Expression('1.0', element=self.Q.ufl_element())
    
    s = "::: calculating boundaries :::"
    print_text(s, self.color)
    
    # this function contains markers which may be applied to facets of the mesh
    self.ff      = FacetFunction('size_t', self.mesh,      0)
    self.ff_acc  = FacetFunction('size_t', self.mesh,      0)
    self.ff_flat = FacetFunction('size_t', self.flat_mesh, 0)
    
    self.cf      = CellFunction('size_t',  self.mesh,      0)
    dofmap       = self.Q.dofmap()
    shf_dofs     = []
    gnd_dofs     = []
    
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
    print_text(s, self.color)
    for f in facets(self.mesh):
      n       = f.normal()
      x_m     = f.midpoint().x()
      y_m     = f.midpoint().y()
      z_m     = f.midpoint().z()
      mask_xy = mask(x_m, y_m, z_m)
      adot_xy = adot(x_m, y_m, z_m)
      
      if   n.z() >=  tol and f.exterior():
        if adot_xy > 0:
          self.ff_acc[f] = 1
        if mask_xy > 0:
          self.ff[f] = 6
        else:
          self.ff[f] = 2
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 0:
          self.ff[f] = 5
        else:
          self.ff[f] = 3
    
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        #if mask_xy > 0:
        #  self.ff[f] = 4
        #else:
        #  self.ff[f] = 7
        self.ff[f] = 4
    
    for f in facets(self.flat_mesh):
      n       = f.normal()
      x_m     = f.midpoint().x()
      y_m     = f.midpoint().y()
      z_m     = f.midpoint().z()
      mask_xy = mask(x_m, y_m, z_m)
    
      if   n.z() >=  tol and f.exterior():
        if mask_xy > 0:
          self.ff_flat[f] = 6
        else:
          self.ff_flat[f] = 2
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 0:
          self.ff_flat[f] = 5
        else:
          self.ff_flat[f] = 3
    
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        #if mask_xy > 0:
        #  self.ff_flat[f] = 4
        #else:
        #  self.ff_flat[f] = 7
        self.ff_flat[f] = 4
    
    s = "    - iterating through %i cells - " % self.num_cells
    print_text(s, self.color)
    for c in cells(self.mesh):
      x_m     = c.midpoint().x()
      y_m     = c.midpoint().y()
      z_m     = c.midpoint().z()
      mask_xy = mask(x_m, y_m, z_m)

      if mask_xy > 0:
        self.cf[c] = 1
        shf_dofs.extend(dofmap.cell_dofs(c.index()))
      else:
        self.cf[c] = 0
        gnd_dofs.extend(dofmap.cell_dofs(c.index()))

    self.shf_dofs = list(set(shf_dofs))
    self.gnd_dofs = list(set(gnd_dofs))

    self.ds      = Measure('ds')[self.ff]
    self.ds_flat = Measure('ds')[self.ff_flat]
    self.dx      = Measure('dx')[self.cf]

  def init_U(self, u, v, w):
    """
    """
    s = "::: initializing velocity :::"
    print_text(s, self.color)
    self.assign_variable(self.u, u)
    self.assign_variable(self.v, v)
    self.assign_variable(self.w, w)
    print_min_max(self.u, 'u')
    print_min_max(self.v, 'v')
    print_min_max(self.w, 'w')
  
  def init_P(self, P):
    """
    """
    s = "::: initializing pressure :::"
    print_text(s, self.color)
    self.assign_variable(self.P, P)
    print_min_max(self.P, 'P')
  
  def init_T(self, T):
    """
    """
    s = "::: initializing temperature :::"
    print_text(s, self.color)
    self.assign_variable(self.T, T)
    print_min_max(self.T, 'T')
  
  def init_W(self, W):
    """
    """
    s = "::: initializing water content :::"
    print_text(s, self.color)
    self.assign_variable(self.W, W)
    print_min_max(self.W, 'W')
  
  def init_Mb(self, Mb):
    """
    """
    s = "::: initializing basal melt rate :::"
    print_text(s, self.color)
    self.assign_variable(self.Mb, Mb)
    print_min_max(self.Mb, 'Mb')
  
  def init_adot(self, adot):
    """
    """
    s = "::: initializing accumulation :::"
    print_text(s, self.color)
    self.assign_variable(self.adot, adot)
    print_min_max(self.adot, 'adot')
    s = "::: removing junk accumulation :::"
    print_text(s, self.color)
    adot_v = self.adot.vector().array()
    adot_v[adot_v < -100] = 0
    self.assign_variable(self.adot, adot_v)
    print_min_max(self.adot, 'adot')
  
  def init_beta(self, beta):
    """
    """
    s = "::: initializing basal friction coefficient :::"
    print_text(s, self.color)
    self.assign_variable(self.beta, beta)
    print_min_max(self.beta, 'beta')
  
  def init_b(self, b):
    """
    """
    s = "::: initializing rate factor over grounded and shelves :::"
    print_text(s, self.color)
    self.init_b_shf(b)
    self.init_b_gnd(b)
  
  def init_b_shf(self, b_shf):
    """
    """
    s = "::: initializing rate factor over shelves :::"
    print_text(s, self.color)
    if type(self.b_shf) != Function:
      self.b_shf = Function(self.Q)
    self.assign_variable(self.b_shf, b_shf)
    print_min_max(self.b_shf, 'b_shf')
  
  def init_b_gnd(self, b_gnd):
    """
    """
    s = "::: initializing rate factor over grounded ice :::"
    print_text(s, self.color)
    if type(self.b_gnd) != Function:
      self.b_gnd = Function(self.Q)
    self.assign_variable(self.b_gnd, b_gnd)
    print_min_max(self.b_gnd, 'b_gnd')
  
  def init_E(self, E):
    """
    """
    s = "::: initializing enhancement factor over grounded and shelves :::"
    print_text(s, self.color)
    self.init_E_shf(E)
    self.init_E_gnd(E)
  
  def init_E_shf(self, E_shf):
    """
    """
    s = "::: initializing enhancement factor over shelves :::"
    print_text(s, self.color)
    self.assign_variable(self.E_shf, E_shf)
    print_min_max(self.E_shf, 'E_shf')
  
  def init_E_gnd(self, E_gnd):
    """
    """
    s = "::: initializing enhancement factor over grounded ice :::"
    print_text(s, self.color)
    self.assign_variable(self.E_gnd, E_gnd)
    print_min_max(self.E_gnd, 'E_gnd')
  
  def init_eta(self, eta):
    """
    """
    s = "::: initializing viscosity :::"
    print_text(s, self.color)
    self.assign_variable(self.eta, eta)
    print_min_max(self.eta, 'eta')
  
  def init_etabar(self, etabar):
    """
    """
    s = "::: initializing vertically averaged viscosity :::"
    print_text(s, self.color)
    self.assign_variable(self.etabar, etabar)
    print_min_max(self.etabar, 'etabar')
  
  def init_component_Ubar(self, ubar, vbar):
    """
    """
    s = "::: initializing vertically averaged horizontal velocity :::"
    print_text(s, self.color)
    self.assign_variable(self.ubar, ubar)
    self.assign_variable(self.vbar, vbar)
    print_min_max(self.ubar, 'ubar')
    print_min_max(self.vbar, 'vbar')
  
  def init_T_surface(self, T_s):
    """
    """
    s = "::: initializing surface temperature :::"
    print_text(s, self.color)
    self.assign_variable(self.T_surface, T_s)
    print_min_max(self.T_surface, 'T_surface')
  
  def init_q_geo(self, q_geo):
    """
    """
    s = "::: initializing geothermal heat flux :::"
    print_text(s, self.color)
    self.assign_variable(self.q_geo, q_geo)
    print_min_max(self.q_geo, 'q_geo')
  
  def init_U_ob(self, u_ob, v_ob):
    """
    """
    s = "::: initializing surface velocity :::"
    print_text(s, self.color)
    self.assign_variable(self.u_ob, u_ob)
    self.assign_variable(self.v_ob, v_ob)
    u_v      = self.u_ob.vector().array()
    v_v      = self.v_ob.vector().array()
    U_mag_v  = np.sqrt(u_v**2 + v_v**2 + 1e-16)
    self.assign_variable(self.U_ob, U_mag_v)
    print_min_max(self.u_ob, 'u_ob')
    print_min_max(self.v_ob, 'v_ob')
    print_min_max(self.U_ob, 'U_ob')
    
  def init_H(self, H):
    """
    """
    s = "::: initializing thickness :::"
    print_text(s, self.color)
    self.assign_variable(self.H,  H)
    self.assign_variable(self.H0, H)
    print_min_max(self.H, 'H')

  def init_H_bounds(self, H_min, H_max):
    """
    """
    s = "::: initializing bounds on thickness :::"
    print_text(s, self.color)
    self.assign_variable(self.H_min, H_min)
    self.assign_variable(self.H_max, H_max)
    print_min_max(self.H_min, 'H_min')
    print_min_max(self.H_max, 'H_max')
  
  def init_Ubar(self, Ubar):
    """
    """
    s = "::: initializing balance velocity :::"
    print_text(s, self.color)
    self.assign_variable(self.Ubar, Ubar)
    print_min_max(self.Ubar, 'Ubar')
  
  def init_u_lat(self, u_lat):
    """
    """
    s = "::: initializing u lateral boundary condition :::"
    print_text(s, self.color)
    self.assign_variable(self.u_lat, u_lat)
    print_min_max(self.u_lat, 'u_lat')
  
  def init_v_lat(self, v_lat):
    """
    """
    s = "::: initializing v lateral boundary condition :::"
    print_text(s, self.color)
    self.assign_variable(self.v_lat, v_lat)
    print_min_max(self.v_lat, 'v_lat')
  
  def init_w_lat(self, w_lat):
    """
    """
    s = "::: initializing w lateral boundary condition :::"
    print_text(s, self.color)
    self.assign_variable(self.w_lat, w_lat)
    print_min_max(self.w_lat, 'w_lat')
  
  def init_mask(self, mask):
    """
    """
    s = "::: initializing shelf mask :::"
    print_text(s, self.color)
    self.assign_variable(self.mask, mask)
    print_min_max(self.mask, 'mask')
    self.shf_dofs = np.where(self.mask.vector().array() >  0.0)[0]
    self.gnd_dofs = np.where(self.mask.vector().array() == 0.0)[0]

  def set_parameters(self, params):
    """
    Sets the model's dictionary of parameters
    
    :param params: :class:`~src.physical_constants.IceParameters` object 
       containing model-relavent parameters
    """
    self.params = params
  
  def get_surface_mesh(self):
    """
    Returns the surface of the mesh for this model instance.
    """
    s = "::: extracting bed mesh :::"
    print_text(s, self.color)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = CellFunction("size_t", bmesh, 0)
    for c in cells(bmesh):
      if Facet(self.mesh, cellmap[c.index()]).normal().z() > 1e-3:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    return submesh

  def get_bed_mesh(self):
    """
    Returns the bed of the mesh for this model instance.
    """
    s = "::: extracting bed mesh :::"
    print_text(s, self.color)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = CellFunction("size_t", bmesh, 0)
    for c in cells(bmesh):
      if Facet(self.mesh, cellmap[c.index()]).normal().z() < -1e-3:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    return submesh
      
  def init_beta_SIA(self, U_mag=None, eps=0.5):
    r"""
    Init beta  :`\tau_b = \tau_d`, the shallow ice approximation, 
    using the observed surface velocity <U_mag> as approximate basal 
    velocity and <gradS> the projected surface gradient. i.e.,

    .. math::
    \beta^2 \Vert U_b \Vert H^r = \rho g H \Vert \nabla S \Vert
    
    """
    s = "::: initializing beta from SIA :::"
    print_text(s, self.color)
    r        = 0.0
    Q        = self.Q
    rhoi     = self.rhoi
    g        = self.g
    gradS    = self.gradS
    H        = self.S - self.B
    U_s      = Function(Q)
    if U_mag == None:
      U_v = self.U_ob.vector().array()
    else:
      U_v = U_mag.vector().array()
    U_v[U_v < eps] = eps
    self.assign_variable(U_s, U_v)
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta_0   = project(sqrt((rhoi*g*H*S_mag) / (H**r * U_s)), Q)
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < 1e-3] = 1e-3
    self.betaSIA = Function(Q)
    self.assign_variable(self.betaSIA, beta_0_v)
    print_min_max(self.betaSIA, 'betaSIA')
    
    self.assign_variable(self.beta, DOLFIN_EPS)
    bc_beta = DirichletBC(self.Q, self.betaSIA, self.ff, 3)
    bc_beta.apply(self.beta.vector())
    print_min_max(self.beta, 'beta')
      
  def init_beta_SIA_new_slide(self, U_mag=None, eps=0.5):
    r"""
    Init beta  :`\tau_b = \tau_d`, the shallow ice approximation, 
    using the observed surface velocity <U_mag> as approximate basal 
    velocity and <gradS> the projected surface gradient. i.e.,

    .. math::
    \beta^2 \Vert U_b \Vert H^r = \rho g H \Vert \nabla S \Vert
    
    """
    s = "::: initializing new sliding beta from SIA :::"
    print_text(s, self.color)
    r        = 0.0
    Q        = self.Q
    rhoi     = self.rhoi
    rhow     = self.rhow
    g        = self.g
    gradS    = self.gradS
    H        = self.S - self.B
    D        = self.D
    p        = -0.383
    q        = -0.349
    
    U_s      = Function(Q)
    if U_mag == None:
      U_v = self.U_ob.vector().array()
    else:
      U_v = U_mag.vector().array()
    U_v[U_v < eps] = eps
    self.assign_variable(U_s, U_v)
    
    Ne       = H + rhow/rhoi * D
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta     = U_s**(1/p) / ( rhoi * g * H * S_mag * Ne**(q/p) )
    beta_0   = project(beta, Q)
    
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    #self.assign_variable(beta_0, beta_0_v)
    print_min_max(beta_0, 'beta_0')

    #self.assign_variable(self.beta, beta_0)
    
    self.assign_variable(self.beta, DOLFIN_EPS)
    bc_beta = DirichletBC(self.Q, beta_0, self.ff, 3)
    bc_beta.apply(self.beta.vector())
    
    #self.betaSIA = Function(Q)
    #self.assign_variable(self.betaSIA, beta_0_v)
    
  def init_beta_stats(self, mdl='U', use_temp=False):
    """
    """
    s    = "::: initializing beta from stats :::"
    print_text(s, self.color)
    
    config = self.config
    q_geo  = self.q_geo
    T_s    = self.T_surface
    adot   = self.adot
    Mb     = self.Mb
    Ubar   = self.Ubar
    Q      = self.Q
    B      = self.B
    S      = self.S
    T      = self.T
    T_s    = self.T_surface
    rho    = self.rhoi
    g      = self.g
    H      = S - B

    if mdl == 'Ubar' or mdl == 'U_Ubar':
      #config['balance_velocity']['on']    = True
      config['balance_velocity']['kappa'] = 5.0
   
    elif mdl == 'stress':
      config['stokes_balance']['on']      = True

    Ubar_v = Ubar.vector().array()
    Ubar_v[Ubar_v < 1e-10] = 1e-10
    self.assign_variable(Ubar, Ubar_v)
           
    D      = Function(Q)
    B_v    = B.vector().array()
    D_v    = D.vector().array()
    D_v[B_v < 0] = B_v[B_v < 0]
    self.assign_variable(D, D_v)

    gradS = as_vector([S.dx(0), S.dx(1), 0.0])
    gradB = as_vector([B.dx(0), B.dx(1), 0.0])
    gradH = as_vector([H.dx(0), H.dx(1), 0.0])

    nS   = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    nB   = sqrt(inner(gradB, gradB) + DOLFIN_EPS)
    nH   = sqrt(inner(gradH, gradH) + DOLFIN_EPS)
    
    #if mdl == 'Ubar':
    #  u_x    = -rho * g * H * S.dx(0)
    #  v_x    = -rho * g * H * S.dx(1)
    #  U_i    = as_vector([u_x,  v_x, 0.0])
    #  U_j    = as_vector([v_x, -u_x, 0.0])
    #elif mdl == 'U' or mdl == 'U_Ubar':
    #  U_i    = as_vector([self.u,  self.v, 0.0])
    #  U_j    = as_vector([self.v, -self.u, 0.0])
    U_i    = as_vector([self.u,  self.v, 0.0])
    U_j    = as_vector([self.v, -self.u, 0.0])
    Umag   = sqrt(inner(U_i,U_i) + DOLFIN_EPS)
    Uhat_i = U_i / Umag
    Uhat_j = U_j / Umag

    dBdi = dot(gradB, Uhat_i)
    dBdj = dot(gradB, Uhat_j)
    dSdi = dot(gradS, Uhat_i)
    dSdj = dot(gradS, Uhat_j)
    dHdi = dot(gradH, Uhat_i)
    dHdj = dot(gradH, Uhat_j)

    ini  = sqrt(rho * g * H * nS / (Umag + 0.1))

    x0   = S
    x1   = T_s
    x2   = nS
    x3   = D
    x4   = nB
    x5   = H
    x6   = q_geo
    x7   = adot
    x8   = T
    x9   = Mb
    x10  = self.u
    x11  = self.v
    x12  = self.w
    x13  = ln(Ubar + DOLFIN_EPS)
    x14  = ln(Umag + DOLFIN_EPS)
    x15  = ini
    x16  = dBdi
    x17  = dBdj
    x18  = dSdi
    x19  = dSdj
    x20  = nH
    x21  = self.tau_id
    x22  = self.tau_jd
    x23  = self.tau_ii
    x24  = self.tau_ij
    x25  = self.tau_ji
    x26  = self.tau_jj

    names = ['S', 'T_s', 'gradS', 'D', 'gradB', 'H', 'q_geo', 'adot', 'T',
             'Mb', 'u', 'v', 'w', 'ln(Ubar)', 'ln(Umag)', 'ini',
             'dBdi', 'dBdj', 'dSdi', 'dSdj', 'nablaH', 'tau_id', 'tau_jd',
             'tau_ii', 'tau_ij', 'tau_ji', 'tau_jj']
    names = np.array(names)

    if mdl == 'Ubar':
      if not use_temp:
        X    = [x0,x1,x5,x7,x13,x16,x18]
        idx  = [ 0, 1, 5, 7, 13, 16, 18]
        bhat = [ -1.01661102e+02,   6.59472291e-03,   8.34479667e-01,
                 -3.20751595e-04,  -1.86910058e+00,  -1.50122785e-01,
                 -1.61283407e+01,   3.42099244e+01,  -1.38190017e-07,
                 -2.42124307e-05,   5.28420031e-08,  -5.71485389e-05,
                 -3.75168897e-06,   6.62615357e-04,  -2.09616017e-03,
                 -1.63919106e-03,  -4.67468432e-07,   7.70150910e-03,
                 -1.06827565e-05,   5.82852747e-02,  -1.59176855e-01,
                  2.60703978e-08,   1.12176250e-04,  -9.96266233e-07,
                  1.54898171e-04,  -7.75201260e-03,  -3.97881378e-02,
                 -9.66212690e-04,  -6.88656946e-01,   2.86508703e+00,
                 -4.77406074e-03,   4.46234782e-03,  -9.93937326e-02,
                 -1.11058398e+01,   1.19703551e+01,  -3.46378138e+01]
        #bhat = [ -1.06707322e+02,   6.93681939e-03,   8.72090381e-01,
        #         -2.05377136e-04,  -1.68695225e+00,  -1.54427603e-01,
        #         -1.48494954e+01,   3.13320531e+01,  -1.46372911e-07,
        #         -2.54809386e-05,   5.58213888e-08,  -5.05686875e-05,
        #         -3.57485925e-06,   6.74423417e-04,  -1.90332998e-03,
        #         -1.70912922e-03,  -9.14015814e-07,   6.90894685e-03,
        #          5.38728829e-06,   5.52828014e-02,  -1.49677701e-01,
        #          2.10321794e-08,   1.26574205e-04,  -1.58804814e-06,
        #         -1.07066137e-04,  -6.59781673e-03,  -4.21221477e-02,
        #         -9.11842753e-04,  -5.91089434e-01,   2.37465616e+00,
        #         -4.79794725e-03,  -1.20787950e-03,  -8.37001425e-02,
        #         -1.35364012e+01,   2.01047113e+01,  -3.48057200e+01]
     
      else: 
        X    = [x0,x1,x5,x7,x8,x9,x13,x16,x18]
        idx  = [ 0, 1, 5, 7, 8, 9, 13, 16, 18]
        bhat = [  1.99093750e+01,  -9.37152784e-04,  -1.53849816e-03,
                 -2.72682710e-03,   3.11376629e+00,  -6.22550705e-02,
                 -4.78841821e+02,   1.18870083e-01,   1.46462501e+01,
                  4.73228083e+00,  -1.23039512e-05,   4.80948459e-08,
                 -1.75152253e-04,   1.57869882e-05,  -1.85979092e-03,
                 -5.31979350e-06,  -2.94994855e-04,  -2.88696470e-03,
                  9.87920894e-06,  -1.67014309e-02,   1.38310308e-05,
                  1.29911016e+00,   8.79462642e-06,   2.58486129e-02,
                  4.59079956e-01,  -1.62460133e-04,   8.39672735e-07,
                 -1.44977594e-02,   5.58957555e-07,   7.38625502e-04,
                 -9.92789432e-03,   6.02766800e-03,   2.74638935e-01,
                 -7.24036641e-05,  -4.63126335e-01,   2.92369712e+00,
                  5.07887934e-01,  -4.57929508e-04,  -8.33728342e-02,
                 -4.71625234e-01,  -5.85160316e-02,  -1.74723504e+01,
                 -1.83509536e+01,   5.35514345e-04,  -8.46507380e-02,
                 -1.60127263e+01]
    
    elif mdl == 'U':
      if not use_temp:
        X    = [x0,x1,x5,x7,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 14, 16, 18]
        bhat = [ -9.28289389e+01,   5.73687339e-03,   7.33526290e-01,
                  2.76998568e-03,  -1.08656857e-01,  -1.08545047e+00,
                 -1.50267782e+01,  -7.04864127e+01,  -7.76085391e-08,
                 -2.17802438e-05,  -4.99587467e-08,   5.87139196e-05,
                  1.64670170e-05,   1.06212966e-04,   7.11755177e-05,
                 -1.37677776e-03,  -9.08932836e-06,   3.60621065e-04,
                  2.97118032e-03,   5.50814766e-02,   2.21044611e-01,
                 -1.15497725e-07,   8.63993130e-05,  -2.12395318e-06,
                  7.21699958e-04,  -1.09346933e-02,  -3.12224072e-02,
                 -2.39690796e-02,  -2.95080157e-01,  -3.40502802e-01,
                 -2.62000881e-02,  -1.78157283e-02,   7.19763432e-02,
                 -1.94919730e+00,  -9.82413027e+00,  -7.61245200e+01]
      else:
        X    = [x0,x1,x5,x7,x8,x9,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 8, 9, 14, 16, 18]
        bhat = [  2.09623581e+01,   6.66919839e-04,  -7.02196170e-02,
                 -1.15080308e-03,   5.34783070e+00,  -7.11388758e-02,
                 -4.07361631e+01,   1.02018632e+00,  -1.86900651e+01,
                 -4.20181324e+01,  -9.26143019e-06,  -7.72058925e-08,
                 -4.15062408e-05,   7.02170069e-06,   2.70372865e-03,
                 -1.37333418e-05,   8.87920333e-05,   1.42938174e-03,
                  7.77557165e-06,  -2.35402146e-02,   3.04680358e-04,
                 -1.71597355e-01,   1.40252311e-04,   4.10097716e-02,
                  2.55567246e-01,  -1.33628767e-07,  -2.15459028e-06,
                  6.29599393e-05,  -4.11071912e-05,   1.28619782e-03,
                 -1.46657539e-02,   3.09279801e-03,  -2.27450062e-01,
                 -7.40025166e-03,  -5.06709113e-01,  -6.76120111e-01,
                  3.10802402e-01,  -5.34552872e-03,   2.19914707e-02,
                 -1.40943367e-01,   3.07890125e-01,  -9.03508676e+00,
                  8.27529346e+01,   6.60448755e-03,   2.42989633e+00,
                 -4.31461210e+01]
    
    elif mdl == 'U_Ubar':
      if not use_temp:
        X    = [x0,x1,x5,x7,x13,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 13, 14, 16, 18]
        bhat = [ -9.25221622e+01,   5.70295987e-03,   7.30768422e-01,
                  2.75877006e-03,   7.37861453e-02,  -2.93985236e-03,
                 -1.07390793e+00,  -1.45320123e+01,  -7.18521246e+01,
                 -7.86411913e-08,  -2.15769127e-05,  -4.80926515e-08,
                  5.56842889e-05,   1.28402687e-06,   1.12826733e-05,
                  9.07581727e-05,  -7.62357377e-05,  -1.37165484e-03,
                 -8.99331396e-06,  -3.36292037e-04,   4.24771193e-05,
                  2.97610385e-03,   5.34869351e-02,   2.28993842e-01,
                 -1.17987943e-07,   8.26468590e-05,   2.32815553e-06,
                 -6.66323072e-06,   6.73934903e-04,  -1.12192482e-02,
                 -3.22339742e-02,  -3.78492901e-04,  -2.38023512e-02,
                 -2.88687981e-01,  -4.11715791e-01,   3.06665249e-04,
                  3.29695662e-04,   4.96515338e-03,   1.28914720e-02,
                 -2.83133687e-02,  -3.08127082e-02,  -3.19074160e-02,
                 -1.60977763e+00,  -1.10451113e+01,  -7.66011531e+01]
      else:
        X    = [x0,x1,x5,x7,x8,x9,x13,x14,x16,x18]
        idx  = [ 0, 1, 5, 7, 8, 9, 13, 14, 16, 18]
        bhat = [  1.95228446e+01,   6.59477606e-04,  -6.45139002e-02,
                 -1.10071394e-03,   5.13699019e+00,  -6.45652015e-02,
                 -5.14739582e+01,  -3.68769001e-03,   9.57519905e-01,
                 -1.77507405e+01,  -4.37983921e+01,  -9.02491948e-06,
                 -7.61384926e-08,  -3.73066416e-05,   6.79516468e-06,
                  2.83564402e-03,  -4.68103812e-07,  -1.20747491e-05,
                  4.00845895e-05,   1.67755582e-03,   7.73371401e-06,
                 -2.23470170e-02,   2.78775317e-04,  -1.61211932e-01,
                  4.64633086e-05,   4.37335336e-04,   4.27466758e-02,
                  2.50573113e-01,  -4.81341231e-06,  -2.31708961e-06,
                 -1.68503900e-04,   3.54318161e-06,  -4.20165147e-05,
                  1.26878513e-03,  -1.54490818e-02,   2.66749014e-03,
                 -2.98194766e-01,  -2.92113296e-04,  -4.31378498e-03,
                 -4.83721711e-01,  -7.30055588e-01,   3.42250813e-01,
                 -3.22616161e-05,  -5.40195432e-03,   1.73408633e-02,
                 -1.31066469e-01,   9.73640123e-03,   2.61368301e-01,
                 -9.93273895e+00,   8.31773699e+01,  -5.74031885e-04,
                  9.54289863e-03,  -3.57353698e-02,   3.62295735e-03,
                  2.54399352e+00,  -4.21129483e+01]
    
    elif mdl == 'stress':
      X    = [x0,x1,x5,x7,x14,x16,x18,x21,x22,x24,x25,x26]
      idx  = [ 0, 1, 5, 7, 14, 16, 18, 21, 23, 24, 25, 26]
      bhat = [  5.47574225e+00,   9.14001489e-04,  -1.03229081e-03,
               -7.04987042e-04,   2.15686223e+00,  -1.52869679e+00,
               -1.74593819e+01,  -2.05459701e+01,  -1.23768850e-05,
                2.01460255e-05,   1.97622781e-05,   3.68067438e-05,
                6.63468606e-06,  -3.69046174e-06,  -4.47828887e-08,
               -3.67070759e-05,   2.53827543e-05,  -1.88069561e-05,
                2.05942231e-03,  -5.95566325e-10,   1.00881255e-09,
                6.11553989e-10,  -4.11737126e-10,   6.27370976e-10,
                3.42275389e-06,  -8.17017771e-03,   4.01803819e-03,
                6.78767571e-02,   4.29444354e-02,   4.45551518e-08,
               -8.23509210e-08,  -7.90182526e-08,  -1.48650850e-07,
               -2.36138203e-08,  -4.75130905e-05,  -1.81655894e-05,
                9.79852186e-04,  -1.49411705e-02,  -2.35701903e-10,
                2.32406866e-09,   1.48224703e-09,  -1.09016625e-09,
               -1.31162142e-09,   1.47593911e-02,  -1.84965301e-01,
               -1.62413731e-01,   2.38867744e-07,   2.09579112e-07,
                6.11572155e-07,   1.44891826e-06,  -4.94537953e-07,
               -3.30400642e-01,   7.93664407e-01,   7.76571489e-08,
               -1.64476914e-07,  -2.13414311e-07,   4.75810302e-07,
                2.55787543e-07,  -6.37972323e+00,  -3.77364196e-06,
                8.65062737e-08,   6.13207853e-06,   8.39233482e-07,
               -3.76402983e-06,  -2.02633500e-05,  -7.28788200e-06,
               -2.72030382e-05,  -1.33298507e-05,   1.11838930e-05,
                9.74762098e-14,  -2.37844072e-14,  -1.11310490e-13,
                8.91237008e-14,   1.16770903e-13,   5.77230478e-15,
               -4.87322338e-14,   9.62949381e-14,  -2.12122129e-13,
                1.55871983e-13]
   
    for xx,nam in zip(X, names[idx]):
      print_min_max(xx, nam)

    X_i  = []
    X_i.extend(X)
     
    for i,xx in enumerate(X):
      if mdl == 'Ubar' or mdl == 'U' and not use_temp:
        k = i
      else:
        k = i+1
      for yy in X[k:]:
        X_i.append(xx*yy)
    
    #self.beta_f = exp(Constant(bhat[0]))
    self.beta_f = Constant(bhat[0])
    
    for xx,bb in zip(X_i, bhat[1:]):
      self.beta_f += Constant(bb)*xx
      #self.beta_f *= exp(Constant(bb)*xx)
    self.beta_f = exp(self.beta_f)
    
    if config['mode'] == 'steady':
      beta0                   = project(self.beta_f, Q)
      beta0_v                 = beta0.vector().array()
      beta0_v[beta0_v < 1e-2] = 1e-2
      self.assign_variable(beta0, beta0_v)
    
      self.assign_variable(self.beta, 1e-2)
      bc_beta = DirichletBC(self.Q, beta0, self.ff, 3)
      bc_beta.apply(self.beta.vector())
    elif config['mode'] == 'transient':
      self.assign_variable(self.beta, 200.0)
    
    print_min_max(self.beta, 'beta0')
     
  def init_b(self, b, U_ob, gradS):
    r"""
    Init rate-factor b from U_ob. 
    """
    s = "::: initializing b from U_ob :::"
    print_text(s, self.color)
   
    x      = self.x
    S      = self.S
    Q      = self.Q
    rhoi   = self.rhoi
    rhow   = self.rhow
    g      = self.g
    u      = U_ob[0]
    v      = U_ob[1]
    n      = 3.0
    
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = min(0, x[2])
    D = Depth(element = Q.ufl_element())
    
    N      = FacetNormal(self.mesh)
    dSde   = ds(4)
           
    b_f    = TrialFunction(Q)
    phi    = TestFunction(Q)

    epi    = self.BP_strain_rate(U_ob)
    ep_xx  = epi[0,0]
    ep_yy  = epi[1,1]
    ep_xy  = epi[0,1]
    ep_xz  = epi[0,2]
    ep_yz  = epi[1,2]   
 
    epsdot = ep_xx**2 + ep_yy**2 + ep_xx*ep_yy + ep_xy**2 + ep_xz**2 + ep_yz**2
    eta    = 0.5 * b_f * (epsdot + 1e-10)**((1-n)/(2*n))

    f_w    = rhoi*g*(S - x[2]) + rhow*g*D

    epi_1  = as_vector([   2*u.dx(0) + v.dx(1), 
                        0.5*(u.dx(1) + v.dx(0)),
                        0.5* u.dx(2)            ])
    epi_2  = as_vector([0.5*(u.dx(1) + v.dx(0)),
                             u.dx(0) + 2*v.dx(1),
                        0.5* v.dx(2)            ])

    R  = - 2 * eta * dot(epi_1, grad(phi)) * dx \
         + rhoi * g * gradS[0] * phi * dx \
         #+ 2 * eta * dot(epi_2, grad(phi)) * dx \
         #+ rhoi * g * gradS[1] * phi * dx \
   
    b_f = Function(Q)
    solve(lhs(R) == rhs(R), b_f)
    self.assign_variable(b, b_f)

  def unify_eta(self):
    """
    Unifies viscosity defined over grounded and shelves to model.eta.
    """
    s = "::: unifying viscosity on shelf and grounded areas to model.eta :::"
    print_text(s, self.color)
    
    eta_shf = project(self.eta_shf, self.Q)
    eta_gnd = project(self.eta_gnd, self.Q)
   
    # remove areas where viscosities overlap : 
    eta_shf.vector()[self.gnd_dofs] = 0.0
    eta_gnd.vector()[self.shf_dofs] = 0.0

    # unify eta to self.eta :
    eta = project(eta_shf + eta_gnd, self.Q)
    self.assign_variable(self.eta, eta)
    print_min_max(self.eta, 'eta')
 
  def calc_eta(self):
    """
    Calculates viscosity, set to model.eta.
    """
    s     = "::: calculating viscosity :::"
    print_text(s, self.color)
    config  = self.config
    Q       = self.Q
    R       = self.R
    T       = self.T
    W       = self.W
    n       = self.n
    u       = self.u
    v       = self.v
    w       = self.w
    eps_reg = self.eps_reg
    E_shf   = self.E_shf
    E_gnd   = self.E_gnd
    E       = self.E
    U       = as_vector([u,v,w])
    
    if config['velocity']['full_BP'] or config['model_order'] == 'stokes':
      epsdot = self.effective_strain(U)
    else:
      epsdot = self.BP_effective_strain(U)

    # manually calculate a_T and Q_T to avoid oscillations with 'conditional' :
    a_T    = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
    Q_T    = conditional( lt(T, 263.15), 6e4,          13.9e4)
    #a_T     = Function(Q)
    #Q_T     = Function(Q)
    #T_v     = T.vector().array()
    #a_T_v   = a_T.vector().array()
    #Q_T_v   = Q_T.vector().array()
    #a_T_v[T_v  < 263.15] = 1.1384496e-5
    #a_T_v[T_v >= 263.15] = 5.45e10 
    #Q_T_v[T_v  < 263.15] = 6e4
    #Q_T_v[T_v >= 263.15] = 13.9e4 
    #self.assign_variable(a_T, a_T_v)
    #self.assign_variable(Q_T, Q_T_v)
   
    # unify the enhancement factor over shelves and grounded ice : 
    E   = Function(Q)
    E_v = E.vector().array()
    E_gnd_v = E_gnd.vector().array()
    E_shf_v = E_shf.vector().array()
    E_v[self.gnd_dofs] = E_gnd_v[self.gnd_dofs]
    E_v[self.shf_dofs] = E_shf_v[self.shf_dofs]
    self.assign_variable(E, E_v)

    # calculate viscosity :
    b       = ( E*a_T*(1 + 181.25*W)*exp(-Q_T/(R*T)) )**(-1/n)
    eta     = 0.5 * b * (epsdot + eps_reg)**((1-n)/(2*n))
    eta     = project(eta, Q)
    self.assign_variable(self.eta, eta)
    print_min_max(self.eta, 'eta')

  def strain_rate_tensor(self, U):
    """
    return the strain-rate tensor of <U>.
    """
    return 0.5 * (grad(U) + grad(U).T)

  def BP_strain_rate_tensor(self, U):
    """
    return the 'Blatter-Pattyn' simplified strain-rate tensor of <U>.
    """
    u,v,w = U
    epi   = 0.5 * (grad(U) + grad(U).T)
    epi02 = 0.5*u.dx(2)
    epi12 = 0.5*v.dx(2)
    epi22 = -u.dx(0) - v.dx(1)  # incompressibility
    epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02],
                        [epi[1,0],  epi[1,1],  epi12],
                        [epi02,     epi12,     epi22]])
    return epsdot
    
  def effective_strain(self, U):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
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
    
  def BP_effective_strain(self, U):
    """
    return the BP effective strain rate squared.
    """
    epi    = self.BP_strain_rate_tensor(U)
    ep_xx  = epi[0,0]
    ep_yy  = epi[1,1]
    ep_zz  = epi[2,2]
    ep_xy  = epi[0,1]
    ep_xz  = epi[0,2]
    ep_yz  = epi[1,2]
    
    # Second invariant of the strain rate tensor squared
    epsdot = + ep_xx**2 + ep_yy**2 + ep_xx*ep_yy \
             + ep_xy**2 + ep_xz**2 + ep_yz**2
    return epsdot

  def stress_tensor(self):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the Cauchy stress tensor :::"
    print_text(s, self.color)
    U   = as_vector([self.u, self.v, self.w])
    epi = self.strain_rate_tensor(U)
    dim = self.mesh.ufl_cell().topological_dimension()
    I   = Identity(dim)

    sigma = 2*self.eta*epi - self.P*I
    return sigma

  def BP_stress_tensor(self):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the BP Cauchy stress tensor :::"
    print_text(s, self.color)
    U   = as_vector([self.u, self.v, self.w])
    epi = self.BP_strain_rate_tensor(U)
    dim = self.mesh.ufl_cell().topological_dimension()
    I   = Identity(dim)

    sigma = 2*self.eta*epi - self.P*I
    return sigma
  
  def calc_BP_pressure(self):
    """
    Calculate the continuous pressure field.
    """
    s    = "::: calculating pressure :::"
    print_text(s, self.color)
    rhoi = self.rhoi
    g    = self.g
    S    = self.S
    x    = self.x
    eta  = self.eta
    w    = self.w
    p    = project(rhoi*g*(S - x[2]) + 2*eta*w.dx(2), self.Q)
    self.assign_variable(self.P, p)
    print_min_max(self.P, 'p')
  
  def calc_thickness(self):
    """
    Calculate the continuous thickness field which increases from 0 at the 
    surface to the actual thickness at the bed.
    """
    s = "::: calculating z-varying thickness :::"
    print_text(s, self.color)
    H = project(self.S - self.x[2], self.Q)
    print_min_max(H, 'H')
    return H
  
  def vert_extrude(self, u, d='up', Q='self'):
    r"""
    This extrudes a function <u> vertically in the direction <d> = 'up' or
    'down'.
    It does this by formulating a variational problem:
  
    :Conditions: 
    .. math::
    \frac{\partial v}{\partial z} = 0
    
    v|_b = u
  
    and solving.  
    """
    s = "::: extruding function :::"
    print_text(s, self.color)
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    a   = v.dx(2) * phi * dx
    L   = DOLFIN_EPS * phi * dx
    bcs = []
    # extrude bed (ff = 3,5) 
    if d == 'up':
      bcs.append(DirichletBC(Q, u, ff, 3))  # grounded
      bcs.append(DirichletBC(Q, u, ff, 5))  # shelves
    # extrude surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, u, ff, 2))  # grounded
      bcs.append(DirichletBC(Q, u, ff, 6))  # shelves
    v   = Function(Q)
    solve(a == L, v, bcs)
    print_min_max(u, 'function to be extruded')
    print_min_max(v, 'extruded function')
    return v
  
  def vert_integrate(self, u, d='up', Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
    s = "::: vertically integrating function :::"
    print_text(s, self.color)

    if type(Q) != FunctionSpace:
      Q = self.Q
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    bcs = []
    # integral is zero on bed (ff = 3,5) 
    if d == 'up':
      bcs.append(DirichletBC(Q, 0.0, ff, 3))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, 5))  # shelves
      a      = v.dx(2) * phi * dx
    # integral is zero on surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, 0.0, ff, 2))  # grounded
      bcs.append(DirichletBC(Q, 0.0, ff, 6))  # shelves
      a      = -v.dx(2) * phi * dx
    L      = u * phi * dx
    v      = Function(Q)
    solve(a == L, v, bcs)
    print_min_max(u, 'vertically integrated function')
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
    print_text(s, self.color)
    ubar = project(uhat/H, self.Q)
    print_min_max(ubar, 'ubar')
    ubar = self.vert_extrude(ubar, d='down')
    return ubar

  def calc_misfit(self, integral):
    """
    Calculates the misfit of model and observations, 

      D = ||U - U_ob||

    over shelves or grounded depending on the paramter <integral>, then 
    updates model.misfit with D for plotting.
    """
    s   = "::: calculating misfit L-infty norm ||U - U_ob|| over '%s' :::"
    print_text(s % integral, self.color)

    Q2     = self.Q2
    ff     = self.ff
    U_s    = Function(Q2)
    U_ob_s = Function(Q2)
    U      = as_vector([self.u,    self.v])
    U_ob   = as_vector([self.u_ob, self.v_ob])

    if integral == 'shelves':
      bc_U    = DirichletBC(self.Q2, U,    ff, 6)
      bc_U_ob = DirichletBC(self.Q2, U_ob, ff, 6)
    elif integral == 'grounded':
      bc_U    = DirichletBC(self.Q2, U,    ff, 2)
      bc_U_ob = DirichletBC(self.Q2, U_ob, ff, 2)
    
    bc_U.apply(U_s.vector())
    bc_U_ob.apply(U_ob_s.vector())

    # calculate L_inf vector norm :
    U_s_v    = U_s.vector().array()
    U_ob_s_v = U_ob_s.vector().array()
    D_v      = U_s_v - U_ob_s_v
    D        = MPI.max(mpi_comm_world(), D_v.max())
    
    s    = "||U - U_ob|| : %.3E" % D
    print_text(s, '208', 1)
    self.misfit = D

  def get_theta(self):
    """
    Returns the angle in radians of the horizontal velocity vector from 
    the x-axis.
    """
    u_v     = self.u.vector().array()
    v_v     = self.v.vector().array()
    theta_v = np.arctan2(u_v, v_v)
    theta   = Function(self.Q)
    self.assign_variable(theta, theta_v)
    return theta

  def rotate(self, M, theta):
    """
    rotate the tensor <M> about the z axes by angle <theta>.
    """
    c  = cos(theta)
    s  = sin(theta)
    Rz = as_matrix([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])
    R  = dot(Rz, dot(M, Rz.T))
    return R

  def get_norm(self, U, type='l2'):
    """
    returns the norm of vector <U>.
    """
    # iterate through each component and convert to array :
    U_v = []
    for u in U:
      # convert to array and normailze the components of U :
      u_v = u.vector().array()
      U_v.append(u_v)
    U_v = np.array(U_v)

    # calculate the norm :
    if type == 'l2':
      norm_u = np.sqrt(np.sum(U_v**2,axis=0))
    elif type == 'linf':
      norm_u = np.amax(U_v,axis=0)
    
    return U_v, norm_u

  def normalize_vector(self, U, Q='self'):
    """
    Create a normalized vector of the UFL vector <U>.
    """
    if type(Q) != FunctionSpace:
      Q = self.Q

    U_v, norm_u = self.get_norm(U)

    norm_u[norm_u <= 0.0] = 1e-15
    
    # normalize the vector :
    U_v /= norm_u
    
    # convert back to fenics :
    U_f = []
    for u_v in U_v:
      u_f = Function(Q)
      self.assign_variable(u_f, u_v)
      U_f.append(u_f)

    # return a UFL vector :
    return as_vector(U_f)

  def assign_variable(self, u, var):
    """
    Manually assign the values from <var> to Function <u>.  <var> may be an
    array, float, Expression, or Function.
    """
    if   isinstance(var, pc.PhysicalConstant):
      u.vector()[:] = var.real

    elif isinstance(var, float) or isinstance(var, int):
      u.vector()[:] = var
    
    elif isinstance(var, np.ndarray):
      u.vector().set_local(var)
      u.vector().apply('insert')
    
    elif isinstance(var, Expression) or isinstance(var, Constant)  \
         or isinstance(var, GenericVector) or isinstance(var, Function):
      u.interpolate(var)

    elif isinstance(var, str):
      File(var) >> u

    else:
      s =  "*************************************************************\n" + \
           "assign_variable() function requires a Function, array, float,\n" + \
           " int, Vector, Expression, or string path to .xml, not \n" + \
           "%s.  Replacing object entirely\n" + \
           "*************************************************************"
      print_text(s % type(var) , 'red')
      u = var

  def globalize_parameters(self, namespace=None):
    """
    This function converts the parameter dictinary into global object
    
    :param namespace: Optional namespace in which to place the global variables
    """
    for v in self.variables.iteritems():
      vars(namespace)[v[0]] = v[1]

  def save_pvd(self, var, name):
    """
    Save a <name>.pvd file of the FEniCS Function <var> to this model's log 
    directory specified by the config['output_path'] field.
    """
    outpath = self.config['output_path']
    s       = "::: saving %s%s.pvd file :::" % (outpath, name)
    print_text(s, self.color)
    File(outpath + name + '.pvd') << var

  def save_xml(self, var, name):
    """
    Save a <name>.xml file of the FEniCS Function <var> to this model's log 
    directory specified by the config['output_path'] field.
    """
    outpath = self.config['output_path']
    s       = "::: saving %s%s.xml file :::" % (outpath, name)
    print_text(s, self.color)
    File(outpath + '/' +  name + '.xml') << var

  def init_viscosity_mode(self, mode):
    """
    Initialize the rate-factor b, viscosit eta, and viscous dissipation term 
    used by the Enthalpy and Dukowicz velocity solvers.  The values of <mode>
    may be :
    
      'isothermal' :  use the values of model.A0 for b = A0^{-1/n}
      'linear'     :  use the values in the current model.u, model.v, 
                      and model.w to form the viscosity; requires that b has 
                      been initialized previously
      'full'       :  use the full temperature-dependent rate factor.

    """
    s = "::: initializing viscosity :::"
    print_text(s, self.color)

    config = self.config

    # Set the value of b, the temperature dependent ice hardness parameter,
    # using the most recently calculated temperature field.
    #
    #   eta = visosity,
    #   b   = rate factor
    #   Vd  = viscous dissipation
    #   shf = volume over shelves
    #   gnd = volume over grounded ice
    #
    if   mode == 'isothermal':
      s  = "    - using isothermal visosity formulation -"
      print_text(s, self.color)
      print_min_max(self.A0, 'A')
      A0         = self.A0
      n          = self.n
      epsdot     = self.epsdot
      eps_reg    = self.eps_reg
      b          = A0**(-1/n)
      self.b_shf = b
      self.b_gnd = b
      print_min_max(self.b_shf, 'b')
      self.eta_shf = 0.5 * self.b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      self.eta_gnd = 0.5 * self.b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      self.Vd_shf  = (2*n)/(n+1)*self.b_shf*(epsdot + eps_reg)**((n+1)/(2*n))
      self.Vd_gnd  = (2*n)/(n+1)*self.b_gnd*(epsdot + eps_reg)**((n+1)/(2*n))
    
    elif mode == 'linear':
      s     = "    - using linear viscosity formulation -"
      print_text(s, self.color)
      n         = self.n
      eps_reg   = self.eps_reg
      u_cpy     = self.u.copy(True)
      v_cpy     = self.v.copy(True)
      w_cpy     = self.w.copy(True)
      U         = as_vector([u_cpy, v_cpy, w_cpy])
      if config['velocity']['full_BP'] or config['model_order'] == 'stokes':
        epsdot  = self.effective_strain(U)
      else:
        epsdot  = self.BP_effective_strain(U)
      self.eta_shf = 0.5 * self.b_shf * epsdot**((1-n)/(2*n))
      self.eta_gnd = 0.5 * self.b_gnd * epsdot**((1-n)/(2*n))
      self.Vd_shf  = 2 * self.eta_shf * self.epsdot
      self.Vd_gnd  = 2 * self.eta_gnd * self.epsdot
    
    elif mode == 'full':
      s     = "    - using full viscosity formulation -"
      print_text(s, self.color)
      n       = self.n
      epsdot  = self.epsdot
      eps_reg = self.eps_reg
      T       = self.T
      W       = self.W
      R       = self.R
      E_shf   = self.E_shf
      E_gnd   = self.E_gnd
      a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
      self.b_shf   = ( E_shf*a_T*(1 + 181.25*W)*exp(-Q_T/(R*T)) )**(-1/n)
      self.b_gnd   = ( E_gnd*a_T*(1 + 181.25*W)*exp(-Q_T/(R*T)) )**(-1/n)
      self.eta_shf = 0.5 * self.b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      self.eta_gnd = 0.5 * self.b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      self.Vd_shf  = (2*n)/(n+1)*self.b_shf*(epsdot + eps_reg)**((n+1)/(2*n))
      self.Vd_gnd  = (2*n)/(n+1)*self.b_gnd*(epsdot + eps_reg)**((n+1)/(2*n))
    
    else:
      s = "    - ACCEPTABLE CHOICES FOR VISCOSITY ARE 'linear', " + \
          "'isothermal', OR 'full' -"
      print_text(s, 'red', 1)
      sys.exit(1)
    
  def init_hybrid_variables(self):
    """
    """
    # hybrid mass-balance :
    self.H             = Function(self.Q)
    self.H0            = Function(self.Q)
    self.ubar_c        = Function(self.Q)
    self.vbar_c        = Function(self.Q)
    self.H_min         = Function(self.Q)
    self.H_max         = Function(self.Q)

    # hybrid energy-balance :
    N_T                = self.config['enthalpy']['N_T']
    self.deltax        = 1./(N_T-1.)
    self.sigmas        = np.linspace(0, 1, N_T, endpoint=True)
    self.T_            = Function(self.Z)
    self.T0_           = Function(self.Z)
    self.Ts            = Function(self.Q)
    self.Tb            = Function(self.Q)
    
    # hybrid momentum :
    self.U             = Function(self.HV)
    self.u_s           = Function(self.Q)
    self.v_s           = Function(self.Q)
    self.w_s           = Function(self.Q)
    self.u_b           = Function(self.Q)
    self.v_b           = Function(self.Q)
    self.w_b           = Function(self.Q)
    
  def init_SSA_variables(self):
    """
    """
    s = "    - initializing BP variables -"
    print_text(s, self.color)
    
    self.U   = Function(self.Q2)
    self.dU  = TrialFunction(self.Q2)
    self.Phi = TestFunction(self.Q2)
    self.Lam = Function(self.Q2)

    self.etabar = Function(self.Q)
    self.ubar   = Function(self.Q)
    self.vbar   = Function(self.Q)
    
    #self.epsdot = self.effective_strain(self.U)
    #self.init_higher_order_variables()

  def init_BP_variables(self):
    """
    """
    if self.config['velocity']['full_BP']:
      s = "    - initializing full BP variables -"
      self.U   = Function(self.Q3)
      self.dU  = TrialFunction(self.Q3)
      self.Phi = TestFunction(self.Q3)
      self.Lam = Function(self.Q3)
    else :
      s = "    - initializing BP variables -"
      self.U   = Function(self.Q2)
      self.dU  = TrialFunction(self.Q2)
      self.Phi = TestFunction(self.Q2)
      self.Lam = Function(self.Q2)
    print_text(s, self.color)
    
    # Second invariant of the strain rate tensor squared
    if self.config['velocity']['full_BP']:
      self.epsdot = self.effective_strain(self.U)
    else:
      U_t = as_vector([self.U[0], self.U[1], 0.0])
      self.epsdot = self.BP_effective_strain(U_t)
    self.init_higher_order_variables()

  def init_stokes_variables(self):
    """
    """
    s = "    - initializing full-stokes variables -"
    print_text(s, self.color)
    # velocity :
    if self.config['use_dukowicz']:
      self.U   = Function(self.Q4)
      self.dU  = TrialFunction(self.Q4)
      self.Phi = TestFunction(self.Q4)
      self.Lam = Function(self.Q4)
      U        = as_vector([self.U[0], self.U[1], self.U[2]])
    else:
      self.U   = Function(self.MV)
      self.dU  = TrialFunction(self.MV)
      self.Phi = TestFunction(self.MV)
      self.Lam = Function(self.MV)
      U, P     = split(self.U)
    
    # Second invariant of the strain rate tensor squared
    self.epsdot = self.effective_strain(U)
    self.init_higher_order_variables()

  def init_higher_order_variables(self):
    """
    """
    s = "    - initializing higher-order variables -"
    print_text(s, self.color)

    # sigma coordinate :
    self.sigma         = project((self.x[2] - self.B) / (self.S - self.B))
    print_min_max(self.sigma, 'sigma')

    # free surface model :
    self.Shat          = Function(self.Q_flat)
    self.dSdt          = Function(self.Q_flat)
    self.ahat          = Function(self.Q_flat)
    self.uhat_f        = Function(self.Q_flat)
    self.vhat_f        = Function(self.Q_flat)
    self.what_f        = Function(self.Q_flat)
    self.M             = Function(self.Q_flat)
    
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    s = "::: initializing variables :::"
    print_text(s, self.color)

    config = self.config
    
    # initialize constants and make them globally available :
    self.set_parameters(pc.IceParameters())
    self.params.globalize_parameters(self)

    # Coordinates of various types 
    self.x             = SpatialCoordinate(self.mesh)
    self.h             = CellSize(self.mesh)
    self.N             = FacetNormal(self.mesh)

    # shelf mask (1 if shelf) :
    self.mask          = Function(self.Q)
    
    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = min(0, x[2])
    self.D = Depth(element=self.Q.ufl_element())

    # Velocity model
    self.u             = Function(self.Q)
    self.v             = Function(self.Q)
    self.w             = Function(self.Q)
    self.P             = Function(self.Q)
    self.beta          = Function(self.Q)
    self.mhat          = Function(self.Q)
    self.E             = Function(self.Q)
    self.E_gnd         = Function(self.Q)
    self.E_shf         = Function(self.Q)
    self.eta           = Function(self.Q)
    self.u_ob          = Function(self.Q)
    self.v_ob          = Function(self.Q)
    self.U_ob          = Function(self.Q)
    self.u_lat         = Function(self.Q)
    self.v_lat         = Function(self.Q)
    self.w_lat         = Function(self.Q)
    
    # Enthalpy model
    self.theta_surface = Function(self.Q)
    self.theta_float   = Function(self.Q)
    self.theta         = Function(self.Q)
    self.theta0        = Function(self.Q) # initial enthalpy
    self.T             = Function(self.Q)
    self.q_geo         = Function(self.Q)
    self.W0            = Function(self.Q)
    self.W             = Function(self.Q)
    self.Mb            = Function(self.Q)
    self.thetahat      = Function(self.Q) # Midpoint values
    self.uhat          = Function(self.Q) # Midpoint values
    self.vhat          = Function(self.Q) # Midpoint values
    self.what          = Function(self.Q) # Midpoint values
    self.mhat          = Function(self.Q) # ALE is required: we change the mesh 
    self.T_melt        = Function(self.Q) # pressure-melting point
    self.kappa         = Function(self.Q)
    self.Kcoef         = Function(self.Q)

    # Age model   
    self.age           = Function(self.Q)
    self.a0            = Function(self.Q)

    # Surface climate model
    self.smb           = Function(self.Q)
    self.precip        = Function(self.Q)
    self.T_surface     = Function(self.Q)

    # Adjoint model
    self.adot          = Function(self.Q)
    self.adj_f         = 0.0              # objective function value at end
    self.misfit        = 0.0              # ||U - U_ob||

    # Balance Velocity model :
    self.kappa         = Function(self.Q)
    self.adot          = Function(self.Q)
    self.dSdx          = Function(self.Q)
    self.dSdy          = Function(self.Q)
    self.d_x           = Function(self.Q)
    self.d_y           = Function(self.Q)
    self.Ubar          = Function(self.Q)
    self.Nx            = Function(self.Q)
    self.Ny            = Function(self.Q)
    
    # Stokes-balance model :
    self.u_s           = Function(self.Q)
    self.u_t           = Function(self.Q)
    self.F_id          = Function(self.Q)
    self.F_jd          = Function(self.Q)
    self.F_ib          = Function(self.Q)
    self.F_jb          = Function(self.Q)
    self.F_ip          = Function(self.Q)
    self.F_jp          = Function(self.Q)
    self.F_ii          = Function(self.Q)
    self.F_ij          = Function(self.Q)
    self.F_iz          = Function(self.Q)
    self.F_ji          = Function(self.Q)
    self.F_jj          = Function(self.Q)
    self.F_jz          = Function(self.Q)
    self.tau_id        = Function(self.Q)
    self.tau_jd        = Function(self.Q)
    self.tau_ib        = Function(self.Q)
    self.tau_jb        = Function(self.Q)
    self.tau_ip        = Function(self.Q)
    self.tau_jp        = Function(self.Q)
    self.tau_ii        = Function(self.Q)
    self.tau_ij        = Function(self.Q)
    self.tau_iz        = Function(self.Q)
    self.tau_ji        = Function(self.Q)
    self.tau_jj        = Function(self.Q)
    self.tau_jz        = Function(self.Q)
    
    if   config['model_order'] == 'stokes':
      self.init_stokes_variables()
    elif config['model_order'] == 'BP':
      self.init_BP_variables()
    elif config['model_order'] == 'SSA':
      self.init_SSA_variables()
    elif config['model_order'] == 'L1L2':
      self.init_hybrid_variables()
    else:
      s = "    - PLEASE SPECIFY A MODEL ORDER; MAY BE 'stokes', 'BP', " + \
          "SSA, or 'L1L2' -"
      print_text(s, 'red', 1)
      sys.exit(1)

    



