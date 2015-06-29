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
    self.mesh.init(1,2)
    self.num_facets = self.mesh.size_global(2)
    self.num_cells  = self.mesh.size_global(3)
    self.dof        = self.mesh.size_global(0)
    s = "    - mesh set, %i cells, %i facets, %i vertices - "
    print_text(s % (self.num_cells, self.num_facets, self.dof), self.color)
    self.generate_function_spaces()

  def generate_function_spaces(self):
    """
    """
    s = "::: generating function spaces :::"
    print_text(s, self.color)
    if self.config['periodic_boundary_conditions']:
      self.generate_pbc()
    else:
      self.pBC = None
    self.Q      = FunctionSpace(self.mesh,      "CG", 1, 
                                constrained_domain=self.pBC)
    if self.config['model_order'] != 'L1L2':
      self.Q_flat = FunctionSpace(self.flat_mesh, "CG", 1, 
                                  constrained_domain=self.pBC)
      self.Q2     = MixedFunctionSpace([self.Q]*2)
      self.Q3     = MixedFunctionSpace([self.Q]*3)
      self.Q4     = MixedFunctionSpace([self.Q]*4)
      # Define function spaces
      self.V      = VectorFunctionSpace(self.mesh, "CG", 1)
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
    else:
      poly_degree = self.config['velocity']['poly_degree']
      N_T         = self.config['enthalpy']['N_T']
      self.HV     = MixedFunctionSpace([self.Q]*2*poly_degree) # VELOCITY
      self.Z      = MixedFunctionSpace([self.Q]*N_T)           # TEMPERATURE
    
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
    
  def init_beta_stats(self):
    """
    """
    s    = "::: initializing beta from stats :::"
    print_text(s, self.color)
    q_geo = self.q_geo
    T_s   = self.T_surface
    adot  = self.adot
    Mb    = self.Mb
    Ubar  = self.Ubar
    Q     = self.Q
    B     = self.B
    S     = self.S
    T     = self.T
    T_s   = self.T_surface

    adot_v = adot.vector().array()
    adot_v[adot_v < -1000] = 0
    self.assign_variable(adot, adot_v)

    D     = Function(Q)
    B_v   = B.vector().array()
    D_v   = D.vector().array()
    D_v[B_v < 0] = 1.0
    self.assign_variable(D, D_v)

    Mb_v = Mb.vector().array()
    Mb_v[Mb_v < 0.0]  = 0.0
    Mb_v[Mb_v > 10.0] = 10.0
    self.assign_variable(Mb, Mb_v)

    #nS   = Function(Q)
    #gSx  = project(S.dx(0)).vector().array()
    #gSy  = project(S.dx(1)).vector().array()
    #nS_v = np.sqrt(gSx**2 + gSy**2 + DOLFIN_EPS)
    #self.assign_variable(nS, nS_v)

    nS   = sqrt(inner(grad(S), grad(S)) + DOLFIN_EPS)

    #nB   = Function(Q)
    #gBx  = project(B.dx(0)).vector().array()
    #gBy  = project(B.dx(1)).vector().array()
    #nB_v = np.sqrt(gBx**2 + gBy**2 + DOLFIN_EPS)
    #self.assign_variable(nB, nB_v)

    nB   = sqrt(inner(grad(B), grad(B)) + DOLFIN_EPS)

    U_v  = as_vector([self.u, self.v, self.w])

    ini  = sqrt(917.0 * 9.8 * (S - B) * nS / (Ubar + 0.1))

    x0   = S
    x1   = T_s
    x2   = nS
    x3   = D
    x4   = nB
    x5   = S - B
    x6   = q_geo
    x7   = adot
    x8   = T
    x9   = Mb
    x10  = self.u
    x11  = self.v
    x12  = self.w
    x13  = ln(Ubar + 1)
    x14  = ln(sqrt(inner(U_v,U_v) + DOLFIN_EPS) + 1)
    x15  = ini

    X    = [x0,x1,x2,x4,x5,x7,x13,x15]

    for i,xx in enumerate(X):
      print_min_max(xx, 'x' + str(i))

    bhat = [ -1.02331153e+02,   1.11778868e-02,   7.98617990e-01,
             -2.49213406e+01,   7.01725486e+00,  -2.41266098e-03,
              2.69270964e+00,   1.09757497e+00,   5.09616214e-03,
             -2.93709460e-07,  -4.02852494e-05,   1.36192751e-03,
             -1.92061602e-04,   8.21958184e-08,   6.25273008e-05,
             -6.90628736e-05,  -1.28536531e-07,  -1.50452597e-03,
              9.56728797e-02,  -4.13422401e-02,   8.22401828e-06,
             -1.07791900e-02,  -3.99334789e-03,  -1.62785666e-05,
             -7.98760499e+00,   7.68069728e+00,  -9.24810351e-03,
             -7.59511137e-01,   3.23938736e-01,  -1.41462938e-04,
             -8.56973153e-01,   1.56510813e-04,   7.42539169e-01,
              3.45749805e-01,   4.91766770e-04,   5.96908631e-08,
              1.31701943e-04,  -8.65126169e-05,  -2.92338831e-07,
             -9.48427894e-02,   6.19006380e-03,   3.09552357e-05,
             -4.77862981e-04,   3.32653316e-03,   4.38710297e-08]

    X_i  = []
    X_i.extend(X)
     
    for i,xx in enumerate(X):
      for yy in X[i:]:
        X_i.append(xx*yy)
    
    #self.beta_f = exp(Constant(bhat[0]))
    self.beta_f = Constant(bhat[0])
    
    for xx,bb in zip(X_i, bhat[1:]):
      self.beta_f += Constant(bb)*xx
      #self.beta_f *= exp(Constant(bb)*xx)
    self.beta_f = exp(self.beta_f)
    
    beta                   = project(self.beta_f, Q)
    beta_v                 = beta.vector().array()
    beta_v[beta_v < 0.0]   = 0.0
    self.assign_variable(self.beta, beta_v)
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

    # surface gradient :
    self.gradS         = project(grad(self.S), self.V)
    print_min_max(self.gradS, 'grad(S)')

    # bed gradient :
    self.gradB         = project(grad(self.B), self.V)
    print_min_max(self.gradB, 'grad(B)')

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

    



