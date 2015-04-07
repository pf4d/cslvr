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
      self.V      = VectorFunctionSpace(self.mesh, "CG", 1)
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
    self.mask   = True
    self.ds     = Measure('ds')[self.ff]
    self.dx     = Measure('dx')[self.cf]

  def calculate_boundaries(self, mask=None, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    # default to all grounded ice :
    if mask == None:
      self.mask = Expression('0.0', element=self.Q.ufl_element())
    else:
      self.mask = mask
    
    # default to all positive accumulation :
    if adot == None:
      self.adot_ex = Expression('1.0', element=self.Q.ufl_element())
    else:
      self.adot_ex = adot
    
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
      mask_xy = self.mask(x_m, y_m, z_m)
      adot_xy = self.adot_ex(x_m, y_m, z_m)
      
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
      mask_xy = self.mask(x_m, y_m, z_m)
    
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
      mask_xy = self.mask(x_m, y_m, z_m)

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
  
  def init_E(self, E):
    """
    """
    s = "::: initializing enhancement factor :::"
    print_text(s, self.color)
    self.assign_variable(self.E, E)
    print_min_max(self.E, 'E')
  
  def init_beta(self, beta):
    """
    """
    s = "::: initializing basal friction coefficient :::"
    print_text(s, self.color)
    self.assign_variable(self.beta, beta)
    print_min_max(self.beta, 'beta')
  
  def init_b_shf(self, b_shf):
    """
    """
    s = "::: initializing rate-factor over shelves :::"
    print_text(s, self.color)
    if type(self.b_shf) != Function:
      self.b_shf = Function(self.Q)
    self.assign_variable(self.b_shf, b_shf)
    print_min_max(self.b_shf, 'b_shf')
  
  def init_b_gnd(self, b_gnd):
    """
    """
    s = "::: initializing rate-factor over grounded ice :::"
    print_text(s, self.color)
    self.assign_variable(self.b_gnd, b_gnd)
    print_min_max(self.b_gnd, 'b_gnd')
  
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

  def set_parameters(self, params):
    """
    Sets the model's dictionary of parameters
    
    :param params: :class:`~src.physical_constants.IceParameters` object 
       containing model-relavent parameters
    """
    self.params = params

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
      if Facet(self.mesh, cellmap[c.index()]).normal().z() < 0:
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
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    self.assign_variable(self.beta, beta_0_v)
    print_min_max(self.beta, 'beta')
    self.betaSIA = Function(Q)
    self.assign_variable(self.betaSIA, beta_0_v)
      
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
    
    Ne       = H - rhow/rhoi * D
    lnC      = ln(0.383)
    
    U_s      = Function(Q)
    if U_mag == None:
      U_v = self.U_ob.vector().array()
    else:
      U_v = U_mag.vector().array()
    U_v[U_v < eps] = eps
    self.assign_variable(U_s, U_v)
    
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta     = U_s * Ne**(0.349) * exp(lnC * rhoi * g * H * S_mag)
    #beta     = U_s * exp(lnC * rhoi * g * H * S_mag)
    beta_0   = project(sqrt(beta), Q)
    print_min_max(beta_0, 'beta_0')
    
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    self.assign_variable(self.beta, beta_0_v)
    print_min_max(self.beta, 'beta')
    self.betaSIA = Function(Q)
    self.assign_variable(self.betaSIA, beta_0_v)

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
    T     = self.Tb
    T_s   = self.T_surface

    adot_v = adot.vector().array()
    adot_v[adot_v < -1000] = 0
    self.assign_variable(adot, adot_v)

    Ubar_v = Ubar.vector().array()
    Ubar_v[Ubar_v < 0] = 0
    self.assign_variable(Ubar, Ubar_v)

    absB  = Function(Q)
    B_v   = np.abs(B.vector().array())
    self.assign_variable(absB, B_v)

    absMb = Function(Q)
    Mb_v  = np.abs(Mb.vector().array())
    Mb_v[Mb_v > 40] = 40
    self.assign_variable(absMb, Mb_v)

    nS   = Function(Q)
    gSx  = project(S.dx(0)).vector().array()
    gSy  = project(S.dx(1)).vector().array()
    nS_v = np.sqrt(gSx**2 + gSy**2 + DOLFIN_EPS)
    self.assign_variable(nS, nS_v)

    nB   = Function(Q)
    gBx  = project(B.dx(0)).vector().array()
    gBy  = project(B.dx(1)).vector().array()
    nB_v = np.sqrt(gBx**2 + gBy**2 + DOLFIN_EPS)
    self.assign_variable(nB, nB_v)

    U_v  = as_vector([self.u, self.v])

    x0   = S
    x1   = T_s
    x2   = nS
    x3   = absB
    x4   = nB
    x5   = S - B
    x6   = self.u
    x7   = self.v
    x8   = self.w
    x9   = q_geo
    x10  = adot
    x11  = ln(Ubar + 1)
    x12  = T
    x13  = absMb
    x14  = ln(sqrt(inner(U_v,U_v) + DOLFIN_EPS) + 1)

    #X    = [x0,x1,x2,x3,x4,x5,x10,x11]
    X    = [x0,x1,x2,x3,x4,x5,x10,x11,x12,x13,x14]

    for i,xx in enumerate(X):
      print_min_max(xx, 'x' + str(i))

    # GLM :
    bhat = [ 1.26487569e+01,   2.34567816e-03,  -5.34422108e-02,
             3.93804046e+00,  -4.12801624e-03,   3.25268549e+01,
             1.95255714e-03,   1.06370874e+00,   6.79133343e-01,
            -3.54872218e-02,   4.09903244e-02,  -2.03633122e+00,
            -2.99760096e-06,  -2.29650440e-03,   1.30059715e-07,
             7.63794398e-04,  -2.28886568e-07,  -9.41355141e-05,
            -3.79308468e-05,  -4.02590130e-06,  -1.38896798e-06,
            -3.95327785e-05,   1.16775433e-02,   7.68482665e-07,
            -2.03358934e-03,   1.38949356e-06,   1.96293602e-04,
            -1.68177974e-03,   2.44589171e-04,  -9.73848142e-05,
             4.99092829e-05,   2.33623868e-03,  -8.05881038e+00,
             1.86332339e-03,   1.65167828e-01,  -2.81735481e-01,
            -1.43265646e-02,  -1.45607775e-02,  -7.48203064e-02,
            -9.26079605e-04,   1.40037572e-07,   1.43865686e-04,
            -3.79278444e-05,   1.27236289e-05,   3.48409555e-06,
             2.40578092e-05,  -2.95606621e-04,  -5.88401039e-01,
             1.53401653e-01,  -1.19255343e-01,   3.97027906e-05,
             3.90281123e-01,   7.16558267e-05,   9.30684521e-05,
            -8.56621714e-06,   2.49187662e-06,  -4.40625856e-05,
            -2.83085289e-02,  -3.49092574e-03,   7.84665148e-03,
             1.84762987e-02,  -1.37814457e-03,   5.94246849e-04,
             2.74922117e-02,  -7.16825516e-05,   5.14383788e-03,
             1.69906987e-05]

    ## combined :
    #bhat = [-3.38068626e+01,   8.00815477e-03,   4.32502670e-02,
    #        -1.55898502e+01,  -1.03695611e-02,   1.31647095e+02,
    #         7.65619003e-03,  -4.03972486e+00,   1.85285334e+00,
    #         1.41227706e-01,   1.04228509e-01,  -2.37682966e+00,
    #        -5.03594634e-06,  -1.01495464e-02,   4.06437973e-07,
    #        -2.40096032e-03,  -4.22744068e-07,  -4.14768698e-04,
    #         4.67576058e-05,  -2.28183640e-05,  -6.44380751e-06,
    #         6.59824198e-05,   2.62059227e-02,  -5.16642217e-06,
    #        -3.27126033e-02,  -5.41842588e-06,   2.92134501e-03,
    #        -3.69584445e-03,  -2.08059738e-05,  -4.17568430e-04,
    #         2.53494901e-04,   9.66433564e-03,  -9.42920821e+00,
    #         9.09977967e-03,   1.22402288e+00,  -9.17635883e-01,
    #         5.03523468e-02,   6.75328534e-02,   4.72455314e-01,
    #         1.30016821e-03,   2.80117874e-07,   7.43252200e-04,
    #        -2.54320922e-04,   4.00502614e-05,   1.83641082e-05,
    #        -3.05722555e-05,   2.45430345e-03,  -7.63761162e-01,
    #         7.00892423e-01,  -4.52893496e-01,  -6.42302280e-02,
    #        -7.41595393e-01,   6.58530850e-04,   4.16585868e-04,
    #        -2.50902590e-05,   1.44210507e-05,  -2.95156478e-04,
    #        -6.98435060e-02,   1.24163144e-02,   3.08524704e-02,
    #         3.60910325e-02,  -5.69395057e-03,   1.58658208e-03,
    #         8.96009614e-02,  -8.95661225e-05,   3.30803500e-03,
    #         4.91981788e-03]

    X_i  = []
    X_i.extend(X)
     
    for i,xx in enumerate(X):
      for yy in X[i+1:]:
        X_i.append(xx*yy)
    
    self.beta_f = exp(Constant(bhat[0]))
    
    for xx,bb in zip(X_i, bhat[1:]):
      self.beta_f *= exp(Constant(bb)*xx)
      #self.beta_f += Constant(bb)*xx
    #self.beta_f = exp(self.beta_f)
    
    beta                    = project(self.beta_f, Q)
    beta_v                  = beta.vector().array()
    beta_v[beta_v < 0.0]    = 0.0
    self.assign_variable(self.beta, beta)
    print_min_max(self.beta, 'beta0')
    #self.init_beta_SIA(Ubar)
     
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
    """
    s = "::: unifying viscosity on shelf and grounded areas to model.eta :::"
    print_text(s, self.color)
    eta_shf = self.eta_shf
    eta_gnd = self.eta_gnd
    dx      = self.dx
    dx_s    = dx(1)
    dx_g    = dx(0)
    Q       = self.Q
    psi_i   = TestFunction(Q)
    psi_j   = TrialFunction(Q)
    M       = assemble(psi_i * psi_j * dx)
    eta     = assemble(eta_shf*psi_i*dx_s + eta_gnd*psi_i*dx_g)
    solve(M, self.eta.vector(), eta)
    print_min_max(self.eta, 'eta')

  def calc_eta(self):
    s     = "::: calculating visosity :::"
    print_text(s, self.color)
    R       = self.R
    E       = self.E
    T       = self.T
    W       = self.W
    eps_reg = self.eps_reg
    u       = self.u
    v       = self.v
    w       = self.w
    n       = self.n
    a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
    Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
    b       = ( E*(a_T*(1 + 181.25*W))*exp(-Q_T/(R*T)) )**(-1/n)
    term    = 0.5 * (0.5 * (u.dx(2)**2 + v.dx(2)**2 + (u.dx(1) + v.dx(0))**2) \
                     + u.dx(0)**2 + v.dx(1)**2 + (u.dx(0) + v.dx(1))**2 )
    epsdot  = term + eps_reg
    eta     = project(b * epsdot**((1-n)/(2*n)), self.Q)
    self.assign_variable(self.eta, eta)
    print_min_max(self.eta, 'eta')

  def strain_rate_tensor(self, U):
    """
    return the strain-rate tensor of <U>.
    """
    return 0.5 * (grad(U) + grad(U).T)
  
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
  
  def calc_pressure(self):
    """
    Calculate the continuous pressure field.
    """
    s = "::: calculating pressure :::"
    print_text(s, self.color)
    Q       = self.Q
    rhoi    = self.rhoi
    g       = self.g
    S       = self.S
    x       = self.x
    P       = rhoi*g*(S - x[2])
    self.P  = project(P, Q)
    #dx      = self.dx
    #dx_s    = dx(1)
    #dx_g    = dx(0)
    #dGamma  = dx_s + dx_g
    #eta_shf = project(self.eta_shf, Q)
    #eta_gnd = project(self.eta_gnd, Q)
    #w       = self.w
    #P       = TrialFunction(Q)
    #phi     = TestFunction(Q)
    #M       = assemble(phi*P*dx)
    #P_f     = + rhoi * g * (S - x[2]) * phi * dGamma \
    #          + Constant(2.0) * w.dx(2) * phi * (eta_shf*dx_s + eta_gnd*dx_g)
    #solve(M, self.P.vector(), assemble(P_f))
  
  def extrude(self, f, b, d, Q='self'):
    r"""
    This extrudes a function <f> defined along a boundary list <b> out onto
    the domain in the direction <d>.  It does this by formulating a 
    variational problem:
  
    :Conditions: 
    .. math::
    \frac{\partial u}{\partial d} = 0
    
    u|_b = f
  
    and solving.  
    
    :param f  : Dolfin function defined along a boundary
    :param b  : Boundary condition
    :param d  : Subdomain over which to perform differentiation
    """
    s = "::: extruding function :::"
    print_text(s, self.color)
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    a   = v.dx(d) * phi * dx
    L   = DOLFIN_EPS * phi * dx
    bcs = []
    if type(b) != list:
      b = [b]
    for boundary in b:
      bcs.append(DirichletBC(Q, f, ff, boundary))
    v   = Function(Q)
    solve(a == L, v, bcs)
    print_min_max(f, 'function to be extruded')
    print_min_max(v, 'extruded function')
    return v
  
  def vert_integrate(self, u, Q='self'):
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
    # integral is zero on bed (ff = 3,5) 
    bcs = []
    bcs.append(DirichletBC(Q, 0.0, ff, 3))
    if self.mask != None:
      bcs.append(DirichletBC(Q, 0.0, ff, 5))
    a      = v.dx(2) * phi * dx
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
    uhat = self.vert_integrate(u)
    s = "::: calculating vertical average :::"
    print_text(s, self.color)
    ubar = project(uhat/H, self.Q)
    print_min_max(ubar, 'ubar')
    ubar = self.extrude(ubar, [2,6], 2)
    return ubar

  def calc_misfit(self, integral):
    """
    Calculates the misfit of model and observations, 

      D = ||U - U_ob||

    and updates model.misfit with D.
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
      norm_u = np.sqrt(sum(U_v**2))
    elif type == 'linf':
      norm_u = np.max(U_v)
    
    return U_v, norm_u

  def normalize_vector(self, U, Q='self'):
    """
    Create a normalized vector of the UFL vector <U>.
    """
    if type(Q) != FunctionSpace:
      Q = self.Q

    U_v, norm_u = self.get_norm(U)
    
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

  def calc_component_stress(self, u_dir, Q='self'):
    """
    Calculate the deviatoric component of stress in the direction of 
    the UFL vector <u_dir>.
    """
    s = "::: calculating component stress :::"
    print_text(s, self.color)
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff     = self.ff                           # facet function for boundaries
    #sig    = self.calc_tau()                   # deviatoric stress tensor
    sig    = self.calc_R()                     # resistive stress tensor
    com    = dot(sig, u_dir)                   # component of stress in u-dir.
    com_n  = project(sqrt(inner(com, com)),Q)  # magnitude of com
    #u      = u_dir[0]
    #v      = u_dir[1]
    #w      = u_dir[2]
    #theta  = atan(u/v)
    #com    = self.rotate(sig, theta)
    #com_n  = com[0,0]
    phi    = TestFunction(Q)                   # test function
    v      = TrialFunction(Q)                  # trial function
    bc     = DirichletBC(Q, 0.0, ff, 3)        # boundary condition
    a      = v.dx(2) * phi * dx                # bilinear part
    L      = com_n * phi * dx                  # linear part
    v      = Function(Q)                       # solution function
    solve(a == L, v, bc)                       # solve
    v      = self.extrude(v, 2, 2)             # extrude the integral
    dvdx   = grad(v)                           # spatial derivative
    dvdu   = dot(dvdx, u_dir)                  # projection of dvdx onto dir
    return dvdu
  
  def calc_component_stress_c(self, u_dir, Q='self'):
    """
    Calculate the deviatoric component of stress in the direction of U.
    """
    s = "::: calculating component stress :::"
    print_text(s, self.color)
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff     = self.ff                           # facet function for boundaries
    #sig    = self.calc_tau()                   # deviatoric stress tensor
    sig    = self.calc_R()                     # resistive stress tensor
    x      = u_dir[0]                          # first component of sig
    y      = u_dir[1]                          # second component of sig
    com    = sig[x,y]                          # component of stress
    phi    = TestFunction(Q)                   # test function
    v      = TrialFunction(Q)                  # trial function
    bc     = DirichletBC(Q, 0.0, ff, 3)        # boundary condition
    a      = v.dx(2) * phi * dx                # bilinear part
    L      = com * phi * dx                    # linear part
    v      = Function(Q)                       # solution function
    solve(a == L, v, bc)                       # solve
    v      = self.extrude(v, 2, 2)             # extrude the integral
    dvdx   = v.dx(y)                           # derivative w.r.t. 2nd comp.
    return dvdx

  def calc_tau_bas(self, Q='self'):
    """
    """
    s = "::: calculating tau_bas :::"
    print_text(s, self.color)
    if type(Q) != FunctionSpace:
      Q = self.Q
    beta  = self.beta
    u     = self.u
    v     = self.v
    w     = self.w
    H     = self.S - self.B
  
    beta_e  = self.extrude(beta, 3, 2, Q)
    u_bas_e = self.extrude(u,    3, 2, Q)
    v_bas_e = self.extrude(v,    3, 2, Q)
    w_bas_e = self.extrude(w,    3, 2, Q)

    tau_bas_u = project(beta_e*H*u_bas_e, Q)
    tau_bas_v = project(beta_e*H*v_bas_e, Q)
    tau_bas_w = project(beta_e*H*w_bas_e, Q)

    return as_vector([tau_bas_u, tau_bas_v, tau_bas_w]) 

  def calc_tau_drv(self, Q='self'):
    """
    """
    s = "::: calculating tau_drv :::"
    print_text(s, self.color)
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff    = self.ff
    S     = self.S
    B     = self.B
    rhoi  = self.rhoi
    g     = self.g
    H     = S - B
    gradS = grad(S)
    
    gradS_u = gradS[0]
    gradS_v = gradS[1]
    gradS_w = gradS[2]
  
    tau_drv_u = project(rhoi*g*H*gradS_u, Q)
    tau_drv_v = project(rhoi*g*H*gradS_v, Q)
    tau_drv_w = project(rhoi*g*H*gradS_w, Q)

    return as_vector([tau_drv_u, tau_drv_v, tau_drv_w])

  def component_stress(self):
    """
    Calculate each of the component stresses which define the full stress
    of the ice-sheet.
    
    RETURNS:
      tau_lon - longitudinal stress field
      tau_lat - lateral stress field
      tau_vrt - vertical stress field
      tau_bas - frictional sliding stress at the bed
      tau_drv - driving stress of the system 
    
    Note: tau_drv = tau_lon + tau_lat + tau_bas
    
    """
    s = "::: calculating 'stress-balance' :::"
    print_text(s, self.color)
    Q       = self.Q
    u       = self.u
    v       = self.v
    w       = self.w
    
    ## normailze the vector :
    #U_n     = self.normalize_vector(as_vector([u,v]))    
    #u_n     = U_n[0]
    #v_n     = U_n[1]
    #
    ## unit-vectors along (n) and across (t) flow :
    #U_n = as_vector([u_n, v_n, 0])
    #U_t = as_vector([v_n,-u_n, 0])
    # 
    ## calculate components :
    #tau_lon   = project(self.calc_component_stress(U_n))
    #tau_lat   = project(self.calc_component_stress(U_t))
    #tau_bas   = self.calc_tau_bas()
    #tau_drv   = self.calc_tau_drv()

    ## calculate the component of driving stress and basal drag along flow (n) :
    #tau_bas_n = project(dot(tau_bas, U_n))
    #tau_drv_n = project(dot(tau_drv, U_n))
    
    # calculate components :
    tau_lon   = project(self.calc_component_stress_c([0,0]))
    tau_lat   = project(self.calc_component_stress_c([0,1]))
    tau_bas   = self.calc_tau_bas()
    tau_drv   = self.calc_tau_drv()

    # calculate the component of driving stress and basal drag along flow (n) :
    tau_bas_n = tau_bas[0]
    tau_drv_n = tau_drv[0]
    
    # write them to the specified directory :
    File(out_dir + 'tau_drv_s.pvd') << tau_drv_n
    File(out_dir + 'tau_bas_s.pvd') << tau_bas_n
    File(out_dir + 'tau_lon_s.pvd') << tau_lon
    File(out_dir + 'tau_lat_s.pvd') << tau_lat
 
    # return the values for further analysis :
    return tau_lon, tau_lat, tau_bas_n, tau_drv_n

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
    
    elif isinstance(var, Expression) or isinstance(var, Constant) \
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

  def init_viscosity_mode(self):
    """
    """
    s       = "::: initializing viscosity :::"
    print_text(s, self.color)
    config = self.config

    # Set the value of b, the temperature dependent ice hardness parameter,
    # using the most recently calculated temperature field, if expected.
    if   config['velocity']['viscosity_mode'] == 'isothermal':
      A0 = config['velocity']['A']
      s  = "    - using isothermal visosity formulation -"
      print_text(s, self.color)
      print_min_max(A0, 'A')
      n     = self.n
      b     = A0**(-1/n)
      b_shf = b
      b_gnd = b
      print_min_max(b_shf, 'b_shf')
      print_min_max(b_gnd, 'b_gnd')
    
    elif config['velocity']['viscosity_mode'] == 'linear':
      b_gnd = config['velocity']['eta_gnd']
      b_shf = config['velocity']['eta_shf']
      s     = "    - using linear viscosity formulation -"
      print_text(s, self.color)
      print_min_max(b_shf, 'eta_shf')
      print_min_max(b_gnd, 'eta_gnd')
      self.n  = 1.0
    
    elif config['velocity']['viscosity_mode'] == 'b_control':
      b_shf   = config['velocity']['b_shf']
      b_gnd   = config['velocity']['b_gnd']
      s       = "    - using b_control viscosity formulation -"
      print_text(s, self.color)
      print_min_max(b_shf, 'b_shf')
      print_min_max(b_gnd, 'b_gnd')
    
    elif config['velocity']['viscosity_mode'] == 'constant_b':
      b     = config['velocity']['b']
      b_shf = b
      b_gnd = b
      s = "    - using constant_b viscosity formulation -"
      print_text(s, self.color)
      print_min_max(b, 'b')
    
    elif config['velocity']['viscosity_mode'] == 'full':
      s     = "    - using full viscosity formulation -"
      print_text(s, self.color)
      T     = self.T
      W     = self.W
      R     = self.R
      n     = self.n
      E_shf = self.E_shf
      E_gnd = self.E_gnd
      a_T   = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T   = conditional( lt(T, 263.15), 6e4,          13.9e4)
      b_shf = ( E_shf*(a_T*(1 + 181.25*W))*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd = ( E_gnd*(a_T*(1 + 181.25*W))*exp(-Q_T/(R*T)) )**(-1/n)
    
    elif config['velocity']['viscosity_mode'] == 'E_control':
      E_shf = config['velocity']['E_shf'] 
      E_gnd = config['velocity']['E_gnd']
      s     = "    - using E_control viscosity formulation -"
      print_text(s, self.color)
      print_min_max(E_shf, 'E_shf')
      print_min_max(E_gnd, 'E_gnd')
      T     = self.T
      W     = self.W
      R     = self.R
      n     = self.n
      E     = self.E
      a_T   = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T   = conditional( lt(T, 263.15), 6e4,          13.9e4)
      b_shf = ( E_shf*(a_T*(1 + 181.25*W))*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd = ( E_gnd*(a_T*(1 + 181.25*W))*exp(-Q_T/(R*T)) )**(-1/n)
      self.assign_variable(self.E_shf, E_shf)
      self.assign_variable(self.E_gnd, E_gnd)
    
    else:
      s = "    - ACCEPTABLE CHOICES FOR 'viscosity_mode' ARE 'linear', " + \
          "'isothermal', 'b_control', 'constant_b', 'E_control', OR 'full' -"
      print_text(s, 'red', 1)
      sys.exit(1)
    
    # Glen's flow law :
    n       = self.n
    epsdot  = self.epsdot
    eps_reg = self.eps_reg
    eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
    eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))

    # initialize the viscosity parameters :
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
    self.b_shf   = b_shf
    self.b_gnd   = b_gnd

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
    
  def init_dukowicz_BP_variables(self):
    """
    """
    self.U      = Function(self.Q2)
    u,v         = self.U
    self.epsdot = 0.5 * (+ 0.5 * (+ u.dx(2)**2 + v.dx(2)**2 \
                                  + (u.dx(1) + v.dx(0))**2) \
                         + u.dx(0)**2 + v.dx(1)**2 \
                         + (u.dx(0) + v.dx(1))**2 )
    self.init_higher_order_variables()
    self.init_viscosity_mode()

  def init_BP_variables(self):
    """
    """
    self.U   = Function(self.Q2)
    
    # Second invariant of the strain rate tensor squared
    epi   = self.strain_rate_tensor(as_vector([self.U[0], self.U[1], 0.0]))
    ep_xx = epi[0,0]
    ep_yy = epi[1,1]
    ep_xy = epi[0,1]
    ep_xz = epi[0,2]
    ep_yz = epi[1,2]
    
    self.epsdot = + ep_xx**2 + ep_yy**2 + ep_xx*ep_yy \
                  + ep_xy**2 + ep_xz**2 + ep_yz**2
    self.init_higher_order_variables()
    self.init_viscosity_mode()

  def init_dukowicz_stokes_variables(self):
    """
    """
    self.U   = Function(self.Q4)
    u,v,w,P  = self.U
    
    # Second invariant of the strain rate tensor squared
    self.epsdot  = + 0.5 * (+ 0.5*( + (u.dx(1) + v.dx(0))**2  \
                                    + (u.dx(2) + w.dx(0))**2  \
                                    + (v.dx(2) + w.dx(1))**2) \
                            + u.dx(0)**2 + v.dx(1)**2 + w.dx(2)**2) 
    self.init_higher_order_variables()
    self.init_viscosity_mode()

  def init_stokes_variables(self):
    """
    """
    # velocity :
    self.U   = Function(self.Q3)
    
    # Second invariant of the strain rate tensor squared
    epi   = self.strain_rate_tensor(self.U)
    ep_xx = epi[0,0]
    ep_yy = epi[1,1]
    ep_xy = epi[0,1]
    ep_xz = epi[0,2]
    ep_yz = epi[1,2]
    
    self.epsdot = + ep_xx**2 + ep_yy**2 + ep_xx*ep_yy \
                  + ep_xy**2 + ep_xz**2 + ep_yz**2
    self.init_higher_order_variables()
    self.init_viscosity_mode()

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
    
    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = min(0, x[2])
    self.D = Depth(element=self.Q.ufl_element())

    if   config['model_order'] == 'stokes':
      if config['use_dukowicz']:
        self.init_dukowicz_stokes_variables()
      else:
        self.init_stokes_variables()
    elif config['model_order'] == 'BP':
      if config['use_dukowicz']:
        self.init_dukowicz_BP_variables()
      else:
        self.init_BP_variables()
    elif config['model_order'] == 'L1L2':
      self.init_hybrid_variables()
    else:
      s = "    - PLEASE SPECIFY A MODEL ORDER; MAY BE 'stokes', 'BP', " + \
          "or 'L1L2' -"
      print_text(s, 'red', 1)
      sys.exit(1)

    # Velocity model
    self.u             = Function(self.Q)
    self.v             = Function(self.Q)
    self.w             = Function(self.Q)
    self.P             = Function(self.Q)
    self.beta          = Function(self.Q)
    self.mhat          = Function(self.Q)
    self.E_gnd         = Function(self.Q)
    self.E_shf         = Function(self.Q)
    self.eta           = Function(self.Q)
    self.u_ob          = Function(self.Q)
    self.v_ob          = Function(self.Q)
    self.U_ob          = Function(self.Q)
    
    # Enthalpy model
    self.theta_surface = Function(self.Q)
    self.theta_float   = Function(self.Q)
    self.theta         = Function(self.Q)
    self.T             = Function(self.Q)
    self.q_geo         = Function(self.Q)
    self.W0            = Function(self.Q)
    self.W             = Function(self.Q)
    self.W_r           = Function(self.Q)
    self.Mb            = Function(self.Q)
    self.thetahat      = Function(self.Q) # Midpoint values
    self.uhat          = Function(self.Q) # Midpoint values
    self.vhat          = Function(self.Q) # Midpoint values
    self.what          = Function(self.Q) # Midpoint values
    self.mhat          = Function(self.Q) # ALE is required: we change the mesh 
    self.theta0        = Function(self.Q) # initial enthalpy
    self.T0            = Function(self.Q) # pressure-melting point
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
    self.lam           = Function(self.Q)
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
    self.v_s           = Function(self.Q)
    self.tau_dn        = Function(self.Q)
    self.tau_dt        = Function(self.Q)
    self.tau_bn        = Function(self.Q)
    self.tau_bt        = Function(self.Q)
    self.tau_pn        = Function(self.Q)
    self.tau_pt        = Function(self.Q)
    self.tau_nn        = Function(self.Q)
    self.tau_nt        = Function(self.Q)
    self.tau_nz        = Function(self.Q)
    self.tau_tn        = Function(self.Q)
    self.tau_tt        = Function(self.Q)
    self.tau_tz        = Function(self.Q)
    



