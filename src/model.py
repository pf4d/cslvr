from fenics       import *
from ufl.indexed  import Indexed
from abc          import ABCMeta, abstractmethod
from physics      import Physics
from solvers      import Solver
from helper       import default_config
from io           import print_text, print_min_max
import numpy              as np
import physical_constants as pc

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
    self.Q_flat = FunctionSpace(self.flat_mesh, "CG", 1, 
                                constrained_domain=self.pBC)
    self.Q2     = MixedFunctionSpace([self.Q]*2)
    self.Q3     = MixedFunctionSpace([self.Q]*3)
    self.Q4     = MixedFunctionSpace([self.Q]*4)
    self.V              = VectorFunctionSpace(self.mesh, "CG", 1)
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
    print_min_max(self.u_ob, 'u_ob')
    print_min_max(self.v_ob, 'v_ob')
  
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
      
  def init_beta_SIA(self, U_mag=None):
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
    if U_mag == None:
      U_mag    = Function(Q)
      u_v      = self.u_ob.vector().array()
      v_v      = self.v_ob.vector().array()
      U_mag_v  = np.sqrt(u_v**2 + v_v**2 + 1e-16)
    else:
      U_mag_v = U_mag.vector().array()
    U_mag_v[U_mag_v < 0.5] = 0.5
    self.assign_variable(U_mag, U_mag_v)
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta_0   = project(sqrt((rhoi*g*H*S_mag) / (H**r * U_mag)), Q)
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    self.assign_variable(self.beta, beta_0_v)
    print_min_max(self.beta, 'beta')
    self.betaSIA = Function(Q)
    self.assign_variable(self.betaSIA, self.beta)

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

    U_v  = as_vector([self.u, self.v, self.w])

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

    # experimental antarctica :
    bhat = [-1.79050733e+02,   1.21225357e-02,   7.00242275e-01,
            -1.79944157e+02,  -3.92007119e-03,   2.41824847e+02,
             2.68360346e-03,   1.08996135e+01,  -3.30190367e+00,
             6.99415928e-01,   1.56903065e+00,  -8.76360367e+00,
            -5.89194455e-05,  -6.83779816e-03,   2.87684715e-07,
            -1.52889892e-04,  -5.08241294e-07,  -3.51334303e-04,
             6.35242700e-05,   1.12753024e-05,   3.17590324e-05,
             2.44610103e-04,   4.13158937e-01,  -1.60928114e-05,
            -4.47594232e-01,   3.68525379e-05,  -3.13299967e-01,
             1.81440872e-02,  -2.55267424e-03,  -4.30367797e-03,
             2.43652633e-02,   6.78009294e-03,  -1.19606688e+01,
             1.18463541e-02,  -1.51293227e+01,  -2.70661398e+00,
             3.38092068e-01,  -6.38959136e-01,   2.04002460e+00,
            -1.98509132e-03,   1.70764830e-07,   1.42846441e-03,
            -1.98426159e-04,   2.74015031e-05,  -7.76148708e-06,
            -1.20679117e-04,  -1.88107778e-03,   9.57136085e+00,
             2.12328719e+00,  -4.88099394e-01,  -1.93124549e-01,
            -6.62792869e-01,  -1.52428993e-03,   4.51663540e-04,
            -4.39776697e-05,  -5.94617368e-06,  -1.66066895e-04,
            -4.93024203e-01,   2.72992966e-01,  -5.17802786e-02,
            -2.03153825e-01,  -6.92688954e-03,   3.72880798e-02,
             5.69350346e-02,  -2.38251991e-03,   3.98819107e-03,
             9.14996175e-02]

    ## antarctica dependent included :
    #bhat = [-3.93728495e+01,   1.81338348e-03,   1.61581323e-01,
    #        -3.95121988e+01,  -1.75721467e-04,   4.16637532e+01,
    #         4.66544798e-04,  -2.73444006e+00,  -1.92848011e-01,
    #         1.65122061e-01,   3.02294645e-01,   2.11948400e-01,
    #        -8.69054908e-06,  -4.56845894e-05,   6.42972736e-08,
    #         4.38644481e-04,  -6.75628647e-08,   1.77803620e-04,
    #         3.87533437e-05,   1.73263144e-06,   4.60008734e-06,
    #        -7.08079609e-05,   1.04215727e-01,  -5.85880620e-06,
    #        -8.53529040e-02,   8.81023749e-06,  -2.73278139e-02,
    #         2.79539490e-03,  -5.81088051e-04,  -1.41773851e-03,
    #        -2.50101528e-03,   1.87089440e-04,  -2.51548776e+00,
    #         1.47017518e-03,  -3.06533364e+00,  -3.52882440e-01,
    #         6.01730826e-02,  -1.01619720e-01,   1.89035666e-01,
    #        -7.71542575e-04,   7.69699517e-10,   7.03417523e-05,
    #        -5.68891325e-05,   5.61122626e-06,  -2.53577917e-06,
    #         6.09405255e-05,  -2.71849811e-04,   1.70727899e+00,
    #         1.69171945e-01,  -7.38307759e-02,  -5.23143884e-03,
    #        -1.65563468e-01,  -2.15920765e-04,   4.65834091e-05,
    #        -9.81236389e-06,  -6.45447501e-06,  -1.35248450e-05,
    #        -3.13812716e-02,   3.65226182e-02,   1.99641424e-02,
    #         6.68210962e-02,  -2.56907104e-03,   5.12075444e-03,
    #         4.02576642e-02,   2.80501545e-04,   1.47904749e-04,
    #         3.20884773e-03]

    ## antarctica independent only :
    #bhat = [-3.81730613e-01,   1.44979590e-03,   2.30473800e-02,
    #        -4.66557207e+01,  -2.70605024e-04,   2.96236959e+01,
    #         1.35282874e-04,   4.98572401e+00,   5.76099557e-01,
    #        -4.09000148e-06,  -7.66828817e-04,   4.53570633e-08,
    #         3.86192314e-04,  -1.22965666e-07,   3.73291874e-04,
    #        -7.59612262e-05,   1.96585780e-01,  -9.01208788e-07,
    #        -1.17224924e-01,  -1.16892612e-07,  -2.12477085e-02,
    #        -2.63217268e-03,   3.56343313e-04,  -1.68509204e+00,
    #         2.52093755e-03,  -2.93492954e+00,  -3.44265519e-01,
    #        -8.35448819e-05,   1.00058225e-07,  -2.51673071e-04,
    #         4.08917599e-05,  -6.09966964e-04,   1.27385446e-01,
    #         2.51091290e-01,   1.81664750e-05,  -1.77584371e-05,
    #         7.87445598e-02]
   
    ## greenland dependent included :
    #bhat = [-9.39465489e+00,   2.19530652e-03,   4.28985887e-02,
    #         3.18782474e+00,  -2.34050602e-03,   3.73032420e+00,
    #        -1.22317169e-03,   1.28776686e+00,   1.01060056e+00,
    #         6.60535868e-02,   6.69064894e+00,  -1.14935616e+00,
    #        -1.08320681e-05,  -3.20529924e-04,   1.75510050e-08,
    #         1.91612174e-04,  -7.10957013e-09,  -4.26885238e-06,
    #         8.88992656e-05,   1.53702105e-06,   1.91016413e-04,
    #        -6.46213366e-05,  -6.44455268e-03,   1.72392535e-06,
    #         1.42759432e-02,   1.19588342e-05,  -6.05947661e-03,
    #         2.23623001e-04,  -1.96745101e-04,  -1.47302440e-02,
    #         1.24823873e-03,   2.98046284e-04,  -1.17614093e+00,
    #         1.13790819e-04,   5.46704392e-02,   4.35791870e-02,
    #        -3.43609371e-03,  -2.83742011e-01,  -6.35289747e-02,
    #         1.86851654e-05,  -1.43078724e-08,   3.22580805e-05,
    #        -8.28082876e-05,   8.55215944e-06,  -1.53823948e-04,
    #        -9.44222140e-06,  -8.43205343e-05,  -3.75357711e-02,
    #        -1.08061501e-01,  -2.59821405e-02,  -5.01141501e-01,
    #         3.27036521e-02,   6.40989073e-06,  -1.89547330e-05,
    #        -6.45132233e-06,   7.27148472e-06,   6.93137619e-06,
    #         9.46724452e-04,   1.28674541e-03,   1.61861928e-02,
    #        -1.19089114e-02,  -4.33623624e-03,  -3.28422430e-02,
    #         1.96250108e-02,  -9.23730933e-03,   2.38812961e-03,
    #        -9.46730935e-03] 
    
    ## greenland independent only :
    #bhat = [ 1.05722099e+01,  -3.14375789e-03,  -2.08948258e-02,
    #        -1.33363418e+00,   5.79102422e-03,  -5.25066003e+00,
    #         1.88843384e-03,  -5.17373398e-01,  -1.37549574e+00,
    #         1.36776450e-05,  -3.28684159e-04,  -3.70461389e-10,
    #        -3.59714356e-04,  -1.22610525e-08,  -8.41035670e-05,
    #        -3.35163941e-05,   8.31556947e-03,  -2.35148210e-05,
    #         1.69045150e-02,  -8.26475722e-06,   2.33510653e-03,
    #         5.06403921e-03,   5.21793194e-04,  -1.07521646e+00,
    #        -4.73340816e-06,   3.25557416e-02,  -3.48467832e-02,
    #         7.52948616e-04,  -6.03040655e-08,   1.42208676e-04,
    #         3.01671475e-05,   3.60847814e-04,  -1.55640821e-01,
    #         1.67986111e-01,   1.40480808e-04,   1.02754750e-05,
    #        -2.32648719e-02]
    
    X_i  = []
    X_i.extend(X)
     
    for i,xx in enumerate(X):
      for yy in X[i+1:]:
        X_i.append(xx*yy)
    
    self.beta_f = Constant(bhat[0])
    
    for xx,bb in zip(X_i, bhat[1:]):
      self.beta_f += Constant(bb)*xx
    #self.beta_f = exp(self.beta_f) - Constant(100.0)
    self.beta_f = exp(self.beta_f) - Constant(1.0)
    
    beta                    = project(self.beta_f, Q)
    beta_v                  = beta.vector().array()
    beta_v[beta_v < 0.0]    = 0.0
    self.assign_variable(self.beta, np.sqrt(beta_v))
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

  def BP_strain_rate(self,U):
    """
    return the strain-rate tensor of <U>.
    """
    u,v,w = split(U)
    epi   = 0.5 * (grad(U) + grad(U).T)
    epi02 = 0.5*u.dx(2)
    epi12 = 0.5*v.dx(2)
    epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02   ],
                        [epi[1,0],  epi[1,1],  epi12   ],
                        [epi02,     epi12,     epi[2,2]]])
    return epsdot
  
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

  def get_norm(self, U):
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
    norm_u = np.sqrt(sum(U_v**2))
    
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
           " int, \nVector, Expression, or string path to .xml, \n" + \
           "not \n%s\n" + \
           "*************************************************************"
      print_text(s % type(var) , 'red')
      exit(1)

  def globalize_parameters(self, namespace=None):
    """
    This function converts the parameter dictinary into global object
    
    :param namespace: Optional namespace in which to place the global variables
    """
    for v in self.variables.iteritems():
      vars(namespace)[v[0]] = v[1]

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    s = "::: initializing variables :::"
    print_text(s, self.color)
    
    self.set_parameters(pc.IceParameters())
    self.params.globalize_parameters(self) # make all the variables available 

    # Coordinates of various types 
    self.x             = SpatialCoordinate(self.mesh)
    self.sigma         = project((self.x[2] - self.B) / (self.S - self.B))
    self.gradS         = project(grad(self.S), self.V)
    self.gradB         = project(grad(self.B), self.V)

    # Velocity model
    self.u             = Function(self.Q)
    self.v             = Function(self.Q)
    self.w             = Function(self.Q)
    self.P             = Function(self.Q)
    self.beta          = Function(self.Q)
    self.mhat          = Function(self.Q)
    self.b             = Function(self.Q)
    self.b_shf         = Function(self.Q)
    self.b_gnd         = Function(self.Q)
    self.epsdot        = Function(self.Q)
    self.E             = Function(self.Q)
    self.E_gnd         = Function(self.Q)
    self.E_shf         = Function(self.Q)
    self.eta_gnd       = Function(self.Q)
    self.eta_shf       = Function(self.Q)
    self.eta           = Function(self.Q)
    self.u_ob          = Function(self.Q)
    self.v_ob          = Function(self.Q)
    self.U_ob          = Function(self.Q)
    
    # Enthalpy model
    self.H_surface     = Function(self.Q)
    self.H_float       = Function(self.Q)
    self.H             = Function(self.Q)
    self.T             = Function(self.Q)
    self.q_geo         = Function(self.Q)
    self.W0            = Function(self.Q)
    self.W             = Function(self.Q)
    self.W_r           = Function(self.Q)
    self.Mb            = Function(self.Q)
    self.Hhat          = Function(self.Q) # Midpoint values
    self.uhat          = Function(self.Q) # Midpoint values
    self.vhat          = Function(self.Q) # Midpoint values
    self.what          = Function(self.Q) # Midpoint values
    self.mhat          = Function(self.Q) # ALE is required: we change the mesh 
    self.H0            = Function(self.Q) # initial enthalpy
    self.T0            = Function(self.Q) # pressure-melting point
    self.kappa         = Function(self.Q)
    self.Kcoef         = Function(self.Q)

    # free surface model :
    self.Shat          = Function(self.Q_flat)
    self.dSdt          = Function(self.Q_flat)
    self.ahat          = Function(self.Q_flat)
    self.uhat_f        = Function(self.Q_flat)
    self.vhat_f        = Function(self.Q_flat)
    self.what_f        = Function(self.Q_flat)
    self.M             = Function(self.Q_flat)
    
    # Age model   
    self.age           = Function(self.Q)
    self.a0            = Function(self.Q)

    # Surface climate model
    self.smb           = Function(self.Q)
    self.precip        = Function(self.Q)
    self.T_surface     = Function(self.Q)

    # Adjoint model
    self.u_o           = Function(self.Q)
    self.v_o           = Function(self.Q)
    self.lam           = Function(self.Q)
    self.adot          = Function(self.Q)

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
    



