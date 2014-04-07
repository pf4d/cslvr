from dolfin import *

class Model(object):
  """ 
  Instance of a 2D flowline ice model that contains geometric and scalar 
  parameters and supporting functions.  This class does not contain actual 
  physics but rather the interface to use physics in different simulation 
  types.
  """

  def __init__(self):
    self.per_func_space = False  # function space is undefined

  def set_geometry(self, sur, bed, mask=None):
    """
    Sets the geometry of the surface and bed of the ice sheet.
    
    :param sur  : Expression representing the surface of the mesh
    :param bed  : Expression representing the base of the mesh
    :param mask : Expression representing a mask of grounded (0) and floating 
                  (1) areas of the ice.
    """
    self.S_ex = sur
    self.B_ex = bed
    self.mask = mask
  
  def generate_uniform_mesh(self, nx, ny, nz, xmin, xmax, 
                            ymin, ymax, generate_pbcs=False,deform=True):
    """
    Generates a uniformly spaced 3D Dolfin mesh with optional periodic boundary 
    conditions
    
    :param nx                 : Number of x cells
    :param ny                 : Number of y cells
    :param nz                 : Number of z cells
    :param xmin               : Minimum x value of the mesh
    :param xmax               : Maximum x value of the mesh
    :param ymin               : Minimum y value of the mesh
    :param ymax               : Maximum y value of the mesh
    :param bool generate_pbcs : Optional argument to determine whether
                                to create periodic boundary conditions
    """
    print "::: generating mesh :::"

    self.mesh      = UnitCubeMesh(nx,ny,nz)
    self.flat_mesh = UnitCubeMesh(nx,ny,nz)
    
    # generate periodic boundary conditions if required :
    if generate_pbcs:
      class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
          return bool((near(x[0], 0) or near(x[1], 0)) and (not ((near(x[0], 0) and near(x[1], 1)) or (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

        def map(self, x, y):
          if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
            y[2] = x[2]
          elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
            y[2] = x[2]
          else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.
            y[2] = x[2]
      pBC       = PeriodicBoundary()
      self.Q         = FunctionSpace(self.mesh, "CG", 1, 
                                     constrained_domain = pBC)
      self.Q_non_periodic = FunctionSpace(self.mesh, "CG", 1)
      self.Q_flat    = FunctionSpace(self.flat_mesh, "CG", 1, 
                                     constrained_domain = pBC)
      self.Q_flat_non_periodic = FunctionSpace(self.flat_mesh,"CG",1)
      self.Q2        = MixedFunctionSpace([self.Q]*2)
      self.Q4        = MixedFunctionSpace([self.Q]*4)
      self.per_func_space = True

    # width and origin of the domain for deforming x coord :
    width_x  = xmax - xmin
    offset_x = xmin
    
    # width and origin of the domain for deforming y coord :
    width_y  = ymax - ymin
    offset_y = ymin

    if deform:
    # Deform the square to the defined geometry :
      for x,x0 in zip(self.mesh.coordinates(), self.flat_mesh.coordinates()):
        # transform x :
        x[0]  = x[0]  * width_x + offset_x
        x0[0] = x0[0] * width_x + offset_x
      
        # transform y :
        x[1]  = x[1]  * width_y + offset_y
        x0[1] = x0[1] * width_y + offset_y
    
        # transform z :
        # thickness = surface - base, z = thickness + base
        x[2]  = x[2] * (self.S_ex(x[0], x[1], x[2]) - self.B_ex(x[0], x[1], x[2]))
        x[2]  = x[2] + self.B_ex(x[0], x[1], x[2])

  def set_mesh(self, mesh, flat_mesh=None, deform=True):
    """
    Overwrites the previous mesh with a new one
    
    :param mesh        : Dolfin mesh to be written
    :param flat_mesh   : Dolfin flat mesh to be written
    :param bool deform : If True, deform the mesh to surface and bed data 
                         provided by the set_geometry method.
    """
    self.mesh      = mesh
    self.flat_mesh = flat_mesh

    if deform:
      self.deform_mesh()

  def deform_mesh(self):
    """
    Deforms the mesh to the geometry.
    """
    # transform z :
    # thickness = surface - base, z = thickness + base
    for x in self.mesh.coordinates():
      x[2] = x[2] * ( self.S_ex(x[0],x[1],x[2]) - self.B_ex(x[0],x[1],x[2]) )
      x[2] = x[2] + self.B_ex(x[0], x[1], x[2])

  def calculate_boundaries(self):
    """
    Determines the boundaries of the current model mesh
    """
    print "::: calculating boundaries :::"
    
    mask = self.mask

    # this function contains markers which may be applied to facets of the mesh
    self.ff   = FacetFunction('size_t', self.mesh, 0)
    
    # iterate through the facets and mark each if on a boundary :
    #
    #   2 = high slope, upward facing ................ surface
    #   3 = high slope, downward facing .............. base
    #   4 = low slope, upward or downward facing ..... sides
    #   5 = floating ................................. base
    #   6 = floating ................................. sides
    if mask != None:
      for f in facets(self.mesh):
        n       = f.normal()    # unit normal vector to facet f
        tol     = 1e-3
        mask_xy = mask(x_m, y_m)
      
        if   n.z() >=  tol and f.exterior():
          self.ff[f] = 2
      
        elif n.z() <= -tol and f.exterior():
          if mask_xy > 0:
            self.ff[f] = 5
          else:
            self.ff[f] = 3
      
        elif n.z() >  -tol and n.z() < tol and f.exterior():
          if mask_xy > 0:
            self.ff[f] = 6
          else:
            self.ff[f] = 4

    # iterate through the facets and mark each if on a boundary :
    #
    #   2 = high slope, upward facing ................ surface
    #   3 = high slope, downward facing .............. base
    #   4 = low slope, upward or downward facing ..... sides
    else:
      for f in facets(self.mesh):
        n       = f.normal()    # unit normal vector to facet f
        tol     = 1e-3
      
        if   n.z() >=  tol and f.exterior():
          self.ff[f] = 2
      
        elif n.z() <= -tol and f.exterior():
          self.ff[f] = 3
      
        elif n.z() >  -tol and n.z() < tol and f.exterior():
          self.ff[f] = 4
   
    self.ds = Measure('ds')[self.ff]
     
  def set_parameters(self, params):
    """
    Sets the model's dictionary of parameters
    
    :param params: :class:`~src.physical_constants.IceParameters` object 
       containing model-relavent parameters
    """
    self.params = params
  
  def extrude(self, f, b, d):
    r"""
    This extrudes a function <f> defined along a boundary <b> out onto
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
    Q   = self.Q
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    a   = v.dx(d) * phi * dx
    L   = DOLFIN_EPS * phi * dx  # really close to zero to fool FFC
    bc  = DirichletBC(Q, f, ff, b)
    v   = Function(Q)
    solve(a == L, v, bc)
    return v
  
  def calc_thickness(self):
    """
    Calculate the continuous thickness field which increases from 0 at the 
    surface to the actual thickness at the bed.
    """
    Q   = self.Q
    ff  = self.ff
    H   = TrialFunction(Q)
    phi = TestFunction(Q)
    a   = H.dx(2) * phi * dx
    L   = -1.0 * phi * dx
    bc  = DirichletBC(Q, 0.0, ff, 2)
    H   = Function(Q)
    solve(a == L, H, bc)
    return H
  
  def calc_pressure(self):
    """
    Calculate the continuous pressure field.
    """
    Q   = self.Q
    rho = self.rho
    g   = self.g
    H   = self.calc_thickness() 
    P   = rho * g * H
    return P
  
  def calc_sigma(self, U):
    """
    Calculatethe Cauchy stress tensor of velocity field <u>.
    """
    n   = U.geometric_dimension()
    P   = self.calc_pressure()
    tau = self.calc_tau(u)
    return tau - P*Identity(n)
 
  def n_d_grad(self, u):
    """
    """
    n     = u.shape()[0]
    gradu = []
    for i in range(n):
      graduRow = []
      for j in range(n):
        graduRow.append(u[i].dx(j))
      gradu.append(graduRow)
    gradu = as_matrix(gradu)
    return gradu
  
  def n_d_div(self, u):
    """
    """
    n     = u.shape()[0]
    divu  = 0.0
    for i in range(n):
      divu += u[i].dx(i)
    return divu
  
  def calc_tau(self, U):
    """
    Calculate the deviatoric stress tensor of velocity field <u>.
    """
    n     = U.geometric_dimension()
    eta   = self.eta
    gradU = nabla_grad(U)
    divU  = nabla_div(U)
    tau   = eta * (gradU + gradU.T - 2.0/n * divU * Identity(n))
    return tau
     
  def vert_integrate(self, u):
    """
    Integrate <u> from the bed to the surface.
    """
    ff     = self.ff
    Q      = self.Q
    phi    = TestFunction(Q)
    v      = TrialFunction(Q)
    bc     = DirichletBC(Q, 0.0, ff, 3)
    a      = v.dx(2) * phi * dx
    L      = u * phi * dx
    v      = Function(Q)
    solve(a == L, v, bc)
    return v
  
  def calc_component_stress(self, U, u_dir):
    """
    Calculate the deviatoric component of stress in the direction of <u>.
    """
    ff     = self.ff                           # facet function for boundaries
    Q      = self.Q                            # function space
    sig    = self.calc_tau(U)                  # deviatoric stress tensor
    com    = dot(sig, u_dir)                   # component of stress in u-dir.
    com_n  = project(sqrt(inner(com, com)),Q)  # magnitude of com
    phi    = TestFunction(Q)                   # test function
    v      = TrialFunction(Q)                  # trial function
    bc     = DirichletBC(Q, 0.0, ff, 3)        # boundary condition
    a      = v.dx(2) * phi * dx                # bilinear part
    L      = com_n * phi * dx                  # linear part
    v      = Function(Q)                       # solution function
    solve(a == L, v, bc)                       # solve
    dvdx   = grad(v)                           # spatial derivative
    dvdu   = dot(dvdx, u_dir)                  # projection of dvdx onto dir
    return project(dvdu, Q)

  def component_stress(self):
    """
    Calculate each of the component stresses which define the full stress
    of the ice-sheet.
    
    RETURNS:
      tau_lon - longitudinal stress field
      tau_lat - lateral stress field
      tau_bas - frictional sliding stress at the bed
      tau_drv - driving stress of the system 
    
    Note: tau_drv = tau_lon + tau_lat + tau_bas
    """
    beta2 = self.beta2
    eta   = self.eta
    ff    = self.ff
    Q     = self.Q
    u     = self.u
    v     = self.v
    w     = self.w
    S     = self.S
    B     = self.B
    rho   = self.rho
    g     = self.g
    H     = S - B
    zero  = Constant(0.0)
  
    u_n = as_vector([u, v, zero])
    u_t = as_vector([v,-u, zero])
    U   = as_vector([u, v, w])
    
    norm_U   = project(sqrt(inner(U, U)),     Q)
    norm_u   = project(sqrt(inner(u_n, u_n)), Q)
    gradSMag = sqrt(inner(grad(S), grad(S)))
  
    norm_u.update()                              # eliminate ghost vertices
    norm_U.update()                              # eliminate ghost vertices
  
    beta2_e = self.extrude(beta2,  3, 2)
    u_bas_e = self.extrude(norm_U, 3, 2)
 
    tau_lon = self.calc_component_stress(U, u_n/norm_u)
    tau_lat = self.calc_component_stress(U, u_t/norm_u)
    tau_bas = project(beta2_e*H*u_bas_e, Q)
    tau_drv = project(rho*g*H*gradSMag,  Q)

    tau_bas2 = project(tau_drv - tau_lon - tau_lat)
    beta22   = project(tau_bas2 / (H*u_bas_e))

    tau_lon.update()                             # eliminate ghost vertices 
    tau_lat.update()                             # eliminate ghost vertices 
  
    tau_lon = self.extrude(tau_lon, 2, 2)
    tau_lat = self.extrude(tau_lat, 2, 2)
  
    return tau_lon, tau_lat, tau_bas, tau_drv, beta22
   
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    self.params.globalize_parameters(self) # make all the variables available 
    self.calculate_boundaries()            # calculate the boundaries

    # Function Space
    if self.per_func_space == False:
      self.Q           = FunctionSpace(self.mesh,      "CG", 1)
      self.Q_flat      = FunctionSpace(self.flat_mesh, "CG", 1)
      self.Q2          = MixedFunctionSpace([self.Q]*2)
      self.Q4          = MixedFunctionSpace([self.Q]*4)
    
      # surface and bed :
      self.S           = interpolate(self.S_ex, self.Q)
      self.B           = interpolate(self.B_ex, self.Q)
      self.Shat        = Function(self.Q_flat)
      self.dSdt        = Function(self.Q_flat)
    
    else:
      # surface and bed :
      self.S           = interpolate(self.S_ex, self.Q_non_periodic)
      self.B           = interpolate(self.B_ex, self.Q_non_periodic)
      self.Shat        = Function(self.Q_flat_non_periodic)
      self.dSdt        = Function(self.Q_flat)
    
    # Coordinates of various types 
    self.x             = self.Q.cell().x
    self.sigma         = project((self.x[2] - self.B) / (self.S - self.B))

    # Velocity model
    self.U             = Function(self.Q2)
    self.u             = Function(self.Q)
    self.v             = Function(self.Q)
    self.w             = Function(self.Q)
    self.beta2         = Function(self.Q)
    self.mhat          = Function(self.Q)
    self.b             = Function(self.Q)
    self.epsdot        = Function(self.Q)
    self.E             = Function(self.Q)
    self.eta           = Function(self.Q)
    self.P             = Function(self.Q)
    self.Tstar         = Function(self.Q) # None
    self.W             = Function(self.Q) # None 
    self.Vd            = Function(self.Q) # None 
    self.Pe            = Function(self.Q) # None 
    self.Sl            = Function(self.Q) # None 
    self.Pc            = Function(self.Q) # None
    self.Nc            = Function(self.Q) # None
    self.Pb            = Function(self.Q)
    self.Lsq           = Function(self.Q)
    
    # Enthalpy model
    self.H_surface     = Function(self.Q)
    self.H             = Function(self.Q)
    self.T             = Function(self.Q)
    self.W             = Function(self.Q)
    self.Mb            = Function(self.Q)
    self.q_geo         = Function(self.Q)
    self.cold          = Function(self.Q)
    self.Hhat          = Function(self.Q) # Midpoint values, usually set to H_n
    self.uhat          = Function(self.Q) # Midpoint values, usually set to H_n
    self.vhat          = Function(self.Q) # Midpoint values, usually set to H_n
    self.what          = Function(self.Q) # Midpoint values, usually set to H_n
    self.mhat          = Function(self.Q) # ALE is required: we change the mesh 
    self.H0            = Function(self.Q) # None initial enthalpy
    self.T0            = Function(self.Q) # None
    self.h_i           = Function(self.Q) # None
    self.kappa         = Function(self.Q) # None

    # free surface model :
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
    self.U_o           = Function(self.Q)
    self.lam           = Function(self.Q)
    self.adot          = Function(self.Q)

    # Balance Velocity model :
    self.dSdx          = Function(self.Q_flat)
    self.dSdy          = Function(self.Q_flat)
    self.Ub            = Function(self.Q_flat)
    self.u_balance     = Function(self.Q)
    self.v_balance     = Function(self.Q)



