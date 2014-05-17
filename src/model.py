from dolfin import *
import numpy as np

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
          # return True if on left or bottom boundary AND NOT on one 
          # of the two corners (0, 1) and (1, 0)
          return bool((near(x[0], 0) or near(x[1], 0)) and \
                      (not ((near(x[0], 0) and near(x[1], 1)) \
                       or (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

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
      pBC      = PeriodicBoundary()
      self.Q   = FunctionSpace(self.mesh, "CG", 1, constrained_domain=pBC)
      self.Q2  = MixedFunctionSpace([self.Q]*2)
      self.Q4  = MixedFunctionSpace([self.Q]*4)
      
      self.Q_non_periodic = FunctionSpace(self.mesh, "CG", 1)
      self.Q_flat         = FunctionSpace(self.flat_mesh, "CG", 1, 
                                          constrained_domain=pBC)
      self.Q_flat_non_periodic = FunctionSpace(self.flat_mesh,"CG",1)
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
        x[2]  = x[2] * (self.S_ex(x[0], x[1], x[2]) - \
                        self.B_ex(x[0], x[1], x[2]))
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
  
  def calc_sigma(self):
    """
    Calculatethe Cauchy stress tensor of velocity field <u>.
    """
    U   = as_vector([self.u, self.v, self.w])
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
  
  def calc_tau(self):
    """
    Calculate the deviatoric stress tensor of velocity field <u>.
    """
    U     = as_vector([self.u, self.v, self.w])
    n     = U.geometric_dimension()
    eta   = self.eta
    gradU = nabla_grad(U)
    divU  = nabla_div(U)
    tau   = 2 * eta * (gradU + gradU.T - 2.0/n * divU * Identity(n))
    return tau
  
  def calc_tau_dukowicz(self):
    """
    Calculate the deviatoric stress tensor of velocity field <u>.
    """
    u   = self.u
    v   = self.v
    eta = self.eta
    U   = as_vector([self.u, self.v, self.w])
    
    # second invariant of the strain rate tensor squared :
    epsdot2 = + 0.5 * (u.dx(2)**2 + v.dx(2)**2 + (u.dx(1) + v.dx(0))**2) \
              +        u.dx(0)**2 + v.dx(1)**2 + (u.dx(0) + v.dx(1))**2
    gradU   = nabla_grad(U)
    tau     = 2 * eta * epsdot2 * (gradU + gradU.T)
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
    v.update()
    return v

  def calc_component_stress(self, u_dir):
    """
    Calculate the deviatoric component of stress in the direction of <u>.
    """
    print "::: calculating component stress :::"
    Q      = self.Q                            # function space
    ff     = self.ff                           # facet function for boundaries
    sig    = self.calc_tau()                   # deviatoric stress tensor
    com    = dot(sig, u_dir)                   # component of stress in u-dir.
    com_n  = project(sqrt(inner(com, com)),Q)  # magnitude of com
    dvdx   = grad(com_n)                       # spatial derivative
    dvdu   = dot(dvdx, u_dir)                  # projection of dvdx onto dir
    phi    = TestFunction(Q)                   # test function
    v      = TrialFunction(Q)                  # trial function
    bc     = DirichletBC(Q, 0.0, ff, 3)        # boundary condition
    a      = v.dx(2) * phi * dx                # bilinear part
    L      = dvdu * phi * dx                   # linear part
    v      = Function(Q)                       # solution function
    solve(a == L, v, bc)                       # solve
    v.update()                                 # update ghost-vertices 
    v      = self.extrude(v, 2, 2)             # extrude the integral
    return v

  def calc_tau_bas(self):
    """
    """
    print "::: calculating tau_bas :::"
    beta2 = self.beta2
    Q     = self.Q
    u     = self.u
    v     = self.v
    w     = self.w
    H     = self.S - self.B
  
    beta2_e = self.extrude(beta2, 3, 2)
    u_bas_e = self.extrude(u,     3, 2)
    v_bas_e = self.extrude(v,     3, 2)
    w_bas_e = self.extrude(w,     3, 2)

    tau_bas_u = project(beta2_e*H*u_bas_e, Q)
    tau_bas_v = project(beta2_e*H*v_bas_e, Q)
    tau_bas_w = project(beta2_e*H*w_bas_e, Q)

    tau_bas_u.update()                             # eliminate ghost vertices 
    tau_bas_v.update()                             # eliminate ghost vertices 
    tau_bas_w.update()                             # eliminate ghost vertices 
  
    return as_vector([tau_bas_u, tau_bas_v, tau_bas_w]) 

  def calc_tau_drv(self):
    """
    """
    print "::: calculating tau_drv :::"
    ff    = self.ff
    Q     = self.Q
    S     = self.S
    B     = self.B
    rho   = self.rho
    g     = self.g
    H     = S - B
    gradS = grad(S)
    
    gradS_u = gradS[0]
    gradS_v = gradS[1]
    gradS_w = gradS[2]
  
    tau_drv_u = project(rho*g*H*gradS_u, Q)
    tau_drv_v = project(rho*g*H*gradS_v, Q)
    tau_drv_w = project(rho*g*H*gradS_w, Q)

    tau_drv_u.update()                             # eliminate ghost vertices 
    tau_drv_v.update()                             # eliminate ghost vertices 
    tau_drv_w.update()                             # eliminate ghost vertices 
  
    return as_vector([tau_drv_u, tau_drv_v, tau_drv_w])

  def component_stress_old(self):
    """
    Calculate each of the component stresses which define the full stress
    of the ice-sheet.
    
    RETURNS:
      tau_lon - longitudinal stress field
      tau_lat - lateral stress field
      tau_vrt - vertical stress field
      tau_bas - frictional sliding stress at the bed
      tau_drv - driving stress of the system 
    
    Note: tau_drv = tau_lon + tau_lat + tau_vrt + tau_bas
    
    """
    print "::: CALCULATING COMPONENT STRESSES :::"
    Q     = self.Q
    u     = self.u
    v     = self.v
    w     = self.w
  
    u_v = u.vector().array()
    v_v = v.vector().array()
    w_v = w.vector().array()

    norm_u = np.sqrt(u_v**2 + v_v**2)

    u_v_n  = u_v / norm_u
    v_v_n  = v_v / norm_u
    w_v_n  = w_v / norm_u
    
    u_n    = Function(Q)
    v_n    = Function(Q)
    w_n    = Function(Q)

    u_n.vector().set_local(u_v_n)
    v_n.vector().set_local(v_v_n)
    w_n.vector().set_local(w_v_n)
    
    U_n = as_vector([u_n, v_n, w_n])
    U_t = as_vector([v_n,-u_n, w_n])
    U_v = as_vector([0,   0,   1])
 
    tau_lon = project(self.calc_component_stress(U_n), Q)
    tau_lat = project(self.calc_component_stress(U_t), Q)
    tau_vrt = project(self.calc_component_stress(U_v), Q)
    tau_bas = self.calc_tau_bas()
    tau_drv = self.calc_tau_drv()

    return tau_lon, tau_lat, tau_vrt, tau_bas, tau_drv

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
    print "::: CALCULATING COMPONENT STRESSES :::"
    Q     = self.Q
    u     = self.u
    v     = self.v
    w     = self.w
    S     = self.S
    B     = self.B
    H     = S - B
    eta   = self.eta
    beta2 = self.beta2
    
    #===========================================================================
    # convert to array and normailze the components of U :
    u_v = u.vector().array()
    v_v = v.vector().array()
    w_v = w.vector().array()

    norm_u = np.sqrt(u_v**2 + v_v**2)

    u_v_n  = u_v / norm_u
    v_v_n  = v_v / norm_u
    w_v_n  = w_v / norm_u
 
    u_n    = Function(Q)
    v_n    = Function(Q)
    w_n    = Function(Q)

    u_n.vector().set_local(u_v_n)
    v_n.vector().set_local(v_v_n)
    w_n.vector().set_local(w_v_n)

    #===========================================================================
    # form the stokes equations in the normal direction (n) and tangential 
    # direction (t) in relation to the stress-tensor :
    U_n = as_vector([u_n, v_n, 0])
    U_t = as_vector([v_n,-u_n, 0])

    # directional derivatives :
    gradu = grad(u)
    gradv = grad(v)
    
    dudn = dot(gradu, U_n)
    dvdn = dot(gradv, U_n)
    dudt = dot(gradu, U_t)
    dvdt = dot(gradv, U_t)

    # trial and test functions for linear solve :
    phi   = TestFunction(Q)
    dtau  = TrialFunction(Q)
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # integration by parts directional derivative terms :
    gradphi = grad(phi)
    dphidn  = dot(gradphi, U_n)
    dphidt  = dot(gradphi, U_t)
    
    # stokes equation weak form in normal dir. (n) and tangent dir. (t) :
    r_tau_n1 = dphidn*eta*(4*dudn + 2*dvdt) * dx
    r_tau_t1 = dphidt*eta*(  dudt +   dvdn) * dx
    r_tau_t2 = dphidt*eta*(4*dvdt + 2*dudn) * dx
    r_tau_n2 = dphidn*eta*(  dudt +   dvdn) * dx

    # assemble the vectors :
    r_tau_n1_v = assemble(r_tau_n1)
    r_tau_n2_v = assemble(r_tau_n2)
    r_tau_t1_v = assemble(r_tau_t1)
    r_tau_t2_v = assemble(r_tau_t2)
    
    # solution functions :
    tau_n1 = Function(Q)
    tau_n2 = Function(Q)
    tau_t1 = Function(Q)
    tau_t2 = Function(Q)
    
    # solve the linear system :
    solve(M, tau_n1.vector(), r_tau_n1_v)
    solve(M, tau_n2.vector(), r_tau_n2_v)
    solve(M, tau_t1.vector(), r_tau_t1_v)
    solve(M, tau_t2.vector(), r_tau_t2_v)

    # integrate vertically :
    tau_lon = self.vert_integrate(tau_n1 + tau_t1)
    tau_lon = project(self.extrude(tau_lon, 2, 2), Q)
    tau_lat = self.vert_integrate(tau_n2 + tau_t2)
    tau_lat = project(self.extrude(tau_lat, 2, 2), Q)

    # calculate the basal shear and driving stresses :
    tau_bas = self.calc_tau_bas()
    tau_drv = self.calc_tau_drv()

    return tau_lon, tau_lat, tau_bas, tau_drv
   
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



