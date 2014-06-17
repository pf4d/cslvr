from fenics import *
import numpy as np

class Model(object):
  """ 
  Instance of a 2D flowline ice model that contains geometric and scalar 
  parameters and supporting functions.  This class does not contain actual 
  physics but rather the interface to use physics in different simulation 
  types.
  """

  def __init__(self, out_dir='./'):
    self.per_func_space = False  # function space is undefined
    self.out_dir        = out_dir

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
    
  def set_mesh(self, mesh, deform=True):
    """
    Overwrites the previous mesh with a new one
    
    :param mesh        : Dolfin mesh to be written
    :param flat_mesh   : Dolfin flat mesh to be written
    :param bool deform : If True, deform the mesh to surface and bed data 
                         provided by the set_geometry method.
    """
    self.mesh      = mesh
    self.flat_mesh = Mesh(mesh)
    self.Q         = FunctionSpace(mesh, "CG", 1)

    if deform:
      self.deform_mesh_to_geometry()

  def deform_mesh_to_geometry(self):
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
    self.ff      = FacetFunction('size_t', self.mesh,      0)
    self.ff_flat = FacetFunction('size_t', self.flat_mesh, 0)
    
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
    
    self.ff_flat.set_values(self.ff.array())
    
    self.ds      = Measure('ds')[self.ff]
    self.ds_flat = Measure('ds')[self.ff_flat]
     
  def set_parameters(self, params):
    """
    Sets the model's dictionary of parameters
    
    :param params: :class:`~src.physical_constants.IceParameters` object 
       containing model-relavent parameters
    """
    self.params = params
  
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
    Calculatethe Cauchy stress tensor of velocity field U.
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
   0"""
    n     = u.shape()[0]
    divu  = 0.0
    for i in range(n):
      divu += u[i].dx(i)
    return divu
  
  def calc_tau(self):
    """
    Calculate the deviatoric stress tensor of velocity field U.
    """
    U     = as_vector([self.u, self.v, self.w])
    n     = U.geometric_dimension()
    eta   = self.eta
    gradU = nabla_grad(U)
    divU  = nabla_div(U)
    tau   = 2 * eta * (gradU + gradU.T - 2.0/n * divU * Identity(n))
    return tau
  
  def calc_R(self):
    """
    Calculate the resistive stress tensor of velocity field U.
    """
    u   = self.u
    v   = self.v
    eta = self.eta
    U   = as_vector([self.u, self.v, self.w])
    
    gradU    = nabla_grad(U)
    epsdot   = gradU + gradU.T
    epsdot00 = 2*epsdot[0,0] + epsdot[1,1]
    epsdot11 = 2*epsdot[1,1] + epsdot[0,0]
    
    epsdot   = as_matrix([[epsdot00,     epsdot[0,1],  epsdot[0,2]],
                          [epsdot[1,0],  epsdot11,     epsdot[1,2]],
                          [epsdot[2,0],  epsdot[2,1],  epsdot[2,2]]])
    return eta * epsdot
     
  def extrude(self, f, b, d, Q='self'):
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
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff  = self.ff
    phi = TestFunction(Q)
    v   = TrialFunction(Q)
    a   = v.dx(d) * phi * dx
    L   = DOLFIN_EPS * phi * dx  # really close to zero to fool FFC
    bc  = DirichletBC(Q, f, ff, b)
    v   = Function(Q)
    solve(a == L, v, bc)
    return v
  
  def vert_integrate(self, u, Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff     = self.ff                       # facet function defines boundaries
    phi    = TestFunction(Q)               # test function
    v      = TrialFunction(Q)              # trial function
    bc     = DirichletBC(Q, 0.0, ff, 3)    # integral is zero on bed (ff = 3) 
    a      = v.dx(2) * phi * dx            # rhs
    L      = u * phi * dx                  # lhs
    v      = Function(Q)                   # solution function
    solve(a == L, v, bc)                   # solve
    return v

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

  def normalize_vector(self, U, Q='self'):
    """
    Create a normalized vector of the UFL vector <U>.
    """
    if type(Q) != FunctionSpace:
      Q = self.Q

    # iterate through each component and convert to array :
    U_v = []
    for u in U:
      # convert to array and normailze the components of U :
      u_v = u.vector().array()
      U_v.append(u_v)
    U_v = np.array(U_v)

    # calculate the norm :
    norm_u = np.sqrt(sum(U_v**2))
    
    # normalize the vector :
    U_v /= norm_u
    
    # convert back to fenics :
    U_f = []
    for u_v in U_v:
      u_f = Function(Q)
      u_f.vector().set_local(u_v)
      U_f.append(u_f)

    # return a UFL vector :
    return as_vector(U_f)

  def calc_component_stress(self, u_dir, Q='self'):
    """
    Calculate the deviatoric component of stress in the direction of 
    the UFL vector <u_dir>.
    """
    print "::: calculating component stress :::"
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
    print "::: calculating component stress :::"
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
    print "::: calculating tau_bas :::"
    if type(Q) != FunctionSpace:
      Q = self.Q
    beta2 = self.beta2
    u     = self.u
    v     = self.v
    w     = self.w
    H     = self.S - self.B
  
    beta2_e = self.extrude(beta2, 3, 2, Q)
    u_bas_e = self.extrude(u,     3, 2, Q)
    v_bas_e = self.extrude(v,     3, 2, Q)
    w_bas_e = self.extrude(w,     3, 2, Q)

    tau_bas_u = project(beta2_e*H*u_bas_e, Q)
    tau_bas_v = project(beta2_e*H*v_bas_e, Q)
    tau_bas_w = project(beta2_e*H*w_bas_e, Q)

    return as_vector([tau_bas_u, tau_bas_v, tau_bas_w]) 

  def calc_tau_drv(self, Q='self'):
    """
    """
    print "::: calculating tau_drv :::"
    if type(Q) != FunctionSpace:
      Q = self.Q
    ff    = self.ff
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
    print "::: calculating 'stress-balance' :::"
    out_dir = self.out_dir
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

  def component_stress_stokes_c(self):
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
    print "::: calculating 'stokes-balance' :::"
    out_dir = self.out_dir
    Q       = self.Q
    u       = self.u
    v       = self.v
    w       = self.w
    S       = self.S
    B       = self.B
    H       = S - B
    eta     = self.eta
    rho     = self.rho
    g       = self.g
    
    # create functions used to solve for velocity :
    V        = MixedFunctionSpace([Q,Q])
    dU       = TrialFunction(V)
    du, dv   = split(dU)
    Phi      = TestFunction(V)
    phi, psi = split(Phi)
    U_s      = Function(V)
    u_s,v_s  = split(U_s)
    
    #===========================================================================
    # driving stress (d) and basal drag (b) weak form :
    tau_bx = - phi.dx(2) * eta * du.dx(2) * dx
    tau_by = - psi.dx(2) * eta * dv.dx(2) * dx
    tau_dx = phi * rho * g * S.dx(0) * dx
    tau_dy = psi * rho * g * S.dx(1) * dx

    # longitudinal and lateral drag weak form :
    tau_xx = - phi.dx(0) * eta * (4*du.dx(0) + 2*dv.dx(1)) * dx
    tau_xy = - phi.dx(1) * eta * (  du.dx(1) +   dv.dx(0)) * dx
    tau_yx = - psi.dx(0) * eta * (  du.dx(1) +   dv.dx(0)) * dx
    tau_yy = - psi.dx(1) * eta * (4*dv.dx(1) + 2*du.dx(0)) * dx
  
    # form residual in mixed space :
    r1 = tau_xx + tau_xy + tau_bx - tau_dx
    r2 = tau_yy + tau_yx + tau_by - tau_dy
    r  = r1 + r2

    # solve for u and v : 
    solve(lhs(r) == rhs(r), U_s)
    
    #===========================================================================
    # resolve with corrected velocities :
    u_s = project(u_s)
    v_s = project(v_s)

    # trial and test functions for linear solve :
    phi   = TestFunction(Q)
    dtau  = TrialFunction(Q)
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # driving stress (d) and basal drag (b) weak form :
    tau_bx = - phi.dx(2) * eta * u.dx(2) * dx
    tau_by = - phi.dx(2) * eta * v.dx(2) * dx
    tau_dx = phi * rho * g * S.dx(0) * dx
    tau_dy = phi * rho * g * S.dx(1) * dx

    # longitudinal and lateral drag weak form :
    tau_xx = - phi.dx(0) * eta * (4*u.dx(0) + 2*v.dx(1)) * dx
    tau_xy = - phi.dx(1) * eta * (  u.dx(1) +   v.dx(0)) * dx
    tau_yx = - phi.dx(0) * eta * (  u.dx(1) +   v.dx(0)) * dx
    tau_yy = - phi.dx(1) * eta * (4*v.dx(1) + 2*u.dx(0)) * dx
    
    # the residuals :
    tau_totx = tau_xx + tau_xy + tau_bx - tau_dx
    tau_toty = tau_yy + tau_yx + tau_by - tau_dy

    # assemble the vectors :
    tau_xx_v   = assemble(tau_xx)
    tau_xy_v   = assemble(tau_xy)
    tau_yx_v   = assemble(tau_yx)
    tau_yy_v   = assemble(tau_yy)
    tau_bx_v   = assemble(tau_bx)
    tau_by_v   = assemble(tau_by)
    tau_dx_v   = assemble(tau_dx)
    tau_dy_v   = assemble(tau_dy)
    tau_totx_v = assemble(tau_totx)
    tau_toty_v = assemble(tau_toty)
    
    # solution functions :
    tau_xx   = Function(Q)
    tau_xy   = Function(Q)
    tau_yx   = Function(Q)
    tau_yy   = Function(Q)
    tau_bx   = Function(Q)
    tau_by   = Function(Q)
    tau_dx   = Function(Q)
    tau_dy   = Function(Q)
    tau_totx = Function(Q)
    tau_toty = Function(Q)
    
    # solve the linear system :
    solve(M, tau_xx.vector(),   tau_xx_v)
    solve(M, tau_xy.vector(),   tau_xy_v)
    solve(M, tau_yx.vector(),   tau_yx_v)
    solve(M, tau_yy.vector(),   tau_yy_v)
    solve(M, tau_bx.vector(),   tau_bx_v)
    solve(M, tau_by.vector(),   tau_by_v)
    solve(M, tau_dx.vector(),   tau_dx_v)
    solve(M, tau_dy.vector(),   tau_dy_v)
    solve(M, tau_totx.vector(), tau_totx_v)
    solve(M, tau_toty.vector(), tau_toty_v)

    # integrate vertically and extrude the result :
    tau_xx = self.vert_integrate(tau_xx)
    tau_xx = project(self.extrude(tau_xx, 2, 2))
    tau_xy = self.vert_integrate(tau_xy)
    tau_xy = project(self.extrude(tau_xy, 2, 2))
    tau_yx = self.vert_integrate(tau_yx)
    tau_yx = project(self.extrude(tau_yx, 2, 2))
    tau_yy = self.vert_integrate(tau_yy)
    tau_yy = project(self.extrude(tau_yy, 2, 2))
    tau_bx = self.vert_integrate(tau_bx)
    tau_bx = project(self.extrude(tau_bx, 2, 2))
    tau_by = self.vert_integrate(tau_by)
    tau_by = project(self.extrude(tau_by, 2, 2))
    tau_dx = self.vert_integrate(tau_dx)
    tau_dx = project(self.extrude(tau_dx, 2, 2))
    tau_dy = self.vert_integrate(tau_dy)
    tau_dy = project(self.extrude(tau_dy, 2, 2))
    tau_totx = self.vert_integrate(tau_totx)
    tau_totx = project(self.extrude(tau_totx, 2, 2))
    tau_toty = self.vert_integrate(tau_toty)
    tau_toty = project(self.extrude(tau_toty, 2, 2))

    # calculate the magnitudes :
    tau_lon = project(sqrt(tau_xx**2 + tau_yy**2))
    tau_lat = project(sqrt(tau_xy**2 + tau_yx**2))
    tau_drv = project(sqrt(tau_dx**2 + tau_dy**2))
    tau_bas = project(sqrt(tau_bx**2 + tau_by**2))

    # output calculated fields :
    File(out_dir + 'tau_lon.pvd')  << tau_lon
    File(out_dir + 'tau_lat.pvd')  << tau_lat
    File(out_dir + 'tau_drv.pvd')  << tau_drv
    File(out_dir + 'tau_bas.pvd')  << tau_bas

    # output the files to the specified directory :
    File(out_dir + 'tau_xx.pvd')   << tau_xx
    File(out_dir + 'tau_xy.pvd')   << tau_xy
    File(out_dir + 'tau_yx.pvd')   << tau_yx
    File(out_dir + 'tau_yy.pvd')   << tau_yy
    File(out_dir + 'tau_bx.pvd')   << tau_bx
    File(out_dir + 'tau_dx.pvd')   << tau_dx
    File(out_dir + 'tau_totx.pvd') << tau_totx
    File(out_dir + 'tau_toty.pvd') << tau_toty
    File(out_dir + 'u_s.pvd')      << u_s
    File(out_dir + 'v_s.pvd')      << v_s

    output = (tau_xx, tau_xy, tau_yx, tau_yy, tau_bx, tau_by, tau_dx, tau_by,
              tau_totx, tau_toty)
   
    # return the functions for further analysis :
    return output

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
    self.x             = SpatialCoordinate(self.mesh)
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
    self.Tstar         = Function(self.Q)
    self.W             = Function(self.Q) 
    self.Vd            = Function(self.Q) 
    self.Pe            = Function(self.Q) 
    self.Sl            = Function(self.Q) 
    self.Pc            = Function(self.Q)
    self.Nc            = Function(self.Q)
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



