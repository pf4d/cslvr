from fenics       import *
from ufl.indexed  import Indexed
from abc          import ABCMeta, abstractmethod
from physics      import Physics
from solvers      import Solver
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

  def __init__(self):
    """
    Create and instance of the model.
    """
    self.MPI_rank = MPI.rank(mpi_comm_world())
    self.color    = 'magenta'
  
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
    
    Q = FunctionSpace(self.mesh, 'CG', 1)
    
    self.S = interpolate(self.S_ex, Q)
    self.B = interpolate(self.B_ex, Q)

  def deform_mesh_to_geometry(self):
    """
    Deforms the mesh to the geometry.
    """
    s = "::: deforming mesh to geometry :::"
    print_text(s, self.color)
    
    # transform z :
    # thickness = surface - base, z = thickness + base
    # Get the height of the mesh, assumes that the base is at z=0
    max_height = self.mesh.coordinates()[:,2].max()
    min_height = self.mesh.coordinates()[:,2].min()
    mesh_height = max_height - min_height
    
    for x in self.mesh.coordinates():
      x[2] = (x[2] / mesh_height) * ( + self.S_ex(x[0],x[1],x[2]) \
                                      - self.B_ex(x[0],x[1],x[2]) )
      x[2] = x[2] + self.B_ex(x[0], x[1], x[2])

  def generate_uniform_mesh(self, nx, ny, nz, xmin, xmax, ymin, ymax, 
                            generate_pbcs=False):
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
    s = "::: generating rectangle mesh :::"
    print_text(s, self.color)

    self.mesh           = BoxMesh(xmin, ymin, 0, xmax, ymax, 1, nx, ny, nz)
    self.flat_mesh      = Mesh(self.mesh)
   
    # generate periodic boundary conditions if required :
    if generate_pbcs:
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

      pBC      = PeriodicBoundary()
      self.Q   = FunctionSpace(self.mesh, "CG", 1, constrained_domain=pBC)
      self.Q2  = MixedFunctionSpace([self.Q]*2)
      self.Q4  = MixedFunctionSpace([self.Q]*4)
      
      self.Q_flat         = FunctionSpace(self.flat_mesh, "CG", 1, 
                                          constrained_domain=pBC)
      self.Q_flat_non_per = FunctionSpace(self.flat_mesh, "CG", 1)
    
    else :
      self.Q = FunctionSpace(self.mesh, "CG", 1)
   
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    self.mesh      = mesh
    self.flat_mesh = Mesh(mesh)
    self.Q_flat    = FunctionSpace(self.flat_mesh, "CG", 1)
    self.Q         = FunctionSpace(mesh,           "CG", 1)
    self.DQ        = FunctionSpace(self.mesh,      "DG", 1)
    self.Q2        = MixedFunctionSpace([self.Q]*2)
    self.Q3        = MixedFunctionSpace([self.Q]*3)
    self.Q4        = MixedFunctionSpace([self.Q]*4)
    
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

  def calculate_boundaries(self, mask, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    self.mask    = mask
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
    
    tol = 1e-3
    
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
    s = "    - iterating through facets - "
    print_text(s, self.color)
    for f in facets(self.mesh):
      n       = f.normal()
      x_m     = f.midpoint().x()
      y_m     = f.midpoint().y()
      z_m     = f.midpoint().z()
      mask_xy = self.mask(x_m, y_m, z_m)
      if self.adot_ex != None:
        adot_xy = self.adot_ex(x_m, y_m, z_m)
        if n.z() >= tol and f.exterior() and adot_xy > 0:
          self.ff_acc[f] = 1
    
      if   n.z() >=  tol and f.exterior():
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
    
    s = "    - iterating through cells - "
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
      
  def init_beta(self, beta, U_mag, r, gradS):
    r"""
    Init beta from :math:`\tau_b = \tau_d`, the shallow ice approximation, 
    using the observed surface velocity <U_ob> as approximate basal 
    velocity and <gradS> the projected surface gradient. i.e.,

    .. math::
    \beta^2 \Vert U_b \Vert H^r = \rho g H \Vert \nabla S \Vert
    
    """
    s = "::: initializing beta from U_ob :::"
    print_text(s, self.color)
    Q        = self.Q
    rhoi     = self.rhoi
    g        = self.g
    H        = self.S - self.B
    U_mag_v  = U_mag.vector().array()
    U_mag_v[U_mag_v < 0.5] = 0.5
    self.assign_variable(U_mag, U_mag_v)
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta_0   = project(sqrt((rhoi*g*H*S_mag) / (H**r * U_mag)), Q)
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    self.assign_variable(beta, beta_0_v)
    print_min_max(beta, 'beta')
  
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
    #H       = self.calc_thickness()
    #P_f     = + rhoi * g * H * phi * dGamma \
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
    if isinstance(u, Indexed):
      u = project(u, self.Q)
    
    if   isinstance(var, pc.PhysicalConstant):
      u.vector()[:] = var.real

    elif isinstance(var, float) or isinstance(var, int):
      u.vector()[:] = var
    
    elif isinstance(var, np.ndarray):
      u.vector().set_local(var)
      u.vector().apply('insert')
    
    elif isinstance(var, Expression) or isinstance(var, Constant):
      u.interpolate(var)

    elif isinstance(var, GenericVector):
      u.vector().set_local(var.array())
      u.vector().apply('insert')

    elif isinstance(var, Function):
      u.vector().set_local(var.vector().array())
      u.vector().apply('insert')
    
    elif isinstance(var, Indexed):
      u.vector().set_local(project(var, self.Q).vector().array())
      u.vector().apply('insert')

    elif isinstance(var, str):
      File(var) >> u

    else:
      print "*************************************************************"
      print "assign_variable() function requires a Function, array, float," + \
            " int, \nVector, Expression, Indexed, or string path to .xml, " + \
            "not \n%s" % type(var)
      print "*************************************************************"
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
    
    # P1 vector function space :
    self.V             = VectorFunctionSpace(self.mesh, "CG", 1)

    # Coordinates of various types 
    self.x             = SpatialCoordinate(self.mesh)
    self.sigma         = project((self.x[2] - self.B) / (self.S - self.B))

    # Velocity model
    self.U             = Function(self.Q2)
    self.u             = Function(self.Q)
    self.v             = Function(self.Q)
    self.w             = Function(self.Q)
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
    self.P             = Function(self.Q)
    self.W             = Function(self.Q)
    self.W_r           = Function(self.Q)
    self.Vd            = Function(self.Q)
    self.Pe            = Function(self.Q)
    self.Sl_shf        = Function(self.Q)
    self.Sl_gnd        = Function(self.Q)
    self.Pc            = Function(self.Q)
    self.Nc            = Function(self.Q)
    self.Pb            = Function(self.Q)
    self.Lsq           = Function(self.Q)
    self.a_T           = Function(self.Q)
    self.Q_T           = Function(self.Q)
    self.w_T           = Function(self.Q)
    
    # Enthalpy model
    self.H_surface     = Function(self.Q)
    self.H_float       = Function(self.Q)
    self.H             = Function(self.Q)
    self.T             = Function(self.Q)
    self.q_geo         = Function(self.Q)
    self.W0            = Function(self.Q)
    self.W             = Function(self.Q)
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
    



