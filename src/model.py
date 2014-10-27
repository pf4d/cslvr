from fenics      import *
from ufl.indexed import Indexed
from termcolor   import colored, cprint
import numpy         as np
from abc         import ABCMeta, abstractmethod

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
    self.MPI_rank       = MPI.rank(mpi_comm_world())
    
  def set_geometry(self, sur, bed, deform=True):
    """
    Sets the geometry of the surface and bed of the ice sheet.
    
    :param sur  : Expression representing the surface of the mesh
    :param bed  : Expression representing the base of the mesh
    :param mask : Expression representing a mask of grounded (0) and floating 
                  (1) areas of the ice.
    """
    self.boundary_markers = []
    # Contains the u, v, and w components of the velocity values 
    # for each boundary
    self.boundary_u = []
    self.boundary_v = []
    
    # This list will store the integer value corresponding to each boundary
    self.boundary_values = []
    
    # The marker value for the next added boundary. This value will be auto 
    # incremented
    self.marker_val = 8
    self.S_ex = sur
    self.B_ex = bed
    
    if deform:
      self.deform_mesh_to_geometry()
  
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
    if self.MPI_rank==0:
      s    = "::: generating mesh :::"
      text = colored(s, 'magenta')
      print text

    self.mesh      = BoxMesh(xmin, ymin, 0, xmax, ymax, 1, nx, ny, nz)
    self.flat_mesh = Mesh(self.mesh)
    
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
      
      self.Q_non_periodic = FunctionSpace(self.mesh, "CG", 1)
      self.Q_flat         = FunctionSpace(self.flat_mesh, "CG", 1, 
                                          constrained_domain=pBC)
      self.Q_flat_non_periodic = FunctionSpace(self.flat_mesh,"CG",1)
      self.per_func_space = True
   
  def set_mesh(self, mesh):
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

  def deform_mesh_to_geometry(self):
    """
    Deforms the mesh to the geometry.
    """
    if self.MPI_rank==0:
      s    = "::: deforming mesh to geometry :::"
      text = colored(s, 'magenta')
      print text
    
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


  def calculate_boundaries(self, mask=None, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    self.mask = mask
    self.adot = adot
    
    if self.MPI_rank==0:
      s    = "::: calculating boundaries :::"
      text = colored(s, 'magenta')
      print text
    
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
    if self.mask != None:

      if self.MPI_rank==0:
        s    = "    - iterating through facets - "
        text = colored(s, 'magenta')
        print text
      for f in facets(self.mesh):
        n       = f.normal()
        x_m     = f.midpoint().x()
        y_m     = f.midpoint().y()
        z_m     = f.midpoint().z()
        mask_xy = self.mask(x_m, y_m, z_m)
      
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
      
      if self.MPI_rank==0:
        s    = "    - iterating through cells - "
        text = colored(s, 'magenta')
        print text
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


    # iterate through the facets and mark each if on a boundary :
    #
    #   2 = high slope, upward facing ................ surface
    #   3 = high slope, downward facing .............. base
    #   4 = low slope, upward or downward facing ..... sides
    else:
      for f in facets(self.mesh):    
        # Flag that is set to true if the facet belonged to a user defined 
        # boundary
        marked = False
        
        # Check if the facet belongs in any of the user defined boundaries
        for i in range(len(self.boundary_markers)) :
          bm = self.boundary_markers[i]
          # Check if the current facet belongs to the boundary
          if bm.to_mark(f) :
            marked = True
            # If so, mark it with the correct integer value for the boundary
            val = self.boundary_values[i]
            self.ff[f] = val
            # Each facet can belong in only one boundary 
            break
        
        # If the facet hasn't been marked already, then we can test if it's some
        # other type of default boundary
        if not marked :
          n       = f.normal()    # unit normal vector to facet f
          tol     = 1e-3
        
          if n.z() >=  tol and f.exterior():
            self.ff[f] = 2
        
          elif n.z() <= -tol and f.exterior():
            self.ff[f] = 3
        
          elif n.z() >  -tol and n.z() < tol and f.exterior():
            self.ff[f] = 4
      
      for f in facets(self.flat_mesh):
        n       = f.normal()    # unit normal vector to facet f
      
        if   n.z() >=  tol and f.exterior():
          self.ff_flat[f] = 2
      
        elif n.z() <= -tol and f.exterior():
          self.ff_flat[f] = 3
      
        elif n.z() >  -tol and n.z() < tol and f.exterior():
          self.ff_flat[f] = 4
    
    self.ds      = Measure('ds')[self.ff]
    self.ds_flat = Measure('ds')[self.ff_flat]
    self.dx      = Measure('dx')[self.cf]

    # iterate through the facets and mark each if positive accumulation :
    #
    #   1 = high slope, upward facing ................ positive adot
    if self.adot != None:
      for f in facets(self.mesh):
        n       = f.normal()    # unit normal vector to facet f
        x_m     = f.midpoint().x()
        y_m     = f.midpoint().y()
        z_m     = f.midpoint().z()
        adot_xy = self.adot(x_m, y_m, z_m)
        if n.z() >= tol and f.exterior() and adot_xy > 0:
          self.ff_acc[f] = 1
 
  def set_subdomain(self, mesh, flat_mesh, ff, ff_flat):
    """
    Sets the mesh to be Mesh <mesh> and flat_mest to be Mesh <flat_mesh>,
    and sets the subdomains of the mesh and flat mesh to FacetFunction <ff> and
    <ff_flat> respectively.
    """
    if self.MPI_rank==0:
      s    = "::: setting subdomains :::"
      text = colored(s, 'magenta')
      print text
    self.mesh      = mesh
    self.flat_mesh = flat_mesh
    self.ff        = ff
    self.ff_flat   = ff_flat
    self.ds        = Measure('ds')[self.ff]
    self.ds_flat   = Measure('ds')[self.ff_flat]
     
  def set_parameters(self, params):
    """
    Sets the model's dictionary of parameters
    
    :param params: :class:`~src.physical_constants.IceParameters` object 
       containing model-relavent parameters
    """
    self.params = params
      
  def init_beta0(self, beta, U_ob, r, gradS):
    r"""
    Init beta from :math:`\tau_b = \tau_d`, the shallow ice approximation, 
    using the observed surface velocity <U_ob> as approximate basal 
    velocity and <gradS> the projected surface gradient. i.e.,

    .. math::
    \beta^2 \Vert U_b \Vert H^r = \rho g H \Vert \nabla S \Vert
    
    """
    if self.MPI_rank==0:
      s    = "::: initializing beta from U_ob :::"
      text = colored(s, 'magenta')
      print text
    Q        = self.Q
    rho      = self.rho
    g        = self.g
    H        = self.S - self.B
    U_mag    = project(sqrt(inner(U_ob, U_ob) + DOLFIN_EPS), Q)
    U_mag_v  = U_mag.vector().array()
    U_mag_v[U_mag_v < 0.5] = 0.5
    self.assign_variable(U_mag, U_mag_v)
    S_mag    = sqrt(inner(gradS, gradS) + DOLFIN_EPS)
    beta_0   = project(sqrt((rho*g*H*S_mag) / (H**r * U_mag)), Q)
    beta_0_v = beta_0.vector().array()
    beta_0_v[beta_0_v < DOLFIN_EPS] = DOLFIN_EPS
    self.assign_variable(beta, beta_0_v)
  
  def init_b0(self, b, U_ob, gradS):
    r"""
    Init rate-factor b from U_ob. 
    """
    if self.MPI_rank==0:
      s    = "::: initializing b from U_ob :::"
      text = colored(s, 'magenta')
      print text
   
    x      = self.x
    S      = self.S
    Q      = self.Q
    rho    = self.rho
    rho_w  = self.rho_w
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

    f_w    = rho*g*(S - x[2]) + rho_w*g*D

    epi_1  = as_vector([   2*u.dx(0) + v.dx(1), 
                        0.5*(u.dx(1) + v.dx(0)),
                        0.5* u.dx(2)            ])
    epi_2  = as_vector([0.5*(u.dx(1) + v.dx(0)),
                             u.dx(0) + 2*v.dx(1),
                        0.5* v.dx(2)            ])

    R  = - 2 * eta * dot(epi_1, grad(phi)) * dx \
         + rho * g * gradS[0] * phi * dx \
         #+ 2 * eta * dot(epi_2, grad(phi)) * dx \
         #+ rho * g * gradS[1] * phi * dx \
   
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
    eta = self.eta
    w   = self.w
    H   = self.calc_thickness() 
    P   = rho*g*H + 2*eta*w.dx(2)
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
    return 2 * eta * epsdot
     
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
    return v
  
  def vert_integrate(self, u, Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
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

  def norm(self, U):
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

    U_v, norm_u = self.norm(U)
    
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
    if self.MPI_rank==0:
      s    = "::: calculating component stress :::"
      text = colored(s, 'magenta')
      print text
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
    if self.MPI_rank==0:
      s    = "::: calculating component stress :::"
      text = colored(s, 'magenta')
      print text
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
    if self.MPI_rank==0:
      s    = "::: calculating tau_bas :::"
      text = colored(s, 'magenta')
      print text
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
    if self.MPI_rank==0:
      s    = "::: calculating tau_drv :::"
      text = colored(s, 'magenta')
      print text
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
    if self.MPI_rank==0:
      s    = "::: calculating 'stress-balance' :::"
      text = colored(s, 'magenta')
      print text
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

  def print_min_max(self, u, title):
    """
    Print the minimum and maximum values of <u>, a Vector, Function, or array.
    """
    if self.MPI_rank==0:
      if isinstance(u, GenericVector):
        uMin = u.array().min()
        uMax = u.array().max()
      elif isinstance(u, Function):
        uMin = u.vector().array().min()
        uMax = u.vector().array().max()
      elif isinstance(u, np.ndarray):
        uMin = u.min()
        uMax = u.max()
      elif isinstance(u, Indexed):
        u_n  = project(u, self.Q)
        uMin = u_n.vector().array().min()
        uMax = u_n.vector().array().max()
      else:
        print "print_min_max function requires a Vector, Function, array," \
              + " or Indexed, not %s." % type(u)
        uMin = uMax = 0.0
      s    = title + ' <min, max> : <%f, %f>' % (uMin, uMax)
      text = colored(s, 'yellow')
      print text


  def assign_variable(self, u, var):
    """
    Manually assign the values from <var> to Function <u>.  <var> may be an
    array, float, Expression, or Function.
    """
    if isinstance(u, Indexed):
      u = project(u, self.Q)

    if   isinstance(var, float) or isinstance(var, int):
      u.vector()[:] = var
    
    elif isinstance(var, np.ndarray):
      u.vector().set_local(var)
      u.vector().apply('insert')
    
    elif isinstance(var, Expression):
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


  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    if self.MPI_rank==0:
      s    = "::: initializing variables :::"
      text = colored(s, 'magenta')
      print text
    
    self.params.globalize_parameters(self) # make all the variables available 

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
    
    else:
      # surface and bed :
      self.S           = interpolate(self.S_ex, self.Q_non_periodic)
      self.B           = interpolate(self.B_ex, self.Q_non_periodic)
      self.Shat        = Function(self.Q_flat_non_periodic)
    
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
    self.eta           = Function(self.Q)
    self.P             = Function(self.Q)
    self.Tstar         = Function(self.Q)
    self.W             = Function(self.Q)
    self.W_r           = Function(self.Q)
    self.Vd            = Function(self.Q)
    self.Pe            = Function(self.Q)
    self.Sl            = Function(self.Q)
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
    self.Tstar         = self.T + self.gamma * (self.S - self.x[2])

    # free surface model :
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
    self.dSdx          = Function(self.Q_flat)
    self.dSdy          = Function(self.Q_flat)
    self.Ub            = Function(self.Q_flat)
    self.u_balance     = Function(self.Q)
    self.v_balance     = Function(self.Q)
    
  """ Adds a Dirichlet boundary condition to the BP velocity solver.
  Inputs :
  bm : A boundary marker object that determines if a facet belongs on the 
  boundary
  u, v, w : Components of the velocity on the value (constant for now)
  """   
  def add_bp_dbc(self, bm, u, v) :
    # The boundary marker object and the corresponding velocity components
    # are associated by array index
    self.boundary_markers.append(bm)
    self.boundary_u.append(u)
    self.boundary_v.append(v)
    # This is the integer value assigned to this value, which will be used
    # in the facet function
    self.boundary_values.append(self.marker_val)
    # Auto increment the marker value
    self.marker_val += 1
    
class BoundaryMarker(object):
    __metaclass__ = ABCMeta

    """ 
    Takes in a facet, returns true if the facet should be marked and false if
    otherwise.
    Inputs: 
    f : A facet """
    @abstractmethod
    def to_mark(self,f):
        pass




