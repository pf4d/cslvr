from fenics         import *
from dolfin_adjoint import *
from io             import print_text, get_text, print_min_max
from model_new      import Model
from pylab          import inf
import sys

class D3Model(Model):
  """ 
  """

  def __init__(self, out_dir='./results/'):
    """
    Create and instance of the model.
    """
    Model.__init__(self, out_dir)
    self.D3Model_color = '150'
  
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    s = "::: setting 3D mesh :::"
    print_text(s, self.model_color)
    self.mesh       = mesh
    self.flat_mesh  = Mesh(mesh)
    self.dim        = self.mesh.ufl_cell().topological_dimension()
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
    print_text(s, self.model_color)

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D3Model, self).generate_function_spaces(use_periodic)

    s = "::: generating 3D function spaces :::"
    print_text(s, self.D3Model_color)
    
    self.V      = VectorFunctionSpace(self.mesh, "CG", 1)
    
    s = "    - 3D function spaces created - "
    print_text(s, self.D3Model_color)
    
  def calculate_boundaries(self, mask=None, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    s = "::: calculating boundaries :::"
    print_text(s, self.D3Model_color)
    
    # this function contains markers which may be applied to facets of the mesh
    self.ff      = FacetFunction('size_t', self.mesh, 0)
    self.ff_acc  = FacetFunction('size_t', self.mesh, 0)
    self.cf      = CellFunction('size_t',  self.mesh, 0)
    dofmap       = self.Q.dofmap()
    shf_dofs     = []
    gnd_dofs     = []
    
    # default to all grounded ice :
    if mask == None:
      mask = Expression('0.0', element=self.Q.ufl_element())
    
    # default to all positive accumulation :
    if adot == None:
      adot = Expression('1.0', element=self.Q.ufl_element())
    
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
    s = "    - iterating through %i facets of mesh - " % self.num_facets
    print_text(s, self.D3Model_color)
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
        #self.ff[f] = 2
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 0:
          self.ff[f] = 5
        else:
          self.ff[f] = 3
        #self.ff[f] = 3
    
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        if mask_xy > 0:
          self.ff[f] = 4
        else:
          self.ff[f] = 7
        #self.ff[f] = 4
    
    s = "    - done - "
    print_text(s, self.D3Model_color)
    
    s = "    - iterating through %i cells - " % self.num_cells
    print_text(s, self.D3Model_color)
    for c in cells(self.mesh):
      x_m     = c.midpoint().x()
      y_m     = c.midpoint().y()
      z_m     = c.midpoint().z()
      mask_xy = mask(x_m, y_m, z_m)

      if mask_xy > 0:
        self.cf[c] = 1
      else:
        self.cf[c] = 0
    
    s = "    - done - "
    print_text(s, self.D3Model_color)

    self.ds      = Measure('ds')[self.ff]
    self.dx      = Measure('dx')[self.cf]
    
    self.dx_s    = self.dx(1)              # internal shelves
    self.dx_g    = self.dx(0)              # internal grounded
    self.dx      = self.dx(1) + self.dx(0) # entire internal
    self.dGnd    = self.ds(3)              # grounded bed
    self.dFlt    = self.ds(5)              # floating bed
    self.dSde    = self.ds(4)              # sides
    self.dBed    = self.dGnd + self.dFlt   # bed
    self.dSrf_s  = self.ds(6)              # surface
    self.dSrf_g  = self.ds(2)              # surface
    self.dSrf    = self.ds(6) + self.ds(2) # entire surface
    
  def calculate_flat_mesh_boundaries(self, mask=None, adot=None):
    """
    Determines the boundaries of the current model mesh
    """
    s = "::: calculating flat_mesh boundaries :::"
    print_text(s, self.D3Model_color)

    self.Q_flat = FunctionSpace(self.flat_mesh, "CG", 1, 
                                constrained_domain=self.pBC)
    
    # this function contains markers which may be applied to facets of the mesh
    self.ff_flat = FacetFunction('size_t', self.flat_mesh, 0)
    
    # default to all grounded ice :
    if mask == None:
      mask = Expression('0.0', element=self.Q.ufl_element())
    
    # default to all positive accumulation :
    if adot == None:
      adot = Expression('1.0', element=self.Q.ufl_element())
    
    tol = 1e-6
    
    s = "    - iterating through %i facets of flat_mesh - " % self.num_facets
    print_text(s, self.D3Model_color)
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
        if mask_xy > 0:
          self.ff_flat[f] = 4
        else:
          self.ff_flat[f] = 7
        #self.ff_flat[f] = 4
    
    s = "    - done - "
    print_text(s, self.D3Model_color)
    
    self.ds_flat = Measure('ds')[self.ff_flat]
  
  def set_subdomains(self, ff, cf, ff_acc):
    """
    Set the facet subdomains to FacetFunction <ff>, and set the cell subdomains 
    to CellFunction <cf>, and accumulation FacetFunction to <ff_acc>.
    """
    super(D3Model, self).set_subdomains(ff, cf, ff_acc)

    s = "::: setting 3D subdomains :::"
    print_text(s, self.D3Model_color)
    
    self.dx_s    = self.dx(1)              # internal shelves
    self.dx_g    = self.dx(0)              # internal grounded
    self.dx      = self.dx(1) + self.dx(0) # entire internal
    self.dGnd    = self.ds(3)              # grounded bed
    self.dFlt    = self.ds(5)              # floating bed
    self.dSde    = self.ds(4)              # sides
    self.dBed    = self.dGnd + self.dFlt   # bed
    self.dSrf_s  = self.ds(6)              # surface
    self.dSrf_g  = self.ds(2)              # surface
    self.dSrf    = self.ds(6) + self.ds(2) # entire surface

  def deform_mesh_to_geometry(self, S, B):
    """
    Deforms the 3D mesh to the geometry from FEniCS Expressions for the 
    surface <S> and bed <B>.
    """
    s = "::: deforming mesh to geometry :::"
    print_text(s, self.model_color)
    
    # transform z :
    # thickness = surface - base, z = thickness + base
    # Get the height of the mesh, assumes that the base is at z=0
    max_height  = self.mesh.coordinates()[:,2].max()
    min_height  = self.mesh.coordinates()[:,2].min()
    mesh_height = max_height - min_height
    
    s = "    - iterating through %i vertices - " % self.dof
    print_text(s, self.model_color)
    
    for x in self.mesh.coordinates():
      x[2] = (x[2] / mesh_height) * ( + S(x[0],x[1],x[2]) \
                                      - B(x[0],x[1],x[2]) )
      x[2] = x[2] + B(x[0], x[1], x[2])
    s = "    - done - "
    print_text(s, self.model_color)

  def get_surface_mesh(self):
    """
    Returns the surface of the mesh for this model instance.
    """
    s = "::: extracting bed mesh :::"
    print_text(s, self.D3Model_color)

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
    print_text(s, self.D3Model_color)

    bmesh   = BoundaryMesh(self.mesh, 'exterior')
    cellmap = bmesh.entity_map(2)
    pb      = CellFunction("size_t", bmesh, 0)
    for c in cells(bmesh):
      if Facet(self.mesh, cellmap[c.index()]).normal().z() < -1e-3:
        pb[c] = 1
    submesh = SubMesh(bmesh, pb, 1)
    return submesh
      
  def calc_thickness(self):
    """
    Calculate the continuous thickness field which increases from 0 at the 
    surface to the actual thickness at the bed.
    """
    s = "::: calculating z-varying thickness :::"
    print_text(s, self.D3Model_color)
    H = project(self.S - self.x[2], self.Q, annotate=False)
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
    print_text(s, self.D3Model_color)
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
      bcs.append(DirichletBC(Q, u, ff, 3))  # grounded
      bcs.append(DirichletBC(Q, u, ff, 5))  # shelves
    # extrude surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, u, ff, 2))  # grounded
      bcs.append(DirichletBC(Q, u, ff, 6))  # shelves
    name = '%s extruded %s' % (u.name(), d) 
    v    = Function(Q, name=name)
    solve(a == L, v, bcs, annotate=False)
    print_min_max(u, 'function to be extruded')
    print_min_max(v, 'extruded function')
    return v
  
  def vert_integrate(self, u, d='up', Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
    s = "::: vertically integrating function :::"
    print_text(s, self.D3Model_color)

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
    name   = '%s integrated %s' % (u.name(), d) 
    v      = Function(Q, name=name)
    solve(a == L, v, bcs, annotate=False)
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
    print_text(s, self.D3Model_color)
    ubar = project(uhat/H, self.Q, annotate=False)
    print_min_max(ubar, 'ubar')
    name = "vertical average of %s" % u.name()
    ubar.rename(name, '')
    ubar = self.vert_extrude(ubar, d='down')
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
  
  def effective_stress(self):
    """
    return the effective stress squared.
    """
    tau    = self.stress_tensor(self.U3)
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
  
  def vert_integrate(self, u, d='up', Q='self'):
    """
    Integrate <u> from the bed to the surface.
    """
    raiseNotDefined()

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D3Model, self).initialize_variables()

    s = "::: initializing 3D variables :::"
    print_text(s, self.D3Model_color)

    # Depth below sea level :
    class Depth(Expression):
      def eval(self, values, x):
        values[0] = min(0, x[2])
    self.D = Depth(element=self.Q.ufl_element())
    
    # unified velocity :
    self.U3            = Function(self.V, name='U3')
    u,v,w              = self.U3.split()
    u.rename('u', '')
    v.rename('v', '')
    w.rename('w', '')
    self.u             = u
    self.v             = v
    self.w             = w
    self.p             = Function(self.Q, name='p')

    # Enthalpy model
    self.theta_surface = Function(self.Q, name='theta_surface')
    self.theta_float   = Function(self.Q, name='theta_float')
    self.theta         = Function(self.Q, name='theta')
    self.theta0        = Function(self.Q, name='theta0')
    self.W0            = Function(self.Q, name='W0')
    self.thetahat      = Function(self.Q, name='thetahat')
    self.uhat          = Function(self.Q, name='uhat')
    self.vhat          = Function(self.Q, name='vhat')
    self.what          = Function(self.Q, name='what')
    self.mhat          = Function(self.Q, name='mhat')
    self.rho_b         = Function(self.Q, name='rho_b')

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
  
  def init_age(self):
    """ 
    Set up the equations 
    """
    s    = "::: INITIALIZING AGE PHYSICS :::"
    print_text(s, self.D3Model_color)

    config = self.config
    h      = self.h
    U      = self.U3
    
    #Bub = FunctionSpace(self.mesh, "B", 4, constrained_domain=self.pBC)
    self.MQ  = self.Q# + Bub

    # Trial and test
    a   = TrialFunction(self.MQ)
    phi = TestFunction(self.MQ)
    #self.age = Function(self.MQ)

    # Steady state
    if config['mode'] == 'steady':
      s    = "    - using steady-state -"
      print_text(s, self.D3Model_color)
      
      # SUPG method :
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))
      
      # Residual 
      self.age_F = (dot(U,grad(a)) - Constant(1.0)) * phihat * dx
      self.a_a   = dot(U,grad(a)) * phi * dx
      self.a_L   = Constant(1.0) * phi * dx

    else:
      s    = "    - using transient -"
      print_text(s, self.D3Model_color)
      
      # Starting and midpoint quantities
      ahat   = self.ahat
      a0     = self.a0
      uhat   = self.uhat
      vhat   = self.vhat
      what   = self.what
      mhat   = self.mhat

      # Time step
      dt     = config['time_step']

      # SUPG method (note subtraction of mesh velocity) :
      U      = as_vector([uhat, vhat, what-mhat])
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))

      # Midpoint value of age for Crank-Nicholson
      a_mid = 0.5*(a + self.ahat)
      
      # Weak form of time dependent residual
      self.age_F = + (a - a0)/dt * phi * dx \
                   + dot(U, grad(a_mid)) * phihat * dx \
                   - 1.0 * phihat * dx

    # form the boundary conditions :
    if config['age']['use_smb_for_ela']:
      s    = "    - using adot (SMB) boundary condition -"
      print_text(s, self.D3Model_color)
      self.age_bc = DirichletBC(self.MQ, 0.0, self.ff_acc, 1)
    
    else:
      s    = "    - using ELA boundary condition -"
      print_text(s, self.D3Model_color)
      def above_ela(x,on_boundary):
        return x[2] > config['age']['ela'] and on_boundary
      self.age_bc = DirichletBC(self.Q, 0.0, above_ela)

  def solve_age(self, ahat=None, a0=None, uhat=None, what=None, vhat=None):
    """ 
    Solve the system
    
    :param ahat   : Observable estimate of the age
    :param a0     : Initial age of the ice
    :param uhat   : Horizontal velocity
    :param vhat   : Horizontal velocity perpendicular to :attr:`uhat`
    :param what   : Vertical velocity
    """
    # Assign values to midpoint quantities and mesh velocity
    if ahat:
      self.assign_variable(self.ahat, ahat)
      self.assign_variable(self.a0,   a0)
      self.assign_variable(self.uhat, uhat)
      self.assign_variable(self.vhat, vhat)
      self.assign_variable(self.what, what)

    # Solve!
    s    = "::: solving age :::"
    print_text(s, self.D3Model_color)
    solve(lhs(self.age_F) == rhs(self.age_F), self.age, self.age_bc,
          annotate=False)
    #solve(self.a_a == self.a_L, self.age, self.age_bc, annotate=False)
    #self.age.interpolate(self.age)
    print_min_max(self.age, 'age')
  
  def init_free_surface(self):
    """
    """
    s    = "::: INITIALIZING FREE-SURFACE PHYSICS :::"
    print_text(s, self.D3Model_color)
    
    # sigma coordinate :
    self.sigma = project((self.x[2] - self.B) / (self.S - self.B),
                         annotate=False)
    print_min_max(self.sigma, 'sigma')

    Q      = self.Q
    Q_flat = self.Q_flat

    phi    = TestFunction(Q_flat)
    dS     = TrialFunction(Q_flat)
    
    Shat   = Function(Q_flat) # surface elevation velocity 
    ahat   = Function(Q_flat) # accumulation velocity
    uhat   = Function(Q_flat) # horizontal velocity
    vhat   = Function(Q_flat) # horizontal velocity perp. to uhat
    what   = Function(Q_flat) # vertical velocity
    mhat   = Function(Q_flat) # mesh velocity
    dSdt   = Function(Q_flat) # surface height change
    M      = Function(Q_flat) # mass
    ds     = self.ds_flat
    dSurf  = ds(2)
    dBase  = ds(3)
    
    self.static_boundary = DirichletBC(Q, 0.0, self.ff_flat, 4)
    h = CellSize(self.flat_mesh)

    # upwinded trial function :
    unorm       = sqrt(self.uhat**2 + self.vhat**2 + 1e-1)
    upwind_term = h/(2.*unorm)*(self.uhat*phi.dx(0) + self.vhat*phi.dx(1))
    phihat      = phi + upwind_term

    mass_matrix = dS * phihat * dSurf
    lumped_mass = phi * dSurf

    stiffness_matrix = - self.uhat * self.Shat.dx(0) * phihat * dSurf \
                       - self.vhat * self.Shat.dx(1) * phihat * dSurf\
                       + (self.what + self.ahat) * phihat * dSurf
    
    # Calculate the nonlinear residual dependent scalar
    term1            = self.Shat.dx(0)**2 + self.Shat.dx(1)**2 + 1e-1
    term2            = + self.uhat*self.Shat.dx(0) \
                       + self.vhat*self.Shat.dx(1) \
                       - (self.what + self.ahat)
    C                = 10.0*h/(2*unorm) * term1 * term2**2
    diffusion_matrix = C * dot(grad(phi), grad(self.Shat)) * dSurf
    
    # Set up the Galerkin-least squares formulation of the Stokes' functional
    A_pro         = - phi.dx(2)*dS*dx - dS*phi*dBase + dSdt*phi*dSurf 
    M.vector()[:] = 1.0
    self.M        = M*dx

    self.newz                   = Function(self.Q)
    self.mass_matrix            = mass_matrix
    self.stiffness_matrix       = stiffness_matrix
    self.diffusion_matrix       = diffusion_matrix
    self.lumped_mass            = lumped_mass
    self.A_pro                  = A_pro
    self.Shat                   = Shat
    self.ahat                   = ahat
    self.uhat                   = uhat
    self.vhat                   = vhat
    self.what                   = what
    self.mhat                   = mhat
    self.dSdt                   = dSdt
    
  def solve_free_surface(self):
    """
    """
    config = self.config
   
    self.assign_variable(self.Shat, self.S) 
    self.assign_variable(self.ahat, self.adot) 
    self.assign_variable(self.uhat, self.u) 
    self.assign_variable(self.vhat, self.v) 
    self.assign_variable(self.what, self.w) 

    m = assemble(self.mass_matrix,      keep_diagonal=True)
    r = assemble(self.stiffness_matrix, keep_diagonal=True)

    s    = "::: solving free-surface :::"
    print_text(s, self.D3Model_color)
    if config['free_surface']['lump_mass_matrix']:
      m_l = assemble(self.lumped_mass)
      m_l = m_l.get_local()
      m_l[m_l==0.0]=1.0
      m_l_inv = 1./m_l

    if config['free_surface']['static_boundary_conditions']:
      self.static_boundary.apply(m,r)

    if config['free_surface']['use_shock_capturing']:
      k = assemble(self.diffusion_matrix)
      r -= k
      print_min_max(r, 'D')

    if config['free_surface']['lump_mass_matrix']:
      self.assign_variable(self.dSdt, m_l_inv * r.get_local())
    else:
      m.ident_zeros()
      solve(m, self.dSdt.vector(), r, annotate=False)

    A = assemble(lhs(self.A_pro))
    p = assemble(rhs(self.A_pro))
    q = Vector()  
    solve(A, q, p, annotate=False)
    self.assign_variable(self.dSdt, q)
 
  def thermo_solve(self, momentum, energy, callback=None, 
                   rtol=1e-6, max_iter=15):
    """ 
    Perform thermo-mechanical coupling between momentum and energy.
    """
    s    = '::: performing thermo-mechanical coupling :::'
    print_text(s, self.D3Model_color)
    
    from momentum import Momentum
    from energy   import Energy
    
    if momentum.__class__.__base__ != Momentum:
      s = ">>> thermo_solve REQUIRES A 'Momentum' INSTANCE, NOT %s <<<"
      print_text(s % type(momentum) , 'red', 1)
      sys.exit(1)
    
    if energy.__class__.__base__ != Energy:
      s = ">>> thermo_solve REQUIRES AN 'Energy' INSTANCE, NOT %s <<<"
      print_text(s % type(energy) , 'red', 1)
      sys.exit(1)

    t0   = time()

    # L_\infty norm in velocity between iterations :
    inner_error = inf
   
    # number of iterations
    counter     = 0
   
    # previous velocity for norm calculation
    U_prev      = self.U3.copy(True)

    #adj_checkpointing(strategy='multistage', steps=max_iter, 
    #                  snaps_on_disk=0, snaps_in_ram=2, verbose=True)
    
    # perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance :
    while inner_error > rtol and counter < max_iter:
     
      # need zero initial guess for Newton solve to converge : 
      self.assign_variable(momentum.get_U(),  DOLFIN_EPS)
      
      # solve velocity :
      momentum.solve(annotate=False)

      # solve energy (temperature, water content) :
      energy.solve()

      # calculate L_infinity norm, increment counter :
      #adj_inc_timestep()
      counter       += 1
      inner_error_n  = norm(project(U_prev - self.U3, annotate=False))
      U_prev         = self.U3.copy(True)
      if self.MPI_rank==0:
        s0    = '>>> '
        s1    = 'Picard iteration %i (max %i) done: ' % (counter, max_iter)
        s2    = 'r0 = %.3e'  % inner_error
        s3    = ', '
        s4    = 'r = %.3e ' % inner_error_n
        s5    = '(tol %.3e)' % rtol
        s6    = ' <<<'
        text0 = get_text(s0, 'red', 1)
        text1 = get_text(s1, self.D3Model_color)
        text2 = get_text(s2, 'red', 1)
        text3 = get_text(s3, self.D3Model_color)
        text4 = get_text(s4, 'red', 1)
        text5 = get_text(s5, self.D3Model_color)
        text6 = get_text(s6, 'red', 1)
        print text0 + text1 + text2 + text3 + text4 + text5 + text6
      inner_error = inner_error_n

      if callback != None:
        s    = '::: calling callback function :::'
        print_text(s, self.D3Model_color)
        callback()

    # calculate total time to compute
    s = time() - t0
    m = s / 60.0
    h = m / 60.0
    s = s % 60
    m = m % 60
    text = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)


