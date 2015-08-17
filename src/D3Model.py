from fenics         import *
from dolfin_adjoint import *
from io             import print_text, get_text, print_min_max
from model_new      import Model
from pylab          import inf
import sys

class D3Model(Model):
  """ 
  """

  def __init__(self, config=None):
    """
    Create and instance of the model.
    """
    Model.__init__(self, config)
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
    self.generate_function_spaces()

  def generate_function_spaces(self):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D3Model, self).generate_function_spaces()

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

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D3Model, self).initialize_variables()

    s = "::: initializing 3D variables :::"
    print_text(s, self.D3Model_color)

    config = self.config
    
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

    # Enthalpy model
    self.theta_surface = Function(self.Q, name='theta_surface')
    self.theta_float   = Function(self.Q, name='theta_float  ')
    self.theta         = Function(self.Q, name='theta        ')
    self.theta0        = Function(self.Q, name='theta0       ')
    self.W0            = Function(self.Q, name='W0           ')
    self.thetahat      = Function(self.Q, name='thetahat     ')
    self.uhat          = Function(self.Q, name='uhat         ')
    self.vhat          = Function(self.Q, name='vhat         ')
    self.what          = Function(self.Q, name='what         ')
    self.mhat          = Function(self.Q, name='mhat         ')

    # Age model   
    self.age           = Function(self.Q, name='age')
    self.a0            = Function(self.Q, name='a0')

    # Surface climate model
    self.precip        = Function(self.Q, name='precip')

    # Stokes-balance model :
    self.u_s           = Function(self.Q, name='')
    self.u_t           = Function(self.Q, name='')
    self.F_id          = Function(self.Q, name='')
    self.F_jd          = Function(self.Q, name='')
    self.F_ib          = Function(self.Q, name='')
    self.F_jb          = Function(self.Q, name='')
    self.F_ip          = Function(self.Q, name='')
    self.F_jp          = Function(self.Q, name='')
    self.F_ii          = Function(self.Q, name='')
    self.F_ij          = Function(self.Q, name='')
    self.F_iz          = Function(self.Q, name='')
    self.F_ji          = Function(self.Q, name='')
    self.F_jj          = Function(self.Q, name='')
    self.F_jz          = Function(self.Q, name='')
    self.tau_iz        = Function(self.Q, name='')
    self.tau_jz        = Function(self.Q, name='')
  
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
    config = self.config

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
  
  def solve_surface_climate(self):
    """
    Calculates PDD, surface temperature given current model geometry

    """
    s    = "::: solving surface climate :::"
    print_text(s, self.D3Model_color)
    config = self.config

    T_ma  = config['surface_climate']['T_ma']
    T_w   = self.T_w
    S     = self.S.vector().array()
    lat   = self.lat.vector().array()
    
    # Apply the lapse rate to the surface boundary condition
    self.assign_variable(self.T_surface, T_ma(S, lat) + T_w)
    
  def calc_T_melt(self):
    """
    Calculates pressure-melting point in self.T_melt.
    """
    s    = "::: calculating pressure-melting temperature :::"
    print_text(s, self.D3Model_color)

    dx  = self.dx
    x   = self.x
    S   = self.S
    g   = self.gamma
    T_w = self.T_w

    u   = TrialFunction(self.Q)
    phi = TestFunction(self.Q)

    l = assemble((T_w - g * (S - x[2])) * phi * dx)
    a = assemble(u * phi * dx)

    solve(a, self.T_melt.vector(), l, annotate=False)
    print_min_max(self.T_melt, 'T_melt')
  
  def init_energy(self):
    """ 
    Set up energy equation residual. 
    """
    s    = "::: INITIALIZING ENTHALPY PHYSICS :::"
    print_text(s, self.D3Model_color)

    config      = self.config

    r           = config['velocity']['r']
    mesh        = self.mesh
    V           = self.V
    Q           = self.Q
    theta       = self.theta
    theta0      = self.theta0
    n           = self.n
    eta_gnd     = self.eta_gnd
    eta_shf     = self.eta_shf
    T           = self.T
    T_melt      = self.T_melt
    Mb          = self.Mb
    L           = self.L
    ci          = self.ci
    cw          = self.cw
    T_w         = self.T_w
    gamma       = self.gamma
    S           = self.S
    B           = self.B
    H           = S - B
    x           = self.x
    W           = self.W
    R           = self.R
    U           = self.U3
    u           = self.u
    v           = self.v
    w           = self.w
    epsdot      = self.epsdot
    eps_reg     = self.eps_reg
    rhoi        = self.rhoi
    rhow        = self.rhow
    g           = self.g
    beta        = self.beta
    kappa       = self.kappa
    ki          = self.ki
    kw          = self.kw
    T_surface   = self.T_surface
    theta_surface = self.theta_surface
    theta_float   = self.theta_float
    q_geo         = self.q_geo
    thetahat      = self.thetahat
    uhat        = self.uhat
    vhat        = self.vhat
    what        = self.what
    mhat        = self.mhat
    spy         = self.spy
    h           = self.h
    ds          = self.ds
    dSrf        = self.dSrf
    dGnd        = self.dGnd
    dFlt        = self.dFlt
    dSde        = self.dSde
    dBed        = self.dBed
    dx          = self.dx
    dx_s        = self.dx_s
    dx_g        = self.dx_g
    
    # Define test and trial functions       
    psi    = TestFunction(Q)
    dtheta = TrialFunction(Q)

    # Pressure melting point
    self.calc_T_melt()

    T_s_v = T_surface.vector().array()
    T_m_v = T_melt.vector().array()
   
    # Surface boundary condition :
    s = "::: calculating energy boundary conditions :::"
    print_text(s, self.D3Model_color)

    self.assign_variable(theta_surface, T_s_v * ci)
    self.assign_variable(theta_float,   T_m_v * ci)
    print_min_max(theta_surface, 'theta_GAMMA_S')
    print_min_max(theta_float,   'theta_GAMMA_B_SHF')

    # For the following heat sources, note that they differ from the 
    # oft-published expressions, in that they are both multiplied by constants.
    # I think that this is the correct form, as they must be this way in order 
    # to conserve energy.  This also implies that heretofore, models have been 
    # overestimating frictional heat, and underestimating strain heat.

    # Frictional heating :
    q_friction = 0.5 * beta**2 * inner(U,U)

    # Strain heating = stress*strain
    Q_s_gnd = self.Vd_gnd
    Q_s_shf = self.Vd_shf

    # thermal conductivity (Greve and Blatter 2009) :
    ki    =  9.828 * exp(-0.0057*T)
    
    # bulk properties :
    k     =  (1 - W)*ki   + W*kw     # bulk thermal conductivity
    c     =  (1 - W)*ci   + W*cw     # bulk heat capacity
    rho   =  (1 - W)*rhoi + W*rhow   # bulk density
    kappa =  k / (rho*c)             # bulk thermal diffusivity

    # configure the module to run in steady state :
    if config['mode'] == 'steady':
      #epi     = 0.5 * (grad(U) + grad(U).T)
      #ep_xx   = epi[0,0]
      #ep_yy   = epi[1,1]
      #ep_zz   = epi[2,2]
      #ep_xy   = epi[0,1]
      #ep_xz   = epi[0,2]
      #ep_yz   = epi[1,2]
      #epsdot  = 0.5 * (+ ep_xx**2 + ep_yy**2 + ep_zz**2) \
      #                 + ep_xy**2 + ep_xz**2 + ep_yz**2
      #Q_s_gnd = 2 * eta_gnd * tr(dot(epi,epi))
      #Q_s_shf = 2 * eta_shf * tr(dot(epi,epi))
      #Q_s_gnd = 4 * eta_gnd * epsdot
      #Q_s_shf = 4 * eta_shf * epsdot

      # skewed test function in areas with high velocity :
      Unorm  = sqrt(dot(U, U) + DOLFIN_EPS)
      PE     = Unorm*h/(2*kappa)
      tau    = 1/tanh(PE) - 1/PE
      #T_c    = conditional( lt(Unorm, 4), 0.0, 1.0 )
      psihat = psi + h*tau/(2*Unorm) * dot(U, grad(psi))

      # residual of model :
      theta_a = + rho * dot(U, grad(dtheta)) * psihat * dx \
                + rho * spy * kappa * dot(grad(psi), grad(dtheta)) * dx \
      
      theta_L = + (q_geo + q_friction) * psihat * dGnd \
                + Q_s_gnd * psihat * dx_g \
                + Q_s_shf * psihat * dx_s
      
    # configure the module to run in transient mode :
    elif config['mode'] == 'transient':
      dt = config['time_step']
    
      epi     = 0.5 * (grad(U) + grad(U).T)
      ep_xx   = epi[0,0]
      ep_yy   = epi[1,1]
      ep_zz   = epi[2,2]
      ep_xy   = epi[0,1]
      ep_xz   = epi[0,2]
      ep_yz   = epi[1,2]
      epsdot  = 0.5 * (+ ep_xx**2 + ep_yy**2 + ep_zz**2) \
                       + ep_xy**2 + ep_xz**2 + ep_yz**2
      #Q_s_gnd = 2 * eta_gnd * tr(dot(epi,epi))
      #Q_s_shf = 2 * eta_shf * tr(dot(epi,epi))
      Q_s_gnd = 4 * eta_gnd * epsdot
      Q_s_shf = 4 * eta_shf * epsdot

      # Skewed test function.  Note that vertical velocity has 
      # the mesh velocity subtracted from it.
      Unorm  = sqrt(dot(U, U) + 1.0)
      PE     = Unorm*h/(2*kappa)
      tau    = 1/tanh(PE) - 1/PE
      #T_c    = conditional( lt(Unorm, 4), 0.0, 1.0 )
      psihat = psi + h*tau/(2*Unorm) * dot(U, grad(psi))

      nu = 0.5
      # Crank Nicholson method
      thetamid = nu*dtheta + (1 - nu)*theta0
      
      # implicit system (linearized) for energy at time theta_{n+1}
      theta_a = + rho * (dtheta - theta0) / dt * psi * dx \
                + rho * dot(U, grad(thetamid)) * psihat * dx \
                + rho * spy * kappa * dot(grad(psi), grad(thetamid)) * dx \
      
      theta_L = + (q_geo + q_friction) * psi * dGnd \
                + Q_s_gnd * psihat * dx_g \
                + Q_s_shf * psihat * dx_s

    self.theta_a = theta_a
    self.theta_L = theta_L
    
    # surface boundary condition : 
    self.theta_bc = []
    self.theta_bc.append( DirichletBC(Q, theta_surface, self.ff, 2) )
    self.theta_bc.append( DirichletBC(Q, theta_surface, self.ff, 6) )
    
    # apply T_w conditions of portion of ice in contact with water :
    self.theta_bc.append( DirichletBC(Q, theta_float,   self.ff, 5) )
    
    # apply lateral boundaries if desired : 
    if config['energy']['lateral_boundaries'] == 'surface':
      self.theta_bc.append( DirichletBC(Q, theta_surface, self.ff, 4) )

    self.c          = c
    self.k          = k
    self.rho        = rho
    self.kappa      = kappa
    self.q_friction = q_friction
     
  
  def solve_energy(self):
    """ 
    """
    config = self.config

    mesh       = self.mesh
    Q          = self.Q
    T_melt     = self.T_melt
    theta      = self.theta
    T          = self.T
    W          = self.W
    W0         = self.W0
    L          = self.L
    ci         = self.ci
    
    # solve the linear equation for energy :
    s    = "::: solving energy :::"
    print_text(s, self.D3Model_color)
    sm = config['energy']['solve_method']
    aw        = assemble(self.theta_a)
    Lw        = assemble(self.theta_L)
    for bc in self.theta_bc:
      bc.apply(aw, Lw)
    theta_solver = LUSolver(sm)
    theta_solver.solve(aw, theta.vector(), Lw, annotate=False)
    #solve(self.theta_a == self.theta_L, theta, self.theta_bc,
    #      solver_parameters = {"linear_solver" : sm}, annotate=False)
    print_min_max(theta, 'theta')

    # temperature solved diagnostically : 
    s = "::: calculating temperature :::"
    print_text(s, self.D3Model_color)
    T_n  = project(theta/ci, Q, annotate=False)
    
    # update temperature for wet/dry areas :
    T_n_v        = T_n.vector().array()
    T_melt_v     = T_melt.vector().array()
    warm         = T_n_v >= T_melt_v
    cold         = T_n_v <  T_melt_v
    T_n_v[warm]  = T_melt_v[warm]
    self.assign_variable(T, T_n_v)
    print_min_max(T,  'T')
    
    # water content solved diagnostically :
    s = "::: calculating water content :::"
    print_text(s, self.D3Model_color)
    W_n  = project((theta - ci*T_melt)/L, Q, annotate=False)
    
    # update water content :
    W_v             = W_n.vector().array()
    W_v[cold]       = 0.0
    W_v[W_v < 0.0]  = 0.0
    W_v[W_v > 0.01] = 0.01  # for rheology; instant water run-off
    self.assign_variable(W0, W)
    self.assign_variable(W,  W_v)
    print_min_max(W,  'W')
   
  def solve_basal_melt_rate(self):
    """
    """ 
    # calculate melt-rate : 
    s = "::: solving for basal melt-rate :::"
    print_text(s, self.D3Model_color)
    
    B          = self.B
    rho        = self.rho
    rhoi       = self.rhoi
    theta      = self.theta
    kappa      = self.kappa
    L          = self.L
    q_geo      = self.q_geo
    q_friction = self.q_friction

    gradB = as_vector([B.dx(0), B.dx(1), -1])
    dHdn  = rho * kappa * dot(grad(theta), gradB)
    nMb   = project((q_geo + q_friction - dHdn) / (L*rhoi), self.Q,
                    annotate=False)
    nMb_v = nMb.vector().array()
    #nMb_v[nMb_v < 0.0]  = 0.0
    #nMb_v[nMb_v > 10.0] = 10.0
    self.assign_variable(self.Mb, nMb_v)
    print_min_max(self.Mb, 'Mb')

  def calc_bulk_density(self):
    """
    """
    # calculate bulk density :
    s = "::: calculating bulk density :::"
    print_text(s, self.D3Model_color)
    rho_b      = project(self.rho, annotate=False)
    self.rho_b = rho_b
    print_min_max(rho_b,'rho_b')
  
  def init_free_surface(self):
    """
    """
    s    = "::: INITIALIZING FREE-SURFACE PHYSICS :::"
    print_text(s, self.D3Model_color)
    
    # sigma coordinate :
    self.sigma = project((self.x[2] - self.B) / (self.S - self.B),
                         annotate=False)
    print_min_max(self.sigma, 'sigma')

    config = self.config
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
  
  def get_obj_ftn(self, kind='log', integral=2, g1=0.01, g2=1000):
    """
    Returns an objective functional for use with adjoint.
    """
    dGamma   = self.ds(integral)
    u_ob     = self.u_ob
    v_ob     = self.v_ob
    adot     = self.adot
    S        = self.S
    U        = self.U

    if kind == 'log':
      J   = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                      / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dGamma 
      s   = "::: forming log objective function :::"
    
    elif kind == 'kinematic':
      J   = 0.5 * (+ U[0]*S.dx(0) + U[1]*S.dx(1) - (U[2] + adot))**2 * dGamma
      s   = "::: getting kinematic objective function :::"

    elif kind == 'linear':
      J   = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dGamma
      s   = "::: getting linear objective function :::"
    
    elif kind == 'log_lin_hybrid':
      J1  = g1 * 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dGamma
      J2  = g2 * 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dGamma
      J   = J1  + J2
      s   = "::: getting log/linear hybrid objective with gamma_1 = " \
            "%.1e and gamma_2 = %.1e :::" % (g1, g2)

    else:
      s = ">>> ADJOINT OBJECTION FUNCTION MAY BE 'linear', " + \
          "'log', 'kinematic', OR 'log_lin_hybrid' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    print_text(s, self.D3Model_color)
    return J

  def get_reg_ftn(self, c, kind='Tikhonov', integral=2, alpha=1.0):
    """
    Returns a regularization functional for used with adjoint.
    """
    dR = self.ds(integral)
    
    # form regularization term 'R' :
    if kind != 'TV' and kind != 'Tikhonov':
      s    =   ">>> VALID REGULARIZATIONS ARE 'TV' AND 'Tikhonov' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    if kind == 'TV':
      R  = alpha * 0.5 * sqrt(inner(grad(c), grad(c)) + DOLFIN_EPS) * dR
      Rp = 0.5 * sqrt(inner(grad(c), grad(c)) + DOLFIN_EPS) * dR
    elif kind == 'Tikhonov':
      R  = alpha * 0.5 * inner(grad(c), grad(c)) * dR
      Rp = 0.5 * inner(grad(c), grad(c)) * dR
    s   = "::: forming %s regularization with parameter alpha = %.2E :::"
    print_text(s % (kind, alpha), self.D3Model_color)
    return R
    
  def Lagrangian(self):
    """
    Returns the Lagrangian of the momentum equations.
    """
    # this is the adjoint of the momentum residual, the Lagrangian :
    return replace(self.mom_F, {self.Phi:self.dU})

  def Hamiltonian(self, I):
    """
    Returns the Hamiltonian of the momentum equations with objective function
    <I>.
    """
    # the Hamiltonian :
    return I + self.Lagrangian()

  def dHdc(self, I, L, c): 
    """
    Returns the derivative of the Hamiltonian consisting of ajoint-computed
    self.Lam values w.r.t. the control variable <c>, i.e., 

       dH    d [                 ]
       -- = -- [ I + L(self.Lam) ]
       dc   dc [                 ]

    """
    # we need to evaluate the Hamiltonian with the values of Lam computed from
    # self.dI in order to get the derivative of the Hamiltonian w.r.t. the 
    # control variables.  Hence we need a new Lagrangian with the trial 
    # functions replaced with the computed Lam values.
    L_lam  = replace(L, {self.dU : self.Lam})

    # the Hamiltonian with unknowns replaced with computed Lam :
    H_lam  = I + L_lam

    # the derivative of the Hamiltonian w.r.t. the control variables in the 
    # direction of a P1 test function :
    return derivative(H_lam, c, TestFunction(self.Q))
    
  def solve_adjoint_momentum(self, H):
    """
    Solves for the adjoint variables self.Lam from the Hamiltonian <H>.
    """
    config = self.config
    
    # we desire the derivative of the Hamiltonian w.r.t. the model state U
    # in the direction of the test function Phi to vanish :
    dI = derivative(H, self.U, self.Phi)
    
    s  = "::: solving adjoint momentum :::"
    print_text(s, self.D3Model_color)
    
    aw = assemble(lhs(dI))
    Lw = assemble(rhs(dI))
    
    a_solver = KrylovSolver('cg', 'hypre_amg')
    a_solver.solve(aw, self.Lam.vector(), Lw, annotate=False)

    #lam_nx, lam_ny = model.Lam.split(True)
    #lam_ix, lam_iy = model.Lam.split()

    #if config['adjoint']['surface_integral'] == 'shelves':
    #  lam_nx.vector()[model.gnd_dofs] = 0.0
    #  lam_ny.vector()[model.gnd_dofs] = 0.0
    #elif config['adjoint']['surface_integral'] == 'grounded':
    #  lam_nx.vector()[model.shf_dofs] = 0.0
    #  lam_ny.vector()[model.shf_dofs] = 0.0

    ## function assigner translates between mixed space and P1 space :
    #U_sp = model.U.function_space()
    #assx = FunctionAssigner(U_sp.sub(0), lam_nx.function_space())
    #assy = FunctionAssigner(U_sp.sub(1), lam_ny.function_space())

    #assx.assign(lam_ix, lam_nx)
    #assy.assign(lam_iy, lam_ny)
    
    #solve(self.aw == self.Lw, model.Lam,
    #      solver_parameters = {"linear_solver"  : "cg",
    #                           "preconditioner" : "hypre_amg"},
    #      annotate=False)
    #print_min_max(norm(model.Lam), '||Lam||')
    print_min_max(self.Lam, 'Lam')
    
  def thermo_solve(self, rtol=1e-6, max_iter=15):
    """ 
    Solve the problem using a Picard iteration, evaluating the velocity,
    energy, surface mass balance, temperature boundary condition, and
    the age equation.  Turn off any solver by editing the appropriate config
    dict entry to "False".  If config['coupled']['on'] is "False", solve only
    once.
    """
    s    = '::: performing thermo-mechanical coupling :::'
    print_text(s, self.D3Model_color)

    t0   = time()

    config  = self.config
    
    # L_\infty norm in velocity between iterations :
    inner_error = inf
   
    # number of iterations
    counter     = 0
   
    # previous velocity for norm calculation
    U_prev      = self.U3.copy(True)

    adj_checkpointing(strategy='multistage', steps=max_iter, 
                      snaps_on_disk=0, snaps_in_ram=2, verbose=True)
    
    # perform a Picard iteration until the L_\infty norm of the velocity 
    # difference is less than tolerance :
    while inner_error > rtol and counter < max_iter:
     
      # need zero initial guess for Newton solve to converge : 
      self.assign_variable(self.U,  DOLFIN_EPS)
      
      # solve velocity :
      self.solve_momentum()
      if config['velocity']['solve_vert_velocity']:
        self.solve_vert_velocity()
      if config['velocity']['calc_pressure']:
        self.calc_pressure()

      if config['velocity']['log'] and config['log']:
        self.save_pvd(self.U3, 'U')
        # save pressure if desired :
        if config['velocity']['calc_pressure']:
          self.save_pvd(self.p, 'p')
      
      # solve surface mass balance and temperature boundary condition :
      if config['surface_climate']['on']:
        self.solve_surface_climate()

      # solve energy (temperature, water content) :
      self.solve_energy()
      if config['energy']['log'] and config['log']: 
        self.save_pvd(self.theta, 'theta')
        self.save_pvd(self.T,     'T')
        self.save_pvd(self.W,     'W')
    
      # re-compute the friction field :
      if config['velocity']['transient_beta'] == 'stats':
        s    = "::: updating statistical beta :::"
        print_text(s, self.D3Model_color)
        beta   = project(self.beta_f, self.Q, annotate=False)
        beta_v = beta.vector().array()
        ##betaSIA_v = self.betaSIA.vector().array()
        ##beta_v[beta_v < 10.0]   = betaSIA_v[beta_v < 10.0]
        beta_v[beta_v < 0.0]    = 0.0
        #beta_v[beta_v > 2500.0] = 2500.0
        self.assign_variable(self.beta, beta_v)
        print_min_max(self.beta, 'beta')
        if config['log']:
          self.save_pvd(self.beta, 'beta')

      # calculate L_infinity norm, increment counter :
      adj_inc_timestep()
      counter       += 1
      inner_error_n  = norm(project(U_prev - self.U3, annotate=False))
      U_prev         = self.U3.copy(True)
      if self.MPI_rank==0:
        s1    = 'Picard iteration %i (max %i) done: ' % (counter, max_iter)
        s2    = 'r0 = %.3e'  % inner_error
        s3    = ', '
        s4    = 'r = %.3e ' % inner_error_n
        s5    = '(tol %.3e)' % rtol
        text1 = get_text(s1, self.D3Model_color)
        text2 = get_text(s2, 'red', 1)
        text3 = get_text(s3, self.D3Model_color)
        text4 = get_text(s4, 'red', 1)
        text5 = get_text(s5, self.D3Model_color)
        print text1 + text2 + text3 + text4 + text5
      inner_error = inner_error_n

    # calculate total time to compute
    s = time() - t0
    m = s / 60.0
    h = m / 60.0
    s = s % 60
    m = m % 60
    text = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)


