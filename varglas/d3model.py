from fenics            import *
from dolfin_adjoint    import *
from varglas.io        import print_text, get_text, print_min_max
from varglas.model_new import Model
from pylab             import inf
import sys

class D3Model(Model):
  """ 
  """

  def __init__(self, mesh, out_dir='./results/', save_state=False, 
               use_periodic=False):
    """
    Create and instance of a 3D model.
    """
    self.D3Model_color = '150'
    Model.__init__(self, mesh, out_dir, save_state, use_periodic)

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    s = "    - using 3D periodic boundaries -"
    print_text(s, self.D3Model_color)

    xmin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,0].min())
    xmax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,0].max())
    ymin = MPI.min(mpi_comm_world(), self.mesh.coordinates()[:,1].min())
    ymax = MPI.max(mpi_comm_world(), self.mesh.coordinates()[:,1].max())
    
    self.use_periodic_boundaries = True
    
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
    super(D3Model, self).set_mesh(mesh)
    
    s = "::: setting 3D mesh :::"
    print_text(s, self.D3Model_color)
    
    self.flat_mesh  = Mesh(mesh)
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
    print_text(s, self.D3Model_color)

  def generate_function_spaces(self, use_periodic=False):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D3Model, self).generate_function_spaces(use_periodic)

    s = "::: generating 3D function spaces :::"
    print_text(s, self.D3Model_color)
    
    ## mini elements :
    #self.Bub    = FunctionSpace(self.mesh, "B", 4, 
    #                            constrained_domain=self.pBC)
    #self.MQ     = self.Q + self.Bub
    #M3          = MixedFunctionSpace([self.MQ]*3)
    #self.MV     = MixedFunctionSpace([M3,self.Q])
    
    ## Taylor-Hood elements :
    #V           = VectorFunctionSpace(self.mesh, "CG", 2,
    #                                  constrained_domain=self.pBC)
    #self.MV     = V * self.Q
    
    s = "    - 3D function spaces created - "
    print_text(s, self.D3Model_color)

  def generate_stokes_function_spaces(self, kind='mini'):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.

    If <kind> == 'mini', use enriched mini elements.
    If <kind> == 'th',   use Taylor-Hood elements.
    """
    s = "::: generating Stokes function spaces :::"
    print_text(s, self.D3Model_color)
        
    # mini elements :
    if kind == 'mini':
      self.Bub    = FunctionSpace(self.mesh, "B", 4, 
                                  constrained_domain=self.pBC)
      self.MQ     = self.Q + self.Bub
      M3          = MixedFunctionSpace([self.MQ]*3)
      self.MV     = MixedFunctionSpace([M3,self.Q])

    # Taylor-Hood elements :
    elif kind == 'th':
      V           = VectorFunctionSpace(self.mesh, "CG", 2,
                                        constrained_domain=self.pBC)
      self.MV     = V * self.Q
    
    else:
      s = ">>> METHOD generate_stokes_function_spaces <kind> FIELD <<<\n" + \
          ">>> MAY BE 'mini' OR 'th', NOT '%s'. <<<" % kind
      print_text(s, 'red', 1)
      sys.exit(1)

    s = "    - Stokes function spaces created - "
    print_text(s, self.D3Model_color)
    
  def calculate_boundaries(self, mask=None, adot=None, mark_divide=False):
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
   
    self.init_adot(adot)
    self.init_mask(mask)
    
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
    
      elif n.z() <= -tol and f.exterior():
        if mask_xy > 0:
          self.ff[f] = 5
        else:
          self.ff[f] = 3
      
      elif n.z() >  -tol and n.z() < tol and f.exterior():
        if mark_divide:
          if mask_xy > 0:
            self.ff[f] = 4
          else:
            self.ff[f] = 7
        else:
          self.ff[f] = 4
    
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

    if self.save_state:
      self.state.write(self.ff,     'ff')
      self.state.write(self.ff_acc, 'ff_acc')
      self.state.write(self.cf,     'cf')
    
  def calculate_flat_mesh_boundaries(self, mask=None, adot=None,
                                     mark_divide=False):
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
        if mark_divide:
          if mask_xy > 0:
            self.ff_flat[f] = 4
          else:
            self.ff_flat[f] = 7
        else:
          self.ff_flat[f] = 4
    
    s = "    - done - "
    print_text(s, self.D3Model_color)
    
    self.ds_flat = Measure('ds')[self.ff_flat]
  
  def set_subdomains(self, ff, cf, ff_acc):
    """
    Set the facet subdomains to FacetFunction <ff>, and set the cell subdomains 
    to CellFunction <cf>, and accumulation FacetFunction to <ff_acc>.
    """
    s = "::: setting 3D subdomains :::"
    print_text(s, self.D3Model_color)
    
    self.ff     = ff
    self.cf     = cf
    self.ff_acc = ff_acc
    self.ds     = Measure('ds')[self.ff]
    self.dx     = Measure('dx')[self.cf]
    
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
    print_text(s, self.D3Model_color)

    self.init_S(S)
    self.init_B(B)
    
    # transform z :
    # thickness = surface - base, z = thickness + base
    # Get the height of the mesh, assumes that the base is at z=0
    max_height  = self.mesh.coordinates()[:,2].max()
    min_height  = self.mesh.coordinates()[:,2].min()
    mesh_height = max_height - min_height
    
    s = "    - iterating through %i vertices - " % self.dof
    print_text(s, self.D3Model_color)
    
    for x in self.mesh.coordinates():
      x[2] = (x[2] / mesh_height) * ( + S(x[0],x[1],x[2]) \
                                      - B(x[0],x[1],x[2]) )
      x[2] = x[2] + B(x[0], x[1], x[2])
    s = "    - done - "
    print_text(s, self.D3Model_color)

  def get_surface_mesh(self):
    """
    Returns the surface of the mesh for this model instance.
    """
    s = "::: extracting surface mesh :::"
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
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_GND))  # grounded
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_SHF))  # shelves
    # extrude surface (ff = 2,6) 
    elif d == 'down':
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_GND))  # grounded
      bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_SHF))  # shelves
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
    
  def deviatoric_stress_tensor(self):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
    print_text(s, self.color)
    epi = self.strain_rate_tensor()
    tau = 2*self.eta*epi
    return tau
  
  def effective_stress(self):
    """
    return the effective stress squared.
    """
    tau    = self.deviatoric_stress_tensor()
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



