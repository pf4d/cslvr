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
      # transform z :
      # thickness = surface - base, z = thickness + base
      for x in mesh.coordinates():
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
        mid     = f.midpoint()
        x_m     = mid.x()
        y_m     = mid.y()
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



