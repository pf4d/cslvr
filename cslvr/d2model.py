from dolfin            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import print_text, get_text, print_min_max
from cslvr.model       import Model
from cslvr.helper      import Boundary, VerticalBasis
import numpy               as np
import sys

class D2Model(Model):
  """ 
  """
  OMEGA_GND   = 0   # internal cells over bedrock
  OMEGA_FLT   = 1   # internal cells over water
  OMEGA_ACC   = 1   # internal cells with positive accumulation on surface
  OMEGA_U_GND = 8   # grounded surface with U observations
  OMEGA_U_FLT = 9   # shelf surface with U observations
  GAMMA_L_DVD = 7   # basin divides
  GAMMA_L_GND = 0   # terminus not in contact with water
  GAMMA_L_FLT = 1   # terminus in contact with water
  
  # external boundaries :
  ext_b = {GAMMA_L_DVD : 'interior lateral suface',
           GAMMA_L_FLT : 'exterior lateral surface in contact with water',
           GAMMA_L_GND : 'exterior lateral surface not in contact with water'},

  # internal boundaries :
  int_b = {OMEGA_GND   : 'interior over bedrock',
           OMEGA_FLT   : 'interior over water',
           OMEGA_ACC   : 'interior cells with positive accumulation',
           OMEGA_U_FLT : 'interior over water with U observations',
           OMEGA_U_GND : 'interior over bedrock with U observations'}
  
  # union :
  boundaries = {'OMEGA' : int_b,
                'GAMMA' : ext_b}

  def __init__(self, mesh, out_dir='./results/', order=1,
               use_periodic=False, kind='submesh'):
    """
    Create and instance of a 2D model.
    """
    s = "::: INITIALIZING 2D MODEL OF TYPE '%s' :::" % kind
    print_text(s, cls=self)

    self.kind = kind
    
    Model.__init__(self, mesh, out_dir, order, use_periodic)
   
    # default integration measures : 
    self.dOmega = dx#Boundary(dx, [0], 'entire interior')
    self.dGamma = ds#Boundary(ds, [0], 'entire exterior')
  
  def color(self):
    return '150'

  def generate_pbc(self):
    """
    return a SubDomain of periodic lateral boundaries.
    """
    s = "    - using 2D periodic boundaries -"
    print_text(s, cls=self)

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
        elif near(x[0], xmax):
          y[0] = x[0] - xmax
          y[1] = x[1]
        elif near(x[1], ymax):
          y[0] = x[0]
          y[1] = x[1] - ymax
        else:
          y[0] = x[0]
          y[1] = x[1]

    self.pBC = PeriodicBoundary()
  
  def set_mesh(self, mesh):
    """
    Sets the mesh.
    
    :param mesh : Dolfin mesh to be written
    """
    super(D2Model, self).set_mesh(mesh)
    
    s = "::: setting 2D mesh :::"
    print_text(s, cls=self)
    
    if self.dim != 2:
      s = ">>> 2D MODEL REQUIRES A 2D MESH, EXITING <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    else:
      self.num_edges    = self.mesh.num_edges()
      self.num_cells    = self.mesh.num_cells()
      self.num_vertices = self.mesh.num_vertices()
    s = "    - %iD mesh set, %i cells, %i edges, %i vertices - " \
        % (self.dim, self.num_cells, self.num_edges, self.num_vertices)
    print_text(s, cls=self)

  def generate_function_spaces(self, **kwargs):
    """
    Generates the appropriate finite-element function spaces from parameters
    specified in the config file for the model.
    """
    super(D2Model, self).generate_function_spaces()

    s = "::: generating 2D function spaces :::"
    print_text(s, cls=self)

    if self.kind == 'submesh':
      pass

    elif self.kind == 'balance' and self.use_periodic:
      self.QTH2             = FunctionSpace(self.mesh, self.QTH2e,
                                            constrained_domain=self.pBC)
      self.BDM              = FunctionSpace(self.mesh, self.BDMMe,
                                            constrained_domain=self.pBC)
      self.DG1              = FunctionSpace(self.mesh, self.DG1e,
                                            constrained_domain=self.pBC)
    elif self.kind == 'balance' and not self.use_periodic:
      self.QTH2             = FunctionSpace(self.mesh, self.QTH2e)
      self.BDM              = FunctionSpace(self.mesh, self.BDMMe)
      self.DG1              = FunctionSpace(self.mesh, self.DG1e)
    elif self.kind == 'hybrid':
      # default values if not provided : 
      self.poly_deg = kwargs.get('poly_deg', 2)
      self.N_T      = kwargs.get('N_T',      8)

      # velocity space :
      self.HVe  = MixedElement([self.Q1e]*2*self.poly_deg)
      self.HV   = FunctionSpace(self.mesh, self.HVe)
     
      # temperature space :
      self.Ze   = MixedElement([self.Q1e]*self.N_T)
      self.Z    = FunctionSpace(self.mesh, self.Ze)   
    
    s = "    - 2D function spaces created - "
    print_text(s, cls=self)

  def calculate_boundaries(self,
                           mask        = None,
                           adot        = None,
                           U_mask      = None,
                           mark_divide = False,
                           contour     = None):
    """
    Determines the boundaries of the current model mesh
    """
    s = "::: calculating boundaries :::"
    print_text(s, cls=self)

    if    (contour is     None and     mark_divide) \
       or (contour is not None and not mark_divide):
      s = ">>> IF PARAMETER <mark_divide> OF calculate_boundaries() IS " + \
          "TRUE, PARAMETER <contour> MUST BE A NUMPY ARRAY OF COORDINATES " + \
          "OF THE ICE-SHEET EXTERIOR BOUNDARY <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    elif contour is not None and mark_divide:
      s = "    - marking the interior facets for incomplete meshes -"
      print_text(s, cls=self)
      tree = cKDTree(contour)
     
    # this function contains markers which may be applied to facets of the mesh
    self.ff      = MeshFunction('size_t',  self.mesh, 1)
    self.ff_acc  = MeshFunction('size_t',  self.mesh, 1)
    self.cf      = MeshFunction('size_t',  self.mesh, 2)
    dofmap       = self.Q.dofmap()

    S = self.S
    B = self.B
    
    # default to all grounded ice :
    if mask == None:
      mask = Expression('1.0', element=self.Q.ufl_element())
    
    # default to all positive accumulation :
    if adot == None:
      adot = Expression('1.0', element=self.Q.ufl_element())
    
    # default to U observations everywhere :
    if U_mask == None:
      U_mask = Expression('1.0', element=self.Q.ufl_element())

    self.init_adot(adot)
    self.init_mask(mask)
    self.init_U_mask(U_mask)

    self.mask.set_allow_extrapolation(True)
    self.adot.set_allow_extrapolation(True)
    self.U_mask.set_allow_extrapolation(True)
    
    tol = 1e-6
    
    s = "    - marking boundaries - "
    print_text(s, cls=self)
    
    s = "    - iterating through %i edges - " % self.num_edges
    print_text(s, cls=self)
    for f in facets(self.mesh):
      x_m      = f.midpoint().x()
      y_m      = f.midpoint().y()
      mask_xy  = mask(x_m, y_m)
      
      if f.exterior():
        # if we want to use a basin, we need to mark the interior facets :
        if mark_divide:
          if tree.query((x_m, y_m))[0] > 1000:
            self.ff[f] = self.GAMMA_L_DVD
          else:
            if mask_xy > 1:
              self.ff[f] = self.GAMMA_L_FLT
            else:
              self.ff[f] = self.GAMMA_L_GND
        # otherwise just mark for in contact with or without water :
        else:
          if mask_xy > 1:
            self.ff[f] = self.GAMMA_L_FLT
          else:
            self.ff[f] = self.GAMMA_L_GND
    s = "    - done - "
    print_text(s, cls=self)
    
    s = "    - iterating through %i cells - " % self.num_cells
    print_text(s, cls=self)
    for c in cells(self.mesh):
      x_m       = c.midpoint().x()
      y_m       = c.midpoint().y()
      mask_xy   = mask(x_m, y_m)
      U_mask_xy = U_mask(x_m, y_m)
      adot_xy   = adot(x_m, y_m)

      if adot_xy > 0:
        self.ff_acc[c] = self.OMEGA_ACC
      if mask_xy > 1:
        if U_mask_xy > 0:
          self.cf[c] = self.OMEGA_U_FLT
        else:
          self.cf[c] = self.OMEGA_FLT
      else:
        if U_mask_xy > 0:
          self.cf[c] = self.OMEGA_U_GND
        else:
          self.cf[c] = self.OMEGA_GND
    s = "    - done - "
    print_text(s, cls=self)
    
    self.set_measures()

  def set_measures(self):
    """
    set the new measure space for facets ``self.ds`` and cells ``self.dx`` for
    the boundaries marked by FacetFunction ``self.ff`` and CellFunction 
    ``self.cf``, respectively.

    Also, the number of cells or facets marked by 
    :func:`calculate_boundaries` :

    * ``self.N_OMEGA_GND``   -- number of cells marked ``self.OMEGA_GND``  
    * ``self.N_OMEGA_FLT``   -- number of cells marked ``self.OMEGA_FLT``  
    * ``self.N_OMEGA_U_GND`` -- number of cells marked ``self.OMEGA_U_GND``
    * ``self.N_OMEGA_U_FLT`` -- number of cells marked ``self.OMEGA_U_FLT``
    * ``self.N_GAMMA_L_DVD`` -- number of facets marked ``self.GAMMA_L_DVD``

    The subdomain corresponding to FacetFunction ``self.ff`` is :

    * ``self.dLat_d``  --  lateral divide

    The subdomains corresponding to CellFunction ``self.cf`` are :

    * ``self.dx_g``    --  internal above grounded
    * ``self.dx_f``    --  internal above floating
    * ``self.dx_u_G``  --  grounded with U observations
    * ``self.dx_u_f``  --  floating with U observations
    """
    # calculate the number of cells and facets that are of a certain type
    # for determining Dirichlet boundaries :
    self.N_OMEGA_GND   = sum(self.cf.array() == self.OMEGA_GND)
    self.N_OMEGA_FLT   = sum(self.cf.array() == self.OMEGA_FLT)
    self.N_OMEGA_U_GND = sum(self.cf.array() == self.OMEGA_U_GND)
    self.N_OMEGA_U_FLT = sum(self.cf.array() == self.OMEGA_U_FLT)
    self.N_GAMMA_L_DVD = sum(self.ff.array() == self.GAMMA_L_DVD)
    self.N_GAMMA_L_GND = sum(self.ff.array() == self.GAMMA_L_GND)
    self.N_GAMMA_L_FLT = sum(self.ff.array() == self.GAMMA_L_FLT)

    # create new measures of integration :
    self.ds      = Measure('ds', subdomain_data=self.ff)
    self.dx      = Measure('dx', subdomain_data=self.cf)
    
    self.dx_g    = self.dx(self.OMEGA_GND)   # above grounded
    self.dx_f    = self.dx(self.OMEGA_FLT)   # above floating
    self.dx_u_g  = self.dx(self.OMEGA_U_GND) # above grounded with U
    self.dx_u_f  = self.dx(self.OMEGA_U_FLT) # above floating with U
    self.dLat_d  = self.ds(self.GAMMA_L_DVD) # lateral divide
    
    self.dOmega     = Boundary(self.dx,
                      [self.OMEGA_GND,   self.OMEGA_FLT,
                       self.OMEGA_U_GND, self.OMEGA_U_FLT],
                      'entire interior')
    self.dOmega_g   = Boundary(self.dx, [self.OMEGA_GND, self.OMEGA_U_GND],
                      'interior above grounded ice')
    self.dOmega_w   = Boundary(self.dx, [self.OMEGA_FLT, self.OMEGA_U_FLT],
                      'interior above floating ice')
    self.dOmega_gu  = Boundary(self.dx, [self.OMEGA_U_GND],
                      'interior with U observations above grounded ice')
    self.dOmega_wu  = Boundary(self.dx, [self.OMEGA_U_FLT],
                      'interior with U observations above floating ice')
    self.dOmega_u   = Boundary(self.dx, [self.OMEGA_U_FLT, self.OMEGA_U_GND],
                      'entire interior with U observations')
    self.dGamma     = Boundary(self.ds,
                      [self.GAMMA_L_DVD, self.GAMMA_L_GND, self.GAMMA_L_FLT],
                      'entire exterior')
    self.dGamma_dl  = Boundary(self.ds, [self.GAMMA_L_DVD],
                      'lateral interior surface')
    self.dGamma_wl  = Boundary(self.ds, [self.GAMMA_L_FLT],
                      'lateral exterior surface in contact with water')
    self.dGamma_gl  = Boundary(self.ds, [self.GAMMA_L_GND],
                      'lateral exterior surface not in contact with water')

  def init_T_T0(self, T):
    """
    """
    s = "::: initializing temperature in model.Q space to model.Z space :::"
    print_text(s, cls=self)
    T = project(as_vector([T]*self.N_T), self.Z, annotate=False)
    self.assign_variable(self.T_,  T)
    self.assign_variable(self.T0_, T)
    
  def init_H_H0(self, H):
    """
    """
    s = "::: initializing thickness :::"
    print_text(s, cls=self)
    self.assign_variable(self.H,  H)
    self.assign_variable(self.H0, H)

  def init_H_bounds(self, H_min, H_max):
    """
    """
    s = "::: initializing bounds on thickness :::"
    print_text(s, cls=self)
    self.assign_variable(self.H_min, H_min)
    self.assign_variable(self.H_max, H_max)

  def init_U(self, U):
    """
    """
    if self.kind == 'balance':
      super(D2Model, self).init_U(U)
    elif self.kind == 'hybrid':
      # NOTE: this overides model.init_U
      s = "::: initializing velocity :::"
      print_text(s, cls=self)
      self.assign_variable(self.U3, U)

  def solve_hydrostatic_pressure(self, annotate=False):
    r"""
    Solve for the hydrostatic pressure :math:`p = f_c = \rho g (S - z)` to 
    ``self.p``, with surface height :math:`S` given by ``self.S``, ice 
    density :math:`\rho` given by ``self.rho``, and :math:`z`-coordinate
    given by ``self.x[2]``.

    :param annotate: allow Dolfin-Adjoint annotation of this procedure.
    :type annotate: bool
    """
    rhoi = self.rhoi(0)
    g    = self.g(0)
    H    = self.S.vector().get_local() - self.B.vector().get_local()
    self.assign_variable(self.p, rhoi*g*H)
    
  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(D2Model, self).initialize_variables()

    s = "::: initializing 2D variables :::"
    print_text(s, cls=self)
    
    if   self.kind == 'submesh':
      self.Q_to_Qs_dofmap = Function(self.Q, name='Q_to_Qs_dofmap')

    elif self.kind == 'balance':
      # Enthalpy model
      self.theta0        = Function(self.Q, name='theta0')
      self.W0            = Function(self.Q, name='W0')
      self.thetahat      = Function(self.Q, name='thetahat')
      self.uhat          = Function(self.Q, name='uhat')
      self.vhat          = Function(self.Q, name='vhat')
      self.what          = Function(self.Q, name='what')
      self.mhat          = Function(self.Q, name='mhat')
      
      # Surface climate model
      self.precip        = Function(self.Q, name='precip')
    
    elif self.kind == 'hybrid': 
      # hybrid mass-balance :
      self.H             = Function(self.Q, name='H')
      self.H0            = Function(self.Q, name='H0')
      self.ubar_c        = Function(self.Q, name='ubar_c')
      self.vbar_c        = Function(self.Q, name='vbar_c')
      self.H_min         = Function(self.Q, name='H_min')
      self.H_max         = Function(self.Q, name='H_max')
      
      # hybrid energy-balance :
      self.deltax        = 1.0 / (self.N_T - 1.0)
      self.sigmas        = np.linspace(0, 1, self.N_T, endpoint=True)
      self.Tm            = Function(self.Z, name='Tm')
      self.T_            = Function(self.Z, name='T_')
      self.T0_           = Function(self.Z, name='T0_')
      self.Ts            = Function(self.Q, name='Ts')
      self.Tb            = Function(self.Q, name='Tb')
      
      self.U3            = Function(self.HV, name='U3')
      u_                 = [self.U3[0],   self.U3[2]]
      v_                 = [self.U3[1],   self.U3[3]]
      coef               = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
      dcoef              = [lambda s:0.0, lambda s:5*s**3]
      self.u             = VerticalBasis(u_,  coef, dcoef)
      self.v             = VerticalBasis(v_,  coef, dcoef)
      
      # basal velocity :
      self.U3_b          = Function(self.Q3, name='U3_b')
      u_b, v_b, w_b      = self.U3_b.split()
      u_b.rename('u_b', '')
      v_b.rename('v_b', '')
      w_b.rename('w_b', '')
      self.u_b           = u_b
      self.v_b           = v_b
      self.w_b           = w_b
      
      # surface velocity :
      self.U3_s          = Function(self.Q3, name='U3_s')
      u_s, v_s, w_s      = self.U3_s.split()
      u_s.rename('u_b', '')
      v_s.rename('v_b', '')
      w_s.rename('w_b', '')
      self.u_s           = u_s
      self.v_s           = v_s
      self.w_s           = w_s

  def thermo_solve(self, momentum, energy, annotate=False):
    """
    Perform thermo-mechanical coupling between momentum and energy.
    
    :param momentum: momentum instance to couple with ``energy``.
    :param energy:   energy instance to couple with ``momentum``.
    :type momentum:  :class:`~momentum.Momentum`
    :type energy:    :class:`~energy.Energy`
    """
    
    from cslvr import Momentum
    from cslvr import Energy

    # TODO: also make sure these are D2Model momentum and energy classes. 
    if not isinstance(momentum, Momentum):
      s = ">>> thermo_solve REQUIRES A 'Momentum' INSTANCE, NOT %s <<<"
      print_text(s % type(momentum) , 'red', 1)
      sys.exit(1)
    
    if not isinstance(energy, Energy):
      s = ">>> thermo_solve REQUIRES AN 'Energy' INSTANCE, NOT %s <<<"
      print_text(s % type(energy) , 'red', 1)
      sys.exit(1)

    # this is the simplest thing!
    momentum.solve(annotate=annotate)
    energy.solve(annotate=annotate)

  def transient_iteration(self, momentum, mass, time_step, adaptive, annotate):
    """
    This function defines one interation of the transient solution, and is 
    called by the function ``model.transient_solve``.
    """
    stars = "*****************************************************************"
    
    # TODO: this adaptive interation should be altered to include CFL.
    # do solve adaptively :
    if adaptive:
    
      # solve momentum equation, lower alpha on failure :
      solved_u = False
      par      = momentum.solve_params['solver']['newton_solver']
      while not solved_u:
        if par['relaxation_parameter'] < 0.2:
          status_u = [False, False]
          break
        # always reset velocity for good convergence :
        self.assign_variable(momentum.get_U(), DOLFIN_EPS)
        status_u = momentum.solve(annotate=annotate)
        solved_u = status_u[1]
        # TODO: rewind the dolfin-adjoint tape too!
        if not solved_u:
          par['relaxation_parameter'] /= 1.43
          print_text(stars, 'red', 1)
          s = ">>> WARNING: newton relaxation parameter lowered to %g <<<"
          print_text(s % par['relaxation_parameter'], 'red', 1)
          print_text(stars, 'red', 1)

      # solve mass equation, lowering time step on failure :
      solved_h = False
      dt       = time_step
      while not solved_h:
        if dt < DOLFIN_EPS:
          status_h = [False,False]
          break
        H        = self.H.copy(True)
        status_h = mass.solve(annotate=annotate)
        solved_h = status_h[1]
        # TODO: rewind the dolfin-adjoint tape too!
        if not solved_h:
          dt /= 2.0
          print_text(stars, 'red', 1)
          s = ">>> WARNING: time step lowered to %g <<<"
          print_text(s % dt, 'red', 1)
          self.init_time_step(dt)
          self.init_H_H0(H)
          print_text(stars, 'red', 1)

    # do not solve adaptively :
    else:
      momentum.solve(annotate=annotate)
      mass.solve(annotate=annotate)



