from fenics         import *
from dolfin_adjoint import *
from io             import print_text, print_min_max
from D3Model        import D3Model

class BPModel(D3Model):
  """ 
  Instance of a 2D flowline ice model that contains geometric and scalar 
  parameters and supporting functions.  This class does not contain actual 
  physics but rather the interface to use physics in different simulation 
  types.
  """

  def __init__(self, config=None):
    """
    Create and instance of the model.
    """
    D3Model.__init__(self, config)
    self.color = 'cyan'

  def strain_rate_tensor(self, U):
    """
    return the 'Blatter-Pattyn' simplified strain-rate tensor of <U>.
    """
    u,v,w = U
    epi   = 0.5 * (grad(U) + grad(U).T)
    epi02 = 0.5*u.dx(2)
    epi12 = 0.5*v.dx(2)
    epi22 = -u.dx(0) - v.dx(1)  # incompressibility
    epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02],
                        [epi[1,0],  epi[1,1],  epi12],
                        [epi02,     epi12,     epi22]])
    return epsdot
    
  def effective_strain(self, U):
    """
    return the BP effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    ep_xx  = epi[0,0]
    ep_yy  = epi[1,1]
    ep_zz  = epi[2,2]
    ep_xy  = epi[0,1]
    ep_xz  = epi[0,2]
    ep_yz  = epi[1,2]
    
    # Second invariant of the strain rate tensor squared
    epsdot = + ep_xx**2 + ep_yy**2 + ep_xx*ep_yy \
             + ep_xy**2 + ep_xz**2 + ep_yz**2
    return epsdot

  def stress_tensor(self):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the BP Cauchy stress tensor :::"
    print_text(s, self.color)
    U   = as_vector([self.u, self.v, self.w])
    epi = self.strain_rate_tensor(U)
    I   = Identity(3)

    sigma = 2*self.eta*epi - self.p*I
    return sigma

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    super(BPModel, self).initialize_variables()

    s = "::: initializing BP variables :::"
    print_text(s, self.color)

    config = self.config
    
    # momenturm and adjoint :
    self.U      = Function(self.Q2, name = 'U')
    self.wf     = Function(self.Q,  name = 'w')
    self.dU     = TrialFunction(self.Q2)
    self.Phi    = TestFunction(self.Q2)
    self.Lam    = Function(self.Q2)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx   = FunctionAssigner(self.V.sub(0), self.Q2.sub(0))
    self.assy   = FunctionAssigner(self.V.sub(1), self.Q2.sub(1))
    self.assz   = FunctionAssigner(self.V.sub(2), self.Q)

    U_t         = as_vector([self.U[0], self.U[1], 0.0])
    self.epsdot = self.effective_strain(U_t)

  def default_nonlin_solver_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'linear_solver'            : 'cg',
               'preconditioner'           : 'hypre_amg',
               'relative_tolerance'       : 1e-8,
               'relaxation_parameter'     : 1.0,
               'maximum_iterations'       : 25,
               'error_on_nonconvergence'  : False}
    return {'newton_solver' : nparams}
    
  def init_momentum(self):
    """ 
    Initilize the residuals and Jacobian for the momentum equations.
    """
    s = "::: INITIALIZING BP VELOCITY PHYSICS :::"
    print_text(s, self.color)

    config     = self.config
    mesh       = self.mesh
    r          = config['velocity']['r']
    V          = self.Q2
    Q          = self.Q
    U          = self.U
    dU         = self.dU
    Phi        = self.Phi
    eta_shf    = self.eta_shf
    eta_gnd    = self.eta_gnd
    S          = self.S
    B          = self.B
    H          = S - B
    x          = self.x
    rhoi       = self.rhoi
    rhow       = self.rhow
    R          = self.R
    g          = self.g
    beta       = self.beta
    w          = self.w
    N          = self.N
    D          = self.D
    
    dx_s       = self.dx_s
    dx_g       = self.dx_g
    dx         = self.dx
    dGnd       = self.dGnd
    dFlt       = self.dFlt
    dSde       = self.dSde
    dBed       = self.dBed
    
    gradS      = grad(S)
    gradB      = grad(B)
     
    # new constants :
    p0     = 101325
    T0     = 288.15
    M      = 0.0289644
    ci     = self.ci

    dx     = self.dx
    
    #===========================================================================
    # define variational problem :

    # horizontal velocity :
    u, v      = U
    phi, psi  = Phi
    
    # vertical velocity :
    dw        = TrialFunction(Q)
    chi       = TestFunction(Q)
    
    epi_1  = as_vector([   2*u.dx(0) + v.dx(1), 
                        0.5*(u.dx(1) + v.dx(0)),
                        0.5* u.dx(2)            ])
    epi_2  = as_vector([0.5*(u.dx(1) + v.dx(0)),
                             u.dx(0) + 2*v.dx(1),
                        0.5* v.dx(2)            ])
   
    # boundary integral terms : 
    f_w    = rhoi*g*(S - x[2]) + rhow*g*D               # lateral
    p_a    = p0 * (1 - g*x[2]/(ci*T0))**(ci*M/R)        # surface pressure
    
    #Ne       = H + rhow/rhoi * D
    #P        = -0.383
    #Q        = -0.349
    #Unorm    = sqrt(inner(U,U) + DOLFIN_EPS)
    #Coef     = 1/(beta * Ne**(q/p))
    
    # residual :
    R1 = + 2 * eta_shf * dot(epi_1, grad(phi)) * dx_s \
         + 2 * eta_shf * dot(epi_2, grad(psi)) * dx_s \
         + 2 * eta_gnd * dot(epi_1, grad(phi)) * dx_g \
         + 2 * eta_gnd * dot(epi_2, grad(psi)) * dx_g \
         + rhoi * g * gradS[0] * phi * dx \
         + rhoi * g * gradS[1] * psi * dx \
         + beta**2 * u * phi * dGnd \
         + beta**2 * v * psi * dGnd \
         + Constant(DOLFIN_EPS) * u * phi * dFlt \
         + Constant(DOLFIN_EPS) * v * psi * dFlt \
    
    if (not config['periodic_boundary_conditions']
        and not config['velocity']['use_lat_bcs']
        and config['use_pressure_boundary']):
      R1 -= f_w * (N[0]*phi + N[1]*psi) * dSde \
    
    R2 = + (u.dx(0) + v.dx(1) + dw.dx(2)) * chi * dx \
         + (u*N[0] + v*N[1] + dw*N[2]) * chi * dBed \
  
    # residuals :  
    self.mom_F = R1
    self.w_F   = R2
    
    # Jacobian :
    self.mom_Jac = derivative(R1, U, dU)

    # list of boundary conditions
    self.mom_bcs  = []
    self.bc_w     = None
      
    # add lateral boundary conditions :  
    if config['velocity']['use_lat_bcs']:
      s = "    - using lateral boundary conditions -"
      print_text(s, self.color)

      self.mom_bcs.append(DirichletBC(V.sub(0), self.u_lat, self.ff, 7))
      self.mom_bcs.append(DirichletBC(V.sub(1), self.v_lat, self.ff, 7))
      #self.bc_w = DirichletBC(Q, self.w_lat, self.ff, 4)

  def solve_pressure(self):
    """
    Solve for the BP pressure 'p'.
    """
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving BP pressure :::"
    print_text(s, self.color)
    
    Q       = self.Q
    rhoi    = self.rhoi
    g       = self.g
    S       = self.S
    x       = self.x
    w       = self.wf
    p       = self.p
    eta_shf = self.eta_shf
    eta_gnd = self.eta_gnd

    p_shf   = project(rhoi*g*(S - x[2]) + 2*eta_shf*w.dx(2), Q, annotate=False)
    p_gnd   = project(rhoi*g*(S - x[2]) + 2*eta_gnd*w.dx(2), Q, annotate=False)
    
    # unify the pressure over shelves and grounded ice : 
    p_v                = p.vector().array()
    p_gnd_v            = p_gnd.vector().array()
    p_shf_v            = p_shf.vector().array()
    p_v[self.gnd_dofs] = p_gnd_v[self.gnd_dofs]
    p_v[self.shf_dofs] = p_shf_v[self.shf_dofs]
    self.assign_variable(p, p_v)
   
    print_min_max(p, 'p')

  def solve_vert_velocity(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving BP vertical velocity :::"
    print_text(s, self.color)
    
    sm = config['velocity']['vert_solve_method']
    
    aw       = assemble(lhs(self.w_F))
    Lw       = assemble(rhs(self.w_F))
    if self.bc_w != None:
      self.bc_w.apply(aw, Lw)
    w_solver = LUSolver(sm)
    w_solver.solve(aw, self.wf.vector(), Lw, annotate=False)
    #solve(lhs(self.R2) == rhs(self.R2), self.w, bcs = self.bc_w,
    #      solver_parameters = {"linear_solver" : sm})#,
    #                           "symmetric" : True},
    #                           annotate=False)
    
    self.assz.assign(self.w, self.wf)
    print_min_max(self.wf, 'w')
    
  def solve_momentum(self, annotate=True, params=None):
    """ 
    Perform the Newton solve of the first order equations 
    """
    config = self.config

    if params == None:
      params = self.default_nonlin_solver_params()
    
    # solve nonlinear system :
    params = config['velocity']['newton_params']
    rtol   = params['newton_solver']['relative_tolerance']
    maxit  = params['newton_solver']['maximum_iterations']
    alpha  = params['newton_solver']['relaxation_parameter']
    s      = "::: solving BP horizontal velocity with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color)
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate=annotate, solver_parameters = params)
    u, v = self.U.split()

    #self.assign_variable(self.u, u)
    #self.assign_variable(self.v, v)
    self.assx.assign(self.u, u)
    self.assy.assign(self.v, v)

    print_min_max(self.U, 'U')
      
    if config['velocity']['solve_vert_velocity']:
      self.solve_vert_velocity()
    if config['velocity']['solve_pressure']:
      self.solve_pressure()




