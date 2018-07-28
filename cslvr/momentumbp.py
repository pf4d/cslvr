from dolfin               import *
from dolfin_adjoint       import *
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d3model        import D3Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
import sys


class MomentumBP(Momentum):
  """				
  """
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    
    Initilize the residuals and Jacobian for the momentum equations.
    """
    #NOTE: experimental
    s = "::: INITIALIZING BP VELOCITY PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D3Model:
      s = ">>> MomentumBP REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    self.linear       = linear

    mesh       = model.mesh
    eps_reg    = model.eps_reg
    n          = model.n
    Q          = model.Q
    S          = model.S
    B          = model.B
    z          = model.x[2]
    rhoi       = model.rhoi
    rhow       = model.rhow
    R          = model.R
    g          = model.g
    beta       = model.beta
    A          = model.A
    N          = model.N
    D          = model.D
    Fb         = model.Fb
    
    dx_f       = model.dx_f
    dx_g       = model.dx_g
    dx         = model.dx
    dBed_g     = model.dBed_g
    dBed_f     = model.dBed_f
    dLat_t     = model.dLat_t
    dBed       = model.dBed

    dOmega     = model.dOmega()
    dOmega_g   = model.dOmega_g()
    dOmega_w   = model.dOmega_w()
    dGamma_bg  = model.dGamma_bg()
    dGamma_bw  = model.dGamma_bw()
    dGamma_b   = model.dGamma_b()
    dGamma_sgu = model.dGamma_sgu()
    dGamma_swu = model.dGamma_swu()
    dGamma_su  = model.dGamma_su()
    dGamma_sg  = model.dGamma_sg()
    dGamma_sw  = model.dGamma_sw()
    dGamma_s   = model.dGamma_s()
    dGamma_ld  = model.dGamma_ld()
    dGamma_lto = model.dGamma_lto()
    dGamma_ltu = model.dGamma_ltu()
    dGamma_lt  = model.dGamma_lt()
    dGamma_l   = model.dGamma_l()
    
    # new constants :
    p0     = 101325
    T0     = 288.15
    M      = 0.0289644
    ci     = model.ci

    dx     = model.dx
    
    #===========================================================================
    # define variational problem :
    
    # system unknown function space is created now if periodic boundaries 
    # are not used (see model.generate_function_space()) :
    if model.use_periodic:
      Q2   = model.Q2
    else:
      Q2   = FunctionSpace(mesh, model.QM2e)
    
    # momenturm and adjoint :
    U      = Function(Q2, name = 'U')
    wf     = Function(Q,  name = 'w')
    Lam    = Function(Q2, name = 'Lam')
    dU     = TrialFunction(Q2)
    Phi    = TestFunction(Q2)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), Q2.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), Q2.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), Q)

    # horizontal velocity :
    u, v      = U
    phi, psi  = Phi

    # viscosity :
    U3      = as_vector([u,v,0])
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      U3_c     = model.U3.copy(True)
      eta      = self.viscosity(U3_c)
      Vd       = 2 * eta * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta      = self.viscosity(U3)
      Vd       = (2*n)/(n+1) * A**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())
    
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
    f_w    = rhoi*g*(S - z) - rhow*g*D               # lateral
    p_a    = p0 * (1 - g*z/(ci*T0))**(ci*M/R)        # surface pressure
    
    #Ne       = (S-B) + rhow/rhoi * D
    #P        = -0.383
    #Q        = -0.349
    #Unorm    = sqrt(inner(U,U) + DOLFIN_EPS)
    #Coef     = 1/(beta * Ne**(q/p))
    
    # residual :
    self.mom_F = + 2 * eta * dot(epi_1, grad(phi)) * dOmega \
                 + 2 * eta * dot(epi_2, grad(psi)) * dOmega \
                 + rhoi * g * S.dx(0) * phi * dOmega \
                 + rhoi * g * S.dx(1) * psi * dOmega \
                 + beta * u * phi * dGamma_bg \
                 + beta * v * psi * dGamma_bg \
   
    if (not model.use_periodic and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      self.mom_F += f_w * (N[0]*phi + N[1]*psi) * dGamma_lt
    
    # add lateral boundary conditions :  
    # FIXME: need correct BP treatment here
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c       = model.U3.copy(True)
      eta_l      = self.viscosity(U3_c)
      sig_l      = self.quasi_stress_tensor(U3_c, model.p, eta_l)
      self.mom_F += dot(sig_l, N) * dGamma_ld
    
    self.w_F = + (u.dx(0) + v.dx(1) + dw.dx(2)) * chi * dOmega \
               + (u*N[0] + v*N[1] + dw*N[2] - Fb) * chi * dGamma_b \
  
    # Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)

    # list of boundary conditions
    self.mom_bcs  = []
    self.bc_w     = None
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using lateral boundary conditions -"
      print_text(s, self.color())

      self.mom_bcs.append(DirichletBC(Q2.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(Q2.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
      #self.bc_w = DirichletBC(Q, model.w_lat, model.ff, model.GAMMA_L_DVD)
    
    self.eta     = eta
    self.U       = U 
    self.wf      = wf
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
 
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the unknown Function.
    """
    return self.U

  def velocity(self):
    """
    return the velocity.
    """
    return self.model.U3

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def strain_rate_tensor(self, U):
    """
    return the 'Blatter-Pattyn' simplified strain-rate tensor of <U>.
    """
    u,v,w  = U
    epi    = 0.5 * (grad(U) + grad(U).T)
    epi02  = 0.5*u.dx(2)
    epi12  = 0.5*v.dx(2)
    epi22  = -u.dx(0) - v.dx(1)  # incompressibility
    epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02],
                        [epi[1,0],  epi[1,1],  epi12],
                        [epi02,     epi12,     epi22]])
    return epsdot

  def effective_strain_rate(self, U):
    """
    return the BP effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    epsdot = 0.5 * tr(dot(epi, epi))
    return epsdot

  def stress_tensor(self):
    """
    return the BP Cauchy stress tensor.
    """
    s   = "::: forming the BP Cauchy stress tensor :::"
    print_text(s, self.color())
    U     = as_vector([self.U[0], self.U[1], self.wf])
    epi   = self.strain_rate_tensor(U)
    I     = Identity(3)

    sigma = 2*self.eta*epi - model.p*I
    return sigma
  
  def quasi_strain_rate_tensor(self, U):
    """
    return the Dukowicz 2011 quasi-strain tensor.
    """
    u,v,w  = U
    epi_ii = u.dx(0)
    epi_ij = 0.5*(u.dx(1) + v.dx(0))
    epi_ik = 0.5* u.dx(2)
    epi_jj = v.dx(1)
    epi_jk = 0.5* v.dx(2)
    epi    = as_matrix([[epi_ii, epi_ij, epi_ik],
                        [epi_ij, epi_jj, epi_jk],
                        [0,      0,      0     ]])
    return epi

  def quasi_stress_tensor(self, U, eta):
    """
    return the Dukowicz 2011 quasi-stress tensor.
    """
    u,v,w  = U
    tau_ii = 2*u.dx(0) + v.dx(1)
    tau_ij = 0.5 * (u.dx(1) + v.dx(0))
    tau_ik = 0.5 * u.dx(2)
    tau_jj = 2*v.dx(1) + u.dx(0)
    tau_jk = 0.5 * v.dx(2)
    tau    = as_matrix([[tau_ii, tau_ij, tau_ik],
                        [tau_ij, tau_jj, tau_jk],
                        [0,      0,      0     ]])
    return 2*eta*tau

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-9,
                'relaxation_parameter'     : 0.7,
                'maximum_iterations'       : 25,
                'error_on_nonconvergence'  : False,
                'krylov_solver'            :
                {
                  'monitor_convergence'   : False,
                  #'preconditioner' :
                  #{
                  #  'structure' : 'same'
                  #}
                }
              }}
    m_params  = {'solver'               : nparams,
                 'solve_vert_velocity'  : True,
                 'solve_pressure'       : True,
                 'vert_solve_method'    : 'mumps'}
    return m_params

  def solve_pressure(self, annotate=False):
    """
    Solve for the BP pressure 'p'.
    """
    model  = self.model
    
    # solve for vertical velocity :
    s  = "::: solving BP pressure :::"
    print_text(s, self.color())
    
    Q       = model.Q
    rhoi    = model.rhoi
    g       = model.g
    S       = model.S
    z       = model.x[2]
    eta     = self.eta
    w       = self.wf

    p       = project(rhoi*g*(S - z) + 2*eta*w.dx(2), annotate=annotate)
    
    model.assign_variable(model.p, p, annotate=annotate)

  def solve_vert_velocity(self, annotate=False):
    """ 
    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    
    # solve for vertical velocity :
    s  = "::: solving BP vertical velocity :::"
    print_text(s, self.color())
    
    aw       = assemble(lhs(self.w_F))
    Lw       = assemble(rhs(self.w_F))
    if self.bc_w != None:
      self.bc_w.apply(aw, Lw)
    w_solver = LUSolver(self.solve_params['vert_solve_method'])
    w_solver.solve(aw, self.wf.vector(), Lw, annotate=annotate)
    #solve(lhs(self.R2) == rhs(self.R2), self.w, bcs = self.bc_w,
    #      solver_parameters = {"linear_solver" : sm})#,
    #                           "symmetric" : True},
    #                           annotate=False)
    
    self.assz.assign(model.w, self.wf, annotate=annotate)
    print_min_max(self.wf, 'w')
    
  def solve(self, annotate=False):
    """ 
    Perform the Newton solve of the first order equations 
    """

    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s      = "::: solving BP horizontal velocity with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())

    # zero out self.velocity for good convergence for any subsequent solves,
    # e.g. model.L_curve() :
    model.assign_variable(self.get_U(), DOLFIN_EPS, annotate=annotate)
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v = self.U.split()

    #self.assign_variable(model.u, u)
    #self.assign_variable(model.v, v)
    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)

    print_min_max(self.U, 'U')
      
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate=annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate=annotate)


class MomentumDukowiczBP(Momentum):
  """				
  """
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    
    Initilize the residuals and Jacobian for the momentum equations.
    """
    s = "::: INITIALIZING DUKOWICZ BP VELOCITY PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D3Model:
      s = ">>> MomentumDukowiczBP REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    self.linear       = linear

    mesh       = model.mesh
    Q          = model.Q
    S          = model.S
    B          = model.B
    Fb         = model.Fb
    z          = model.x[2]
    W          = model.W
    R          = model.R
    rhoi       = model.rhoi
    rhosw      = model.rhosw
    g          = model.g
    beta       = model.beta
    A          = model.A
    eps_reg    = model.eps_reg
    n          = model.n
    h          = model.h
    N          = model.N
    D          = model.D

    dOmega     = model.dOmega()
    dOmega_g   = model.dOmega_g()
    dOmega_w   = model.dOmega_w()
    dGamma_bg  = model.dGamma_bg()
    dGamma_bw  = model.dGamma_bw()
    dGamma_b   = model.dGamma_b()
    dGamma_sgu = model.dGamma_sgu()
    dGamma_swu = model.dGamma_swu()
    dGamma_su  = model.dGamma_su()
    dGamma_sg  = model.dGamma_sg()
    dGamma_sw  = model.dGamma_sw()
    dGamma_s   = model.dGamma_s()
    dGamma_ld  = model.dGamma_ld()
    dGamma_lto = model.dGamma_lto()
    dGamma_ltu = model.dGamma_ltu()
    dGamma_lt  = model.dGamma_lt()
    dGamma_l   = model.dGamma_l()
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
    
    # system unknown function space is created now if periodic boundaries 
    # are not used (see model.generate_function_space()) :
    if model.use_periodic:
      Q2   = model.Q2
    else:
      Q2   = FunctionSpace(mesh, model.QM2e)
    
    # momenturm and adjoint :
    U      = Function(Q2, name = 'G')
    Lam    = Function(Q2, name = 'Lam')
    dU     = TrialFunction(Q2)
    Phi    = TestFunction(Q2)
    Lam    = Function(Q2)

    # vertical velocity :
    dw     = TrialFunction(Q)
    chi    = TestFunction(Q)
    w      = Function(Q, name='w_f')
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), Q2.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), Q2.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), Q)
    phi, psi = Phi
    du,  dv  = dU
    u,   v   = U
   
    # viscous dissipation :
    U3      = as_vector([u,v,0])
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      U3_c     = model.U3.copy(True)
      eta      = self.viscosity(U3_c)
      Vd       = 2 * eta * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta      = self.viscosity(U3)
      Vd       = (2*n)/(n+1) * A**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())
      
    # potential energy :
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # dissipation by sliding :
    Sl_gnd = - 0.5 * beta * (u**2 + v**2)

    # pressure boundary :
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * (u*N[0] + v*N[1])
    
    # action :
    A      = (Vd - Pe)*dOmega - Sl_gnd*dGamma_bg - Pb*dGamma_bw
    
    if (not model.use_periodic and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      A -= Pb*dGamma_lt
    
    # add lateral boundary conditions :  
    # FIXME: need correct BP treatment here
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c       = model.U3.copy(True)
      eta_l      = self.viscosity(U3_c)
      sig_l      = self.quasi_stress_tensor(U3_c, model.p, eta_l)
      A -= dot(dot(sig_l, N), U3) * dGamma_ld

    # the first variation of the action in the direction of a
    # test function ; the extremum :
    self.mom_F = derivative(A, U, Phi)

    # the first variation of the extremum in the direction
    # a tril function ; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []
      
    self.w_F = + (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dOmega \
               + (u*N[0] + v*N[1] + dw*N[2] - Fb)*chi*dGamma_b
   
    self.eta     = eta
    self.A       = A
    self.U       = U 
    self.w       = w  
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
 
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the unknown Function.
    """
    return self.U

  def velocity(self):
    """
    return the velocity.
    """
    return self.model.U3
  
  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def strain_rate_tensor(self, U):
    """
    return the Dukowicz 'Blatter-Pattyn' simplified strain-rate tensor of <U>.
    """
    u,v,w  = U
    epi    = 0.5 * (grad(U) + grad(U).T)
    epi02  = 0.5*u.dx(2)
    epi12  = 0.5*v.dx(2)
    epi22  = -u.dx(0) - v.dx(1)  # incompressibility
    epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02],
                        [epi[1,0],  epi[1,1],  epi12],
                        [epi02,     epi12,     epi22]])
    return epsdot
    
  def effective_strain_rate(self, U):
    """
    return the Dukowicz BP effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    epsdot = 0.5 * tr(dot(epi, epi))
    return epsdot

  def stress_tensor(self):
    """
    return the BP Cauchy stress tensor.
    """
    s   = "::: forming the Dukowicz BP Cauchy stress tensor :::"
    print_text(s, self.color())
    U     = as_vector([self.U[0], self.U[1], self.w])
    epi   = self.strain_rate_tensor(U)
    I     = Identity(3)

    sigma = 2*self.eta*epi - model.p*I
    return sigma

  def quasi_stress_tensor(self, U, eta):
    """
    return the Dukowicz 2011 quasi-tensor.
    """
    u,v,w  = U
    tau_ii = 2*u.dx(0) + v.dx(1)
    tau_ij = 0.5 * (u.dx(1) + v.dx(0))
    tau_ik = 0.5 * u.dx(2)
    tau_jj = 2*v.dx(1) + u.dx(0)
    tau_jk = 0.5 * v.dx(2)
    tau    = as_matrix([[tau_ii, tau_ij, tau_ik],
                        [tau_ij, tau_jj, tau_jk],
                        [0,      0,      0     ]])
    return 2*eta*tau
  
  
  def deviatoric_stress_tensor(self, U, eta):
    """
    return the deviatoric part of the Cauchy stress tensor.
    """
    s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
    print_text(s, self.color())

    epi = self.strain_rate_tensor(U)
    tau = 2 * eta * epi
    return tau


  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-9,
                'relaxation_parameter'     : 0.7,
                'maximum_iterations'       : 25,
                'error_on_nonconvergence'  : False,
                'krylov_solver'            :
                {
                  'monitor_convergence'   : False,
                  #'preconditioner' :
                  #{
                  #  'structure' : 'same'
                  #}
                }
              }}
    m_params  = {'solver'               : nparams,
                 'solve_vert_velocity'  : True,
                 'solve_pressure'       : True,
                 'vert_solve_method'    : 'mumps'}
    return m_params

  def solve_pressure(self, annotate=False):
    """
    Solve for the Dukowicz BP pressure to model.p.
    """
    model  = self.model
    
    # solve for vertical velocity :
    s  = "::: solving Dukowicz BP pressure :::"
    print_text(s, self.color())
    
    Q       = model.Q
    rhoi    = model.rhoi
    g       = model.g
    S       = model.S
    z       = model.x[2]
    eta     = self.eta
    w       = self.w

    p       = project(rhoi*g*(S - z) + 2*eta*w.dx(2), annotate=annotate)
    
    # unify the pressure over shelves and grounded ice : 
    model.assign_variable(model.p, p, annotate=annotate)

  def solve_vert_velocity(self, annotate=False):
    """on.dumps(x, sort_keys=True, indent=2)

    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    
    # solve for vertical velocity :
    s  = "::: solving Dukowicz BP vertical velocity :::"
    print_text(s, self.color())
    
    aw       = assemble(lhs(self.w_F))
    Lw       = assemble(rhs(self.w_F))
    #if self.bc_w != None:
    #  self.bc_w.apply(aw, Lw)
    w_solver = LUSolver(self.solve_params['vert_solve_method'])
    w_solver.solve(aw, self.w.vector(), Lw, annotate=annotate)
    #solve(lhs(self.R2) == rhs(self.R2), self.w, bcs = self.bc_w,
    #      solver_parameters = {"linear_solver" : sm})#,
    #                           "symmetric" : True},
    #                           annotate=False)
    
    self.assz.assign(model.w, self.w, annotate=annotate)
    print_min_max(self.w, 'w')
    
  def solve(self, annotate=False):
    """ 
    Perform the Newton solve of the first order equations 
    """

    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s      = "::: solving Dukowicz BP horizontal velocity with %i max" + \
             " iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # zero out self.velocity for good convergence for any subsequent solves,
    # e.g. model.L_curve() :
    model.assign_variable(self.get_U(), DOLFIN_EPS, annotate=annotate)
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v = self.U.split()

    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)

    u,v,w = model.U3.split(True)
    print_min_max(u, 'u')
    print_min_max(v, 'v')
      
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate=annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate=annotate)



