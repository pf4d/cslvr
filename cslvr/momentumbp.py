from fenics               import *
from dolfin_adjoint       import *
from cslvr.io             import print_text, print_min_max
from cslvr.d3model        import D3Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
import sys


class MomentumBP(Momentum):
  """				
  """
  def initialize(self, model, solve_params=None, isothermal=True,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    
    Initilize the residuals and Jacobian for the momentum equations.
    """
    s = "::: INITIALIZING BP VELOCITY PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D3Model:
      s = ">>> MomentumBP REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    
    # momenturm and adjoint :
    U      = Function(model.Q2, name = 'U')
    wf     = Function(model.Q,  name = 'w')
    Lam    = Function(model.Q2, name = 'Lam')
    dU     = TrialFunction(model.Q2)
    Phi    = TestFunction(model.Q2)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), model.Q2.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), model.Q2.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), model.Q)

    mesh       = model.mesh
    eps_reg    = model.eps_reg
    n          = model.n
    r          = model.r
    V          = model.Q2
    Q          = model.Q
    S          = model.S
    B          = model.B
    H          = S - B
    z          = model.x[2]
    rhoi       = model.rhoi
    rhow       = model.rhow
    R          = model.R
    g          = model.g
    beta       = model.beta
    N          = model.N
    D          = model.D
    
    dx_f       = model.dx_f
    dx_g       = model.dx_g
    dx         = model.dx
    dBed_g     = model.dBed_g
    dBed_f     = model.dBed_f
    dLat_t     = model.dLat_t
    dBed       = model.dBed
    
    gradS      = grad(S)
    gradB      = grad(B)

    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot = self.effective_strain_rate(model.U3.copy(True))
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      U_t      = as_vector([U[0], U[1], 0])
      epsdot   = self.effective_strain_rate(U_t)
    
    if isothermal:
      s   = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      b_shf = model.E_shf * model.b_shf
      b_gnd = model.E_gnd * model.b_gnd

    else:
      s   = "    - using temperature-dependent rate-factor -"
      print_text(s, self.color())
      T       = model.T
      Tp      = model.Tp
      W       = model.W
      R       = model.R
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
      W_T     = conditional( lt(W, 0.01),   W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*Tp)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*Tp)) )**(-1/n)
    
    eta_shf    = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
    eta_gnd    = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
     
    # new constants :
    p0     = 101325
    T0     = 288.15
    M      = 0.0289644
    ci     = model.ci

    dx     = model.dx
    
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
    f_w    = rhoi*g*(S - z) - rhow*g*D               # lateral
    p_a    = p0 * (1 - g*z/(ci*T0))**(ci*M/R)        # surface pressure
    
    #Ne       = H + rhow/rhoi * D
    #P        = -0.383
    #Q        = -0.349
    #Unorm    = sqrt(inner(U,U) + DOLFIN_EPS)
    #Coef     = 1/(beta * Ne**(q/p))
    
    # residual :
    self.mom_F = + 2 * eta_shf * dot(epi_1, grad(phi)) * dx_f \
                 + 2 * eta_shf * dot(epi_2, grad(psi)) * dx_f \
                 + 2 * eta_gnd * dot(epi_1, grad(phi)) * dx_g \
                 + 2 * eta_gnd * dot(epi_2, grad(psi)) * dx_g \
                 + rhoi * g * gradS[0] * phi * dx \
                 + rhoi * g * gradS[1] * psi * dx \
                 + beta * u * phi * dBed_g \
                 + beta * v * psi * dBed_g \
    #             + f_w * N[0] * phi * dBed_f \
    #             + f_w * N[1] * psi * dBed_f
    
    if (not model.use_periodic_boundaries 
        and not use_lat_bcs and use_pressure_bc):
      s = "    - using cliff-pressure boundary condition -"
      print_text(s, self.color())
      self.mom_F += f_w * (N[0]*phi + N[1]*psi) * dLat_t
    
    self.w_F = + (u.dx(0) + v.dx(1) + dw.dx(2)) * chi * dx \
               + (u*N[0] + v*N[1] + dw*N[2]) * chi * dBed \
  
    # Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)

    # list of boundary conditions
    self.mom_bcs  = []
    self.bc_w     = None
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using lateral boundary conditions -"
      print_text(s, self.color())

      self.mom_bcs.append(DirichletBC(V.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(V.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
      #self.bc_w = DirichletBC(Q, model.w_lat, model.ff, model.GAMMA_L_DVD)
    
    self.U       = U 
    self.wf      = wf
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
    self.epsdot  = epsdot
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
 
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the velocity Function.
    """
    return self.U

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
    return the BP Cauchy stress tensor.
    """
    s   = "::: forming the BP Cauchy stress tensor :::"
    print_text(s, self.color())
    U     = as_vector([self.U[0], self.U[1], self.wf])
    epi   = self.strain_rate_tensor(U)
    I     = Identity(3)

    sigma = 2*self.eta*epi - model.p*I
    return sigma

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-8,
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
    p       = model.p
    eta_shf = self.eta_shf
    eta_gnd = self.eta_gnd
    w       = self.wf

    p_shf   = project(rhoi*g*(S - z) + 2*eta_shf*w.dx(2), annotate=annotate)
    p_gnd   = project(rhoi*g*(S - z) + 2*eta_gnd*w.dx(2), annotate=annotate)
    
    # unify the pressure over shelves and grounded ice : 
    p_v                 = p.vector().array()
    p_gnd_v             = p_gnd.vector().array()
    p_shf_v             = p_shf.vector().array()
    p_v[model.gnd_dofs] = p_gnd_v[model.gnd_dofs]
    p_v[model.shf_dofs] = p_shf_v[model.shf_dofs]
    model.assign_variable(p, p_v, cls=self, annotate=annotate)

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
    print_min_max(self.wf, 'w', cls=self)
    
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
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v = self.U.split()

    #self.assign_variable(model.u, u)
    #self.assign_variable(model.v, v)
    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)

    print_min_max(self.U, 'U', cls=self)
      
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate=annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate=annotate)


class MomentumDukowiczBP(Momentum):
  """				
  """
  def initialize(self, model, solve_params=None, isothermal=True,
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
    
    # momenturm and adjoint :
    U      = Function(model.Q2, name = 'G')
    Lam    = Function(model.Q2, name = 'Lam')
    dU     = TrialFunction(model.Q2)
    Phi    = TestFunction(model.Q2)
    Lam    = Function(model.Q2)

    # vertical velocity :
    dw     = TrialFunction(model.Q)
    chi    = TestFunction(model.Q)
    w      = Function(model.Q, name='w_f')
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), model.Q2.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), model.Q2.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), model.Q)

    mesh       = model.mesh
    r          = model.r
    S          = model.S
    B          = model.B
    H          = S - B
    z          = model.x[2]
    W          = model.W
    R          = model.R
    rhoi       = model.rhoi
    rhosw      = model.rhosw
    g          = model.g
    beta       = model.beta
    h          = model.h
    N          = model.N
    D          = model.D

    gradS      = grad(S)
    gradB      = grad(B)
    
    dx_f       = model.dx_f
    dx_g       = model.dx_g
    dx         = model.dx
    dBed_g     = model.dBed_g
    dBed_f     = model.dBed_f
    dLat_t     = model.dLat_t
    dLat_d     = model.dLat_d
    dBed       = model.dBed
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
    phi, psi = Phi
    du,  dv  = dU
    u,   v   = U

    eps_reg  = model.eps_reg
    n        = model.n
    
    if isothermal:
      s   = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      b_shf = model.E_shf * model.b_shf
      b_gnd = model.E_gnd * model.b_gnd

    else:
      s   = "    - using energy-dependent rate-factor -"
      print_text(s, self.color())
      T       = model.T
      Tp      = model.T + model.gamma * model.p
      W       = model.W
      R       = model.R
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
      W_T     = conditional( lt(W, 0.01),   W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*Tp)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*Tp)) )**(-1/n)
   
    # 1) Viscous dissipation
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot_l  = self.effective_strain_rate(model.U3.copy(True))
      epsdot    = self.effective_strain_rate(as_vector([u,v,0]))
      eta_shf   = 0.5 * b_shf * (epsdot_l + eps_reg)**((1-n)/(2*n))
      eta_gnd   = 0.5 * b_gnd * (epsdot_l + eps_reg)**((1-n)/(2*n))
      Vd_shf    = 2 * eta_shf * epsdot
      Vd_gnd    = 2 * eta_gnd * epsdot
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot  = self.effective_strain_rate(as_vector([u,v,0]))
      eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      Vd_shf  = (2*n)/(n+1) * b_shf * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd  = (2*n)/(n+1) * b_gnd * (epsdot + eps_reg)**((n+1)/(2*n))
      
    # 2) Potential energy
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # 3) Dissipation by sliding
    Sl_gnd = - 0.5 * beta * (u**2 + v**2)

    # 4) pressure boundary
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * (u*N[0] + v*N[1])
    
    # 5) tangential velocity to divide :
    Db     = - (u**2*N[0] + v**2*N[1])

    # Variational principle
    A      = + Vd_shf*dx_f + Vd_gnd*dx_g - Pe*dx \
             - Sl_gnd*dBed_g - Pb*dBed_f
    
    if (not model.use_periodic_boundaries 
        and not use_lat_bcs and use_pressure_bc):
      s = "    - using calving-front-pressure-boundary condition -"
      print_text(s, self.color())
      A -= Pb*dLat_t

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.mom_F = derivative(A, U, Phi)

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using divide-lateral boundary conditions -"
      print_text(s, self.color())
      self.mom_bcs.append(DirichletBC(model.Q2.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.Q2.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
   
    self.w_F = (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx - \
               (u*N[0] + v*N[1] + dw*N[2])*chi*dBed
   
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
    self.b_shf   = b_shf
    self.b_gnd   = b_gnd
    self.A       = A
    self.U       = U 
    self.w       = w  
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
    self.epsdot  = epsdot
 
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the velocity Function.
    """
    return self.U

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
    return the BP Cauchy stress tensor.
    """
    s   = "::: forming the Dukowicz BP Cauchy stress tensor :::"
    print_text(s, self.color())
    U     = as_vector([self.U[0], self.U[1], self.w])
    epi   = self.strain_rate_tensor(U)
    I     = Identity(3)

    sigma = 2*self.eta*epi - model.p*I
    return sigma

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-6,
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
    p       = model.p
    eta_shf = self.eta_shf
    eta_gnd = self.eta_gnd
    w       = self.w

    p_shf   = project(rhoi*g*(S - z) + 2*eta_shf*w.dx(2),
                      annotate=annotate)
    p_gnd   = project(rhoi*g*(S - z) + 2*eta_gnd*w.dx(2),
                      annotate=annotate)
    
    # unify the pressure over shelves and grounded ice : 
    p_v                 = p.vector().array()
    p_gnd_v             = p_gnd.vector().array()
    p_shf_v             = p_shf.vector().array()
    p_v[model.gnd_dofs] = p_gnd_v[model.gnd_dofs]
    p_v[model.shf_dofs] = p_shf_v[model.shf_dofs]
    model.assign_variable(p, p_v, cls=self)

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
    print_min_max(self.w, 'w', cls=self)
    
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
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v = self.U.split()

    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)

    print_min_max(self.U, 'U', cls=self)
      
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate=False)



