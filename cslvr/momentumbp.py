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
    self.linear       = linear
    
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

    # viscosity :
    self.form_rate_factor(isothermal)
    self.form_viscosity(as_vector([u,v,0]) , linear)
    eta_shf = self.eta_shf
    eta_gnd = self.eta_gnd
    
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
    self.linear       = linear
    
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
    Fb         = model.Fb
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
   
    # 1) Viscous dissipation
    self.form_rate_factor(isothermal)
    self.form_viscosity(as_vector([u,v,0]), linear)
    Vd_shf = self.Vd_shf
    Vd_gnd = self.Vd_gnd
      
    # 2) Potential energy
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # 3) Dissipation by sliding
    Sl_gnd = - 0.5 * beta * (u**2 + v**2)

    # 4) pressure boundary
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * (u*N[0] + v*N[1])
    
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
               (u*N[0] + v*N[1] + (dw - Fb)*N[2])*chi*dBed
   
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

    u,v,w = model.U3.split(True)
    print_min_max(u, 'u', cls=self)
    print_min_max(v, 'v', cls=self)
      
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate=False)


class MomentumDukowiczBPModified(Momentum):
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
    self.linear       = linear
    
    # momenturm and adjoint :
    U      = Function(model.Q3, name = 'G')
    Lam    = Function(model.Q3, name = 'Lam')
    dU     = TrialFunction(model.Q3)
    Phi    = TestFunction(model.Q3)
    Lam    = Function(model.Q3)

    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), model.Q3.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), model.Q3.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), model.Q3.sub(2))

    mesh       = model.mesh
    r          = model.r
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
    h          = model.h
    N          = model.N
    D          = model.D

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
    phi, psi, xi  = Phi
    du,  dv,  dw  = dU
    u,   v,   w   = U

    eps_reg  = model.eps_reg
    n        = model.n
    
    # 1) Viscous dissipation
    self.form_rate_factor(isothermal)
    self.form_viscosity(U, linear)
    Vd_shf  = self.Vd_shf
    Vd_gnd  = self.Vd_gnd
    eta_shf = self.eta_shf
    eta_gnd = self.eta_gnd
      
    # 2) Potential energy
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # 3) Dissipation by sliding
    Sl_gnd = - 0.5 * beta * (u**2 + v**2 + w**2)

    # 4) pressure boundary
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * (u*N[0] + v*N[1] + w*N[2])
    
    # 5) impenetrability constraint :
    # FIXME: this does not work, unlike the FS analog which does...  dunno
    tau_f  = self.quasi_stress_tensor(U, eta_shf)
    tau_g  = self.quasi_stress_tensor(U, eta_gnd)
    lam_f  = - dot(N, dot(tau_f, N))
    lam_g  = - dot(N, dot(tau_g, N))
    Nc_f   = - lam_f * (u*N[0] + v*N[1] + w*N[2])
    Nc_g   = - lam_g * (u*N[0] + v*N[1] + w*N[2])
    
    # 6) incompressiblity constraint :
    Pc     = (u.dx(0) + v.dx(1) + w.dx(2))

    # Variational principle
    A      = + Vd_shf*dx_f + Vd_gnd*dx_g - Pe*dx \
             - Sl_gnd*dBed_g - Nc_f*dBed_f - Nc_g*dBed_g - Pb*dBed_f
    
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
      self.mom_bcs.append(DirichletBC(model.Q3.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.Q3.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.Q3.sub(2),
                          model.w_lat, model.ff, model.GAMMA_L_DVD))
   
    self.A       = A
    self.U       = U 
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
    return epi
    #epi02  = 0.5*u.dx(2)
    #epi12  = 0.5*v.dx(2)
    #epi22  = -u.dx(0) - v.dx(1)  # incompressibility
    #epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02],
    #                    [epi[1,0],  epi[1,1],  epi12],
    #                    [epi02,     epi12,     epi22]])
    #return epsdot
    
  #def effective_strain_rate(self, U):
  #  """
  #  return the Dukowicz BP effective strain rate squared.
  #  """
  #  epi    = self.strain_rate_tensor(U)
  #  ep_xx  = epi[0,0]
  #  ep_yy  = epi[1,1]
  #  ep_zz  = epi[2,2]
  #  ep_xy  = epi[0,1]
  #  ep_xz  = epi[0,2]
  #  ep_yz  = epi[1,2]
  #  
  #  # Second invariant of the strain rate tensor squared
  #  epsdot = + ep_xx**2 + ep_yy**2 + ep_xx*ep_yy \
  #           + ep_xy**2 + ep_xz**2 + ep_yz**2
  #  return epsdot

  def effective_strain_rate(self, U):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
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

  #def quasi_stress_tensor(self, U, eta):
  #  """
  #  return the Dukowicz 2011 quasi-tensor.
  #  """
  #  u,v,w  = U
  #  tau_ii = 2*u.dx(0) * v.dx(1)
  #  tau_ij = 0.5 * (u.dx(1) + v.dx(0))
  #  tau_ik = 0.5 * u.dx(2)
  #  tau_jj = u.dx(0) + 2*v.dx(1)
  #  tau_jk = 0.5 * v.dx(2)
  #  tau    = as_matrix([[tau_ii, tau_ij, tau_ik],
  #                      [tau_ij, tau_jj, tau_jk],
  #                      [0,      0,      0     ]])
  #  return 2*eta*tau

  def quasi_stress_tensor(self, U, eta):
    """
    return the Dukowicz 2011 quasi-tensor.
    """
    u,v,w  = U
    tau_ii = u.dx(0) - w.dx(2)
    tau_ij = 0.5 * (u.dx(1) + v.dx(0))
    tau_ik = 0.5 * (u.dx(2) + w.dx(0))
    tau_jj = v.dx(1) - w.dx(2)
    tau_jk = 0.5 * (v.dx(2) + w.dx(1))
    tau    = as_matrix([[tau_ii, tau_ij, tau_ik],
                        [tau_ij, tau_jj, tau_jk],
                        [0,      0,      0     ]])
    return 2*eta*tau

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
    u,v,w   = self.U

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

    u,v,w = self.U
    N     = model.N
    dBed  = model.dBed
    
    dw     = TrialFunction(model.Q)
    chi    = TestFunction(model.Q)
    w      = Function(model.Q, name='w_f')
    
    w_F = + (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx \
          - (u*N[0] + v*N[1] + dw*N[2])*chi*dBed
    
    aw       = assemble(lhs(w_F))
    Lw       = assemble(rhs(w_F))
    
    w_solver = LUSolver(self.solve_params['vert_solve_method'])
    w_solver.solve(aw, w.vector(), Lw, annotate=annotate)
    
    assz  = FunctionAssigner(model.w.function_space(), model.Q)
    assz.assign(model.w, w, annotate=annotate)
    print_min_max(w, 'w', cls=self)

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
    s      = "::: solving Dukowicz BP Modified 3D velocity with %i max" + \
             " iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v, w = self.U.split()

    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)
    self.assz.assign(model.w, w, annotate=annotate)

    u,v,w  = model.U3.split(True)

    print_min_max(u, 'u', cls=self)
    print_min_max(v, 'v', cls=self)
    print_min_max(w, 'w', cls=self)
      
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate=False)



