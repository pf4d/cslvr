from fenics                 import *
from dolfin_adjoint         import *
from varglas.io             import print_text, print_min_max
from varglas.d3model        import D3Model
from varglas.d2model        import D2Model
from varglas.d1model        import D1Model
from varglas.physics_new    import Physics
from varglas.helper         import VerticalBasis, VerticalFDBasis, \
                                   raiseNotDefined
from varglas.momentum       import Momentum
from copy                   import deepcopy
import numpy                    as np
import sys



class Monolithic(Physics):
  """
  """
  def __init__(self, model, solve_params=None, transient=False,
               use_lat_bcs=False, isothermal=True,
               linear=False, use_pressure_bc=True):
    """
    """
    self.transient = transient

    s    = "::: INITIALIZING MONOLITHIC PHYSICS :::"
    print_text(s, cls=self)

    if type(model) != D3Model:
      s = ">>> Monolithic REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
   
    # set solver parameters : 
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params
    
    r             = model.r
    mesh          = model.mesh
    V             = model.Q3
    Q             = model.Q
    n             = model.n
    eps_reg       = model.eps_reg
    T             = model.T
    T_melt        = model.T_melt
    Mb            = model.Mb
    L             = model.L
    T_w           = model.T_w
    gamma         = model.gamma
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    W             = model.W
    R             = model.R
    eps_reg       = model.eps_reg
    g             = model.g
    beta          = model.beta
    rhoi          = model.rhoi
    rhow          = model.rhow
    rhosw         = model.rhosw
    kw            = model.kw
    cw            = model.cw
    T_surface     = model.T_surface
    theta_surface = model.theta_surface
    theta_float   = model.theta_float
    theta_melt    = model.theta_melt
    theta_app     = model.theta_app
    q_geo         = model.q_geo
    thetahat      = model.thetahat
    uhat          = model.uhat
    vhat          = model.vhat
    what          = model.what
    mhat          = model.mhat
    spy           = model.spy
    h             = model.h
    E_shf         = model.E_shf
    E_gnd         = model.E_gnd
    N             = model.N
    z             = model.x[2]
    D             = model.D
    gradS         = grad(S)
    gradB         = grad(B)
    
    dx         = model.dx
    dx_f       = model.dx_f
    dx_g       = model.dx_g
    ds         = model.ds
    dBed_g     = model.dBed_g
    dBed_f     = model.dBed_f
    dLat_t     = model.dLat_t
    dLat_d     = model.dLat_d
    dBed       = model.dBed
    dSrf       = model.dSrf
    
    #===========================================================================
    # define variational problem :
    Q3     = model.Q2 * model.Q
    G      = Function(Q3, name = 'G')
    dG     = TrialFunction(Q3)
    Tst    = TestFunction(Q3)
    
    dU,  dtheta = split(dG)
    U,   theta  = split(G)
    Phi, xi     = split(Tst)
    
    phi, psi    = Phi
    du,  dv     = dU
    u,   v      = U
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.Q3.sub(0), Q3.sub(0).sub(0))
    self.assy  = FunctionAssigner(model.Q3.sub(1), Q3.sub(0).sub(1))
    self.assz  = FunctionAssigner(model.Q3.sub(2), model.Q)
    self.asst  = FunctionAssigner(model.Q,         Q3.sub(1))
    
    #w = u*B.dx(0) + v*B.dx(1) - (u.dx(0) + v.dx(1))*(z - B)
    w = - u.dx(0)*(z - B) + u*B.dx(0) - v.dx(1)*(z - B) + v*B.dx(1)

    U_v = as_vector([u,v,w])
    
    if isothermal:
      s   = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      b_shf = model.E_shf * model.b_shf
      b_gnd = model.E_gnd * model.b_gnd

    else:
      s   = "    - using temperature-dependent rate-factor -"
      print_text(s, self.color())
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      T_c     = 263.15
      theta_c = 146.3*T_c + 7.253/2.0*T_c**2
      W_c     = (theta - theta_melt) / L
      a_T     = conditional( lt(theta, theta_c), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(theta, theta_c), 6e4,          13.9e4)
      W_T     = conditional( lt(W_c,   0.01),    W,            0.01)
      #a_T     = conditional( lt(T, T_c),  1.1384496e-5, 5.45e10)
      #Q_T     = conditional( lt(T, T_c),  6e4,          13.9e4)
      #W_T     = conditional( lt(W, 0.01), W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
   
    # 1) Viscous dissipation
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot_l  = self.effective_strain_rate(model.U3.copy(True))
      epsdot    = self.effective_strain_rate(U_v)
      eta_shf   = 0.5 * b_shf * (epsdot_l + eps_reg)**((1-n)/(2*n))
      eta_gnd   = 0.5 * b_gnd * (epsdot_l + eps_reg)**((1-n)/(2*n))
      Vd_shf    = 2 * eta_shf * epsdot
      Vd_gnd    = 2 * eta_gnd * epsdot
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot  = self.effective_strain_rate(U_v)
      eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      Vd_shf  = (2*n)/(n+1) * b_shf * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd  = (2*n)/(n+1) * b_gnd * (epsdot + eps_reg)**((n+1)/(2*n))

    # pressure melting point, never annotate for initial guess :
    model.solve_hydrostatic_pressure(annotate=False)
    self.calc_T_melt(annotate=False)

    T_s_v   = T_surface.vector().array()
    T_m_v   = T_melt.vector().array()
    theta_s = 146.3*T_s_v + 7.253/2.0*T_s_v**2
    theta_f = 146.3*T_m_v + 7.253/2.0*T_m_v**2
    
    # Surface boundary condition :
    s = "::: calculating energy boundary conditions :::"
    print_text(s, cls=self)

    # initialize the boundary conditions :
    model.init_theta_surface(theta_s, cls=self)
    model.init_theta_app(theta_s,     cls=self)
    model.init_theta_float(theta_f,   cls=self)
    model.init_theta(theta_f,   cls=self)

    # thermal conductivity and heat capacity (Greve and Blatter 2009) :
    ki    = 9.828 * exp(-0.0057*T)
    ci    = 146.3 + 7.253*T
    
    # bulk properties :
    k     =  (1 - W)*ki   + W*kw     # bulk thermal conductivity
    c     =  (1 - W)*ci   + W*cw     # bulk heat capacity
    rho   =  (1 - W)*rhoi + W*rhow   # bulk density
    kappa =  k / (rho*c)             # bulk thermal diffusivity

    # coefficient for diffusion of ice-water mixture -- no water diffusion :
    k_c   = conditional( lt(T, T_w), 1.0, 0.0)

    # frictional heating :
    q_fric = beta * inner(U_v,U_v)

    # Strain heating = stress*strain
    Q_s_gnd = 4 * eta_gnd * epsdot
    Q_s_shf = 4 * eta_shf * epsdot

    # basal heat-flux natural boundary condition :
    g_b = conditional( gt(W, 1.0), 0.0, q_geo + q_fric )
    g_b  = q_geo + q_fric

    # skewed test function in areas with high velocity :
    Unorm  = sqrt(dot(U_v, U_v) + 1e-1)
    PE     = Unorm*h/(2*spy*k/(rho*c))
    tau    = 1/tanh(PE) - 1/PE
    xihat  = xi + h*tau/(2*Unorm) * dot(U_v, grad(xi))
    
    # galerkin formulation :
    theta_a = + rho * dot(U_v, grad(theta)) * xihat * dx \
              - spy * dot(grad(k/c), grad(theta)) * xi * dx \
              + spy * k/c * dot(grad(xi), grad(theta)) * dx \
              - g_b * xi * dBed_g \
              - Q_s_gnd * xi * dx_g \
              - Q_s_shf * xi * dx_f
    
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
    mom_F = derivative(A, U, Phi)

    self.F = theta_a + mom_F

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in G
    self.Jac = derivative(self.F, G, dG)
    
    self.bcs = []
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using divide-lateral boundary conditions -"
      print_text(s, self.color())
      self.bcs.append(DirichletBC(Q3.sub(0).sub(0),
                      model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.bcs.append(DirichletBC(Q3.sub(0).sub(1),
                      model.v_lat, model.ff, model.GAMMA_L_DVD))
    self.bcs.append( DirichletBC(Q3.sub(1), model.theta_app,
                                 model.ff, model.GAMMA_L_DVD) )

    # surface boundary condition : 
    self.bcs.append( DirichletBC(Q3.sub(1), theta_surface,
                                 model.ff, model.GAMMA_S_GND) )
    self.bcs.append( DirichletBC(Q3.sub(1), theta_surface,
                                 model.ff, model.GAMMA_S_FLT) )
    self.bcs.append( DirichletBC(Q3.sub(1), theta_surface,
                                 model.ff, model.GAMMA_U_GND) )
    self.bcs.append( DirichletBC(Q3.sub(1), theta_surface,
                                 model.ff, model.GAMMA_U_FLT) )
    
    # apply T_melt conditions of portion of ice in contact with water :
    self.bcs.append( DirichletBC(Q3.sub(1), theta_float, 
                                 model.ff, model.GAMMA_B_FLT) )
    self.bcs.append( DirichletBC(Q3.sub(1), theta_float, 
                                 model.ff, model.GAMMA_L_UDR) )
    
    
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
    self.b_shf   = b_shf
    self.b_gnd   = b_gnd
    self.A       = A
    self.U       = U 
    self.G       = G
    self.w       = w
    self.dU      = dU
    self.Phi     = Phi
    self.epsdot  = epsdot
    
    self.c       = c
    self.k       = k
    self.rho     = rho
    self.q_fric  = q_fric
    self.Q_s_gnd = Q_s_gnd
    self.Q_s_shf = Q_s_shf
    
  def calc_T_melt(self, annotate=True):
    """
    Calculates pressure-melting point in model.T_melt.
    """
    s    = "::: calculating pressure-melting temperature :::"
    print_text(s, cls=self)

    model = self.model

    dx    = model.dx
    gamma = model.gamma
    T_w   = model.T_w
    p     = model.p

    u   = TrialFunction(model.Q)
    phi = TestFunction(model.Q)
    Tm  = Function(model.Q)

    l = assemble((T_w - gamma * p) * phi * dx, annotate=annotate)
    a = assemble(u * phi * dx, annotate=annotate)

    solve(a, Tm.vector(), l, annotate=annotate)
    model.assign_variable(model.T_melt, Tm, cls=self)

    Tm_v    = Tm.vector().array()
    theta_m = 146.3*Tm_v + 7.253/2.0*Tm_v**2
    model.assign_variable(model.theta_melt, theta_m, cls=self)

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params
  
  def solve(self, annotate=False):
    """ 
    Solve the energy equations, saving enthalpy to model.theta, temperature 
    to model.T, and water content to model.W.
    """
    model      = self.model
    mesh       = model.mesh
    Q          = model.Q
    T_melt     = model.T_melt
    T          = model.T
    W          = model.W
    W0         = model.W0
    L          = model.L
    c          = self.c
    params     = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s      = "::: solving monolithic with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.F == 0, self.G, J = self.Jac, bcs = self.bcs,
          annotate = annotate, solver_parameters = params['solver'])
    U, theta = self.G.split()
    u, v     = split(U)
  
    self.assx.assign(model.u,     u,     annotate=False)
    self.assy.assign(model.v,     v,     annotate=False)
    self.asst.assign(model.theta, theta, annotate=False)

    # solve for the vertical velocity :
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate)
    
    U3 = model.U3.split(True)
    print_min_max(U3[0], 'u', cls=self)
    print_min_max(U3[1], 'v', cls=self)
    print_min_max(U3[2], 'w', cls=self)

    # temperature solved with quadradic formula, using expression for c : 
    s = "::: calculating temperature :::"
    print_text(s, cls=self)
    theta_v  = theta.vector().array()
    T_n_v    = (-146.3 + np.sqrt(146.3**2 + 2*7.253*theta_v)) / 7.253
    T_v      = T_n_v.copy()
    
    # update temperature for wet/dry areas :
    T_melt_v     = T_melt.vector().array()
    theta_melt_v = model.theta_melt.vector().array()
    warm         = theta_v >= theta_melt_v
    cold         = theta_v <  theta_melt_v
    T_v[warm]    = T_melt_v[warm]
    model.assign_variable(T, T_v, cls=self)
    
    # water content solved diagnostically :
    s = "::: calculating water content :::"
    print_text(s, cls=self)
    W_v  = (theta_v - theta_melt_v) / L(0)
    
    # update water content :
    #W_v[cold]       = 0.0   # no water where frozen 
    W_v[W_v < 0.0]  = 0.0    # no water where frozen, please.
    #W_v[W_v > 1.0]  = 1.0    # capped at 100% water, i.e., no hot water.
    model.assign_variable(W0, W, cls=self, save=False)
    model.assign_variable(W,  W_v, cls=self)

  def solve_vert_velocity(self, annotate=annotate):
    """ 
    Solve for vertical velocity w.
    """
    model  = self.model
    
    s    = "::: solving Dukowicz reduced vertical velocity :::"
    print_text(s, self.color())
    w = project(self.w, model.Q, annotate=annotate)

    self.assz.assign(model.w, w, annotate=False)
   
  def solve_basal_melt_rate(self):
    """
    Solve for the basal melt rate stored in model.Mb.
    """ 
    # calculate melt-rate : 
    s = "::: solving for basal melt-rate :::"
    print_text(s, cls=self)
    
    model    = self.model
    B        = model.B
    rhoi     = model.rhoi
    theta    = model.theta
    T        = model.T
    T_melt   = model.T_melt
    L        = model.L
    q_geo    = model.q_geo
    rho      = self.rho
    k        = self.k
    c        = self.c
    q_fric   = self.q_fric

    gradB = as_vector([B.dx(0), B.dx(1), -1])
    dTdn  = k/c * dot(grad(theta), gradB)
    nMb   = project((q_geo + q_fric - dTdn) / (L*rho), model.Q,
                    annotate=False)
    nMb_v    = nMb.vector().array()
    T_melt_v = T_melt.vector().array()
    T_v      = T.vector().array()
    nMb_v[T_v < T_melt_v] = 0.0    # if frozen, no melt
    nMb_v[model.shf_dofs] = 0.0    # does apply over floating regions
    model.assign_variable(model.Mb, nMb_v, cls=self)

  def calc_bulk_density(self):
    """
    Calculate the bulk density stored in model.rho_b.
    """
    # calculate bulk density :
    s = "::: calculating bulk density :::"
    print_text(s, cls=self)
    model       = self.model
    rho_b       = project(self.rho, annotate=False)
    model.assign_variable(model.rhob, rho_b, cls=self)
  
  def color(self):
    """
    return the default color for this class.
    """
    return '213'
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
                                  'relative_tolerance'       : 1e-8,
                                  'relaxation_parameter'     : 1.0,
                                  'maximum_iterations'       : 25,
                                  'error_on_nonconvergence'  : False}}
    m_params  = {'solver'      : nparams}
    return m_params
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.F

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
    return the strain-rate tensor for the velocity <U>.
    """
    u,v,w  = U
    epi    = 0.5 * (grad(U) + grad(U).T)
    epi22  = -u.dx(0) - v.dx(1)          # incompressibility
    epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi[0,2]],
                        [epi[1,0],  epi[1,1],  epi[1,2]],
                        [epi[2,0],  epi[2,1],  epi22]])
    return epsdot

  def stress_tensor(self, U, p, eta):
    """
    return the BP Cauchy stress tensor.
    """
    s   = "::: forming the Cauchy stress tensor :::"
    print_text(s, self.color())

    I     = Identity(3)
    tau   = self.deviatoric_stress_tensor(U, eta)

    sigma = tau - p*I
    return sigma
    
  def deviatoric_stress_tensor(self, U, eta):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
    print_text(s, self.color)

    epi = self.strain_rate_tensor(U)
    tau = 2 * eta * epi
    return tau
  
  def effective_stress(self, U, eta):
    """
    return the effective stress squared.
    """
    tau    = self.deviatoric_stress_tensor(U, eta)
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
  

