from fenics                 import *
from dolfin_adjoint         import *
from varglas.io             import print_text, print_min_max
from varglas.d3model        import D3Model
from varglas.d2model        import D2Model
from varglas.d1model        import D1Model
from varglas.physics_new    import Physics
from varglas.helper         import VerticalBasis, VerticalFDBasis
import sys


class Energy(Physics):
  """
  Abstract class outlines the structure of an energy conservation.
  """

  def __new__(self, model, *args, **kwargs):
    """
    Creates and returns a new Energy object.
    """
    instance = Physics.__new__(self, model)
    return instance
  
  def color(self):
    """
    return the default color for this class.
    """
    return '213'
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    params  = {'solver'              : 'mumps',
               'use_surface_climate' : False}
    return params

  def solve_surface_climate(self):
    """
    Calculates PDD, surface temperature given current model geometry and 
    saves to model.T_surface.
    """
    s    = "::: solving surface climate :::"
    print_text(s, self.color())
    model = self.model

    T_ma  = model.T_ma
    T_w   = model.T_w
    S     = model.S.vector().array()
    lat   = model.lat.vector().array()
    
    # Apply the lapse rate to the surface boundary condition
    model.assign_variable(model.T_surface, T_ma(S, lat) + T_w)
  
  def solve(self, annotate=True, params=None):
    """ 
    Perform the Newton solve of the energy equation.
    """
    raiseNotDefined()


class Enthalpy(Energy):
  """
  """ 
  
  def __init__(self, model, solve_params=None, transient=False,
               use_lat_bc=False):
    """ 
    Set up energy equation residual. 
    """
    s    = "::: INITIALIZING ENTHALPY PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D3Model:
      s = ">>> Enthalpy REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    r             = model.r
    mesh          = model.mesh
    V             = model.Q3
    Q             = model.Q
    theta         = model.theta
    theta0        = model.theta0
    n             = model.n
    b_gnd         = model.b_gnd
    b_shf         = model.b_shf
    eps_reg       = model.eps_reg
    T             = model.T
    T_melt        = model.T_melt
    Mb            = model.Mb
    L             = model.L
    ci            = model.ci
    cw            = model.cw
    T_w           = model.T_w
    gamma         = model.gamma
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    W             = model.W
    R             = model.R
    U             = model.U3
    u             = model.u
    v             = model.v
    w             = model.w
    eps_reg       = model.eps_reg
    rhoi          = model.rhoi
    rhow          = model.rhow
    g             = model.g
    beta          = model.beta
    ki            = model.ki
    kw            = model.kw
    T_surface     = model.T_surface
    theta_surface = model.theta_surface
    theta_float   = model.theta_float
    q_geo         = model.q_geo
    thetahat      = model.thetahat
    uhat          = model.uhat
    vhat          = model.vhat
    what          = model.what
    mhat          = model.mhat
    spy           = model.spy
    h             = model.h
    ds            = model.ds
    dSrf          = model.dSrf
    dGnd          = model.dGnd
    dFlt          = model.dFlt
    dSde          = model.dSde
    dBed          = model.dBed
    dx            = model.dx
    dx_s          = model.dx_s
    dx_g          = model.dx_g
    
    # define test and trial functions : 
    psi    = TestFunction(Q)
    dtheta = TrialFunction(Q)

    # pressure melting point, do not annotate for initial guess :
    self.calc_T_melt(annotate=False)

    T_s_v = T_surface.vector().array()
    T_m_v = T_melt.vector().array()
   
    # Surface boundary condition :
    s = "::: calculating energy boundary conditions :::"
    print_text(s, self.color())

    model.assign_variable(theta_surface, T_s_v * ci(0))
    model.assign_variable(theta_float,   T_m_v * ci(0))
    print_min_max(theta_surface, 'theta_GAMMA_S')
    print_min_max(theta_float,   'theta_GAMMA_B_SHF')

    # For the following heat sources, note that they differ from the 
    # oft-published expressions, in that they are both multiplied by constants.
    # I think that this is the correct form, as they must be this way in order 
    # to conserve energy.  This also implies that heretofore, models have been 
    # overestimating frictional heat, and underestimating strain heat.

    # frictional heating :
    q_friction = beta * inner(U,U)

    # Strain heating = stress*strain
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

    epsdot  = model.effective_strain_rate()
    eta_shf = 0.5 * b_shf * epsdot**((1-n)/(2*n))
    eta_gnd = 0.5 * b_gnd * epsdot**((1-n)/(2*n))
    #Q_s_gnd = (2*n)/(n+1) * eta_shf * epsdot
    #Q_s_shf = (2*n)/(n+1) * eta_gnd * epsdot
    Q_s_gnd = 4 * eta_gnd * epsdot
    Q_s_shf = 4 * eta_shf * epsdot

    # thermal conductivity (Greve and Blatter 2009) :
    ki    =  9.828 * exp(-0.0057*T)
    
    # bulk properties :
    k     =  (1 - W)*ki   + W*kw     # bulk thermal conductivity
    c     =  (1 - W)*ci   + W*cw     # bulk heat capacity
    rho   =  (1 - W)*rhoi + W*rhow   # bulk density
    kappa =  k / (rho*c)             # bulk thermal diffusivity

    # configure the module to run in steady state :
    if not transient:
      # skewed test function in areas with high velocity :
      Unorm  = sqrt(dot(U, U) + DOLFIN_EPS)
      PE     = Unorm*h/(2*kappa)
      tau    = 1/tanh(PE) - 1/PE
      T_c    = conditional( lt(Unorm, 4), 0.0, 1.0 )
      psihat = psi + T_c*h*tau/(2*Unorm) * dot(U, grad(psi))

      # residual of model :
      theta_a = + rho * dot(U, grad(dtheta)) * psihat * dx \
                + rho * spy * kappa * dot(grad(psi), grad(dtheta)) * dx \
      
      theta_L = + (q_geo + q_friction) * psihat * dGnd \
                + Q_s_gnd * psihat * dx_g \
                + Q_s_shf * psihat * dx_s
      
    # configure the module to run in transient mode :
    else:
      dt      = model.time_step

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
    self.theta_bc.append( DirichletBC(Q, theta_surface, model.ff, 2) )
    self.theta_bc.append( DirichletBC(Q, theta_surface, model.ff, 6) )
    
    # apply T_w conditions of portion of ice in contact with water :
    self.theta_bc.append( DirichletBC(Q, theta_float,   model.ff, 5) )
    
    # apply lateral boundaries if desired : 
    if use_lat_bc:
      self.theta_bc.append( DirichletBC(Q, theta_surface, model.ff, 4) )

    self.c          = c
    self.k          = k
    self.rho        = rho
    self.kappa      = kappa
    self.q_friction = q_friction
    
  def calc_T_melt(self, annotate=True):
    """
    Calculates pressure-melting point in model.T_melt.
    """
    s    = "::: calculating pressure-melting temperature :::"
    print_text(s, self.color())

    model = self.model

    dx  = model.dx
    x   = model.x
    S   = model.S
    g   = model.gamma
    T_w = model.T_w

    u   = TrialFunction(model.Q)
    phi = TestFunction(model.Q)

    l = assemble((T_w - g * (S - x[2])) * phi * dx)
    a = assemble(u * phi * dx)

    solve(a, model.T_melt.vector(), annotate=annotate)
    print_min_max(model.T_melt, 'T_melt')

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
    theta      = model.theta
    T          = model.T
    W          = model.W
    W0         = model.W0
    L          = model.L
    ci         = model.ci

    if self.solve_params['use_surface_climate']:
      self.solve_surface_climate()
    
    # solve the linear equation for energy :
    s    = "::: solving energy :::"
    print_text(s, self.color())
    aw        = assemble(self.theta_a)
    Lw        = assemble(self.theta_L)
    for bc in self.theta_bc:
      bc.apply(aw, Lw)
    theta_solver = LUSolver(self.solve_params['solver'])
    theta_solver.solve(aw, theta.vector(), Lw, annotate=annotate)
    #solve(self.theta_a == self.theta_L, theta, self.theta_bc,
    #      solver_parameters = {"linear_solver" : sm}, annotate=False)
    print_min_max(theta, 'theta')

    # temperature solved diagnostically : 
    s = "::: calculating temperature :::"
    print_text(s, self.color())
    T_n  = project(theta/ci, Q, annotate=False)
    
    # update temperature for wet/dry areas :
    T_n_v        = T_n.vector().array()
    T_melt_v     = T_melt.vector().array()
    warm         = T_n_v >= T_melt_v
    cold         = T_n_v <  T_melt_v
    T_n_v[warm]  = T_melt_v[warm]
    model.assign_variable(T, T_n_v)
    print_min_max(T,  'T')
    
    # water content solved diagnostically :
    s = "::: calculating water content :::"
    print_text(s, self.color())
    W_n  = project((theta - ci*T_melt)/L, Q, annotate=False)
    
    # update water content :
    W_v             = W_n.vector().array()
    W_v[cold]       = 0.0
    W_v[W_v < 0.0]  = 0.0
    W_v[W_v > 0.01] = 0.01  # for rheology; instant water run-off
    model.assign_variable(W0, W)
    model.assign_variable(W,  W_v)
    print_min_max(W,  'W')
   
  def solve_basal_melt_rate(self):
    """
    Solve for the basal melt rate stored in model.Mb.
    """ 
    # calculate melt-rate : 
    s = "::: solving for basal melt-rate :::"
    print_text(s, self.color())
    
    model      = self.model
    B          = model.B
    rhoi       = model.rhoi
    theta      = model.theta
    L          = model.L
    q_geo      = model.q_geo
    rho        = self.rho
    kappa      = self.kappa
    q_friction = self.q_friction

    gradB = as_vector([B.dx(0), B.dx(1), -1])
    dHdn  = rho * kappa * dot(grad(theta), gradB)
    nMb   = project((q_geo + q_friction - dHdn) / (L*rhoi), model.Q,
                    annotate=False)
    nMb_v = nMb.vector().array()
    #nMb_v[nMb_v < 0.0]  = 0.0
    #nMb_v[nMb_v > 10.0] = 10.0
    model.assign_variable(model.Mb, nMb_v)
    print_min_max(model.Mb, 'Mb')

  def calc_bulk_density(self):
    """
    Calculate the bulk density stored in model.rho_b.
    """
    # calculate bulk density :
    s = "::: calculating bulk density :::"
    print_text(s, self.color())
    model       = self.model
    rho_b       = project(self.rho, annotate=False)
    model.assign_variable(model.rho_b, rho_b)
    print_min_max(model.rho_b,'rho_b')


class EnergyHybrid(Energy):
  """
  New 2D hybrid model.
  """
  def __init__(self, model, transient=False, solve_params=None):
    """ 
    Set up energy equation residual. 
    """
    s    = "::: INITIALIZING HYBRID ENERGY PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D2Model:
      s = ">>> EnergyHybrid REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    # CONSTANTS
    year    = model.spy
    g       = model.g
    n       = model.n
            
    k       = model.ki
    rho     = model.rhoi
    Cp      = model.ci
    kappa   = year*k/(rho*Cp)
            
    q_geo   = model.q_geo
    S       = model.S
    B       = model.B
    beta    = model.beta
    T_s     = model.T_surface
    T_w     = model.T_w
    U       = model.UHV
    H       = model.H
    H0      = model.H0
    T_      = model.T_
    T0_     = model.T0_
    deltax  = model.deltax
    sigmas  = model.sigmas
    eps_reg = model.eps_reg
    h       = model.h
    dt      = model.time_step
    N_T     = model.N_T
    
    Bc      = 3.61e-13*year
    Bw      = 1.73e3*year  # model.a0 ice hardness
    Qc      = 6e4
    Qw      = model.Q0     # ice act. energy
    Rc      = model.R      # gas constant
    gamma   = model.gamma  # pressure melting point depth dependence
   
    # get velocity components : 
    # ANSATZ    
    coef  = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    dcoef = [lambda s:0.0, lambda s:5*s**3]
    
    u_   = [U[0], U[2]]
    v_   = [U[1], U[3]]
    
    u    = VerticalBasis(u_, coef, dcoef)
    v    = VerticalBasis(v_, coef, dcoef)
    
    # FUNCTION SPACES
    Q = model.Q
    Z = model.Z
    
    # ENERGY BALANCE 
    Psi = TestFunction(Z)
    dT  = TrialFunction(Z)

    T  = VerticalFDBasis(T_,  deltax, coef, sigmas)
    T0 = VerticalFDBasis(T0_, deltax, coef, sigmas)

    # METRICS FOR COORDINATE TRANSFORM
    def dsdx(s):
      return 1./H*(S.dx(0) - s*H.dx(0))
    
    def dsdy(s):
      return 1./H*(S.dx(1) - s*H.dx(1))
    
    def dsdz(s):
      return -1./H
    
    def epsilon_dot(s):
      return ( + (u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
               + (v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
               + (u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
               + 0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
               + (+ (u.dx(s,1) + u.ds(s)*dsdy(s)) \
                  + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
               + eps_reg)
    
    def A_v(T):
      return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))
    
    def eta_v(s):
      return A_v(T0.eval(s))**(-1./n)/2.*epsilon_dot(s)**((1.-n)/(2*n))
    
    def w(s):
      w_0 = (U[0].dx(0) + U[1].dx(1))*(s-1.)
      w_2 = + (U[2].dx(0) + U[3].dx(1))*(s**(n+2) - s)/(n+1) \
            + (n+2)/H*U[2]*(1./(n+1)*(s**(n+1) - 1.)*S.dx(0) \
            - 1./(n+1)*(s**(n+2) - 1.)*H.dx(0)) \
            + (n+2)/H*U[3]*(+ 1./(n+1)*(s**(n+1) - 1.)*S.dx(1) \
                            - 1./(n+1)*(s**(n+2) - 1.)*H.dx(1))
      return (u(1)*B.dx(0) + v(1)*B.dx(1)) - 1./dsdz(s)*(w_0 + w_2) 
    
    R_T = 0

    for i in range(N_T):
      # SIGMA COORDINATE
      s = i/(N_T-1.0)
    
      # EFFECTIVE VERTICAL VELOCITY
      w_eff = u(s)*dsdx(s) + v(s)*dsdy(s) + w(s)*dsdz(s)

      if transient:
        w_eff += 1.0/H*(1.0 - s)*(H - H0)/dt
    
      # STRAIN HEAT
      #Phi_strain = (2*n)/(n+1)*2*eta_v(s)*epsilon_dot(s)
      Phi_strain = 4*eta_v(s)*epsilon_dot(s)
    
      # STABILIZATION SCHEME
      Umag   = sqrt(u(s)**2 + v(s)**2 + 1e-3)
      tau    = h/(2*Umag)
      Psihat = Psi[i] + tau*(u(s)*Psi[i].dx(0) + v(s)*Psi[i].dx(1))
    
      # SURFACE BOUNDARY
      if i==0:
        R_T += Psi[i]*(T(i) - T_s)*dx
      # BASAL BOUNDARY
      elif i==(N_T-1):
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 
        R_T += -w_eff*q_geo/(rho*Cp*kappa*dsdz(s))*Psi[i]*dx
        f    = (q_geo + beta*(u(s)**2 + v(s)**2))/(rho*Cp*kappa*dsdz(s))
        R_T += -2.*kappa*dsdz(s)**2*(+ (T(N_T-2) - T(N_T-1)) / deltax**2 \
                                     - f/deltax)*Psi[i]*dx
      # INTERIOR
      else:
        R_T += -kappa*dsdz(s)**2.*T.d2s(i)*Psi[i]*dx
        R_T += w_eff*T.ds(i)*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 
   
      if transient: 
        dTdt = (T(i) - T0(i))/dt
        R_T += dTdt*Psi[i]*dx
    
    # PRETEND THIS IS LINEAR (A GOOD APPROXIMATION IN THE TRANSIENT CASE)
    self.R_T = replace(R_T, {T_:dT})

    # pressure melting point calculation, do not annotate for initial calc :
    self.Tm  = as_vector([T_w - gamma*sigma*H for sigma in sigmas])
    self.calc_T_melt(annotate=False)

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def default_ffc_options(self):
    """ 
    Returns a set of default ffc options that yield good performance
    """
    ffc_options = {"optimize"               : True,
                   "eliminate_zeros"        : True,
                   "precompute_basis_const" : True,
                   "precompute_ip_const"    : True}
    return ffc_options

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    m_params  = {'solver'      : {'linear_solver': 'mumps'},
                 'ffc_params'  : self.default_ffc_options()}
    return m_params

  def solve(self, annotate=True):
    """
    Solves for hybrid energy.
    """
    s    = "::: solving 'EnergyHybrid' for temperature :::"
    print_text(s, self.color())
    
    model  = self.model

    # SOLVE TEMPERATURE
    solve(lhs(self.R_T) == rhs(self.R_T), model.T_,
          solver_parameters=self.solve_params['solver'],
          form_compiler_parameters=self.solve_params['ffc_params'],
          annotate=annotate)
    print_min_max(model.T_, 'T_')

    #  correct for pressure melting point :
    T_v                 = model.T_.vector().array()
    T_melt_v            = model.Tm.vector().array()
    T_v[T_v > T_melt_v] = T_melt_v[T_v > T_melt_v]
    model.assign_variable(model.T_, T_v)
    
    out_T = model.T_.split(True)            # deepcopy avoids projections
    
    model.assign_variable(model.Ts, out_T[0])
    model.assign_variable(model.Tb, out_T[-1]) 

    print_min_max(model.Ts, 'T_S')
    print_min_max(model.Tb, 'T_B')

  def calc_T_melt(self, annotate=True):
    """
    Calculates pressure-melting point in model.T_melt.
    """
    s    = "::: calculating pressure-melting temperature :::"
    print_text(s, self.color())

    model   = self.model
    
    T_melt  = project(self.Tm, annotate=annotate)
    
    Tb_m    = T_melt.split(True)[-1]  # deepcopy avoids projections
    model.assign_variable(model.T_melt, Tb_m)
    model.assign_variable(model.Tm,     T_melt)
    print_min_max(model.T_melt, 'T_melt')
    print_min_max(model.Tm,     'Tm')
    
  def solve_basal_melt_rate(self):
    """
    Solve for the basal melt rate stored in model.Mb.
    """ 
    # FIXME: need to form an expression for dT/dz
    s = "::: calculating basal melt-rate :::"
    print_text(s, self.color())
    
    model  = self.model
    B      = model.B
    rhoi   = model.rhoi
    T      = model.Tb
    L      = model.L
    ki     = model.ki
    q_geo  = model.q_geo
    beta   = model.beta
    u_b    = model.u_b
    v_b    = model.v_b
    w_b    = model.w_b
    q_fric = beta * (u_b**2 + v_b**2 + w_b**2)
    
    gradB = as_vector([B.dx(0), B.dx(1), -1])
    gradT = as_vector([T.dx(0), T.dx(1), 0.0])
    dHdn  = k * dot(gradT, gradB)
    nMb   = project((q_geo + q_friction - dHdn) / (L*rhoi), model.Q,
                    annotate=False)
    nMb_v = nMb.vector().array()
    #nMb_v[nMb_v < 0.0]  = 0.0
    #nMb_v[nMb_v > 10.0] = 10.0
    model.assign_variable(model.Mb, nMb_v)
    print_min_max(model.Mb, 'Mb')


class EnergyFirn(Energy):

  def __init__(self, model, solve_params=None):
    """
    """
    s    = "::: INITIALIZING FIRN ENERGY PHYSICS :::"
    print_text(s, self.color())

    if type(model) != D1Model:
      s = ">>> FirnEnergy REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    mesh    = model.mesh
    Q       = model.Q

    H       = model.H                         # enthalpy
    H_1     = model.H_1                       # previous enthalpy
    T       = model.T                         # temperature
    rho     = model.rho                       # density
    w       = model.w                         # velocity
    m       = model.m                         # mesh velocity
    Kcoef   = model.Kcoef                     # enthalpy ceofficient
    dt      = model.time_step                 # timestep
    rhoi    = model.rhoi                      # density of ice
    spy     = model.spy
    ci      = model.ci
    T       = model.T
    T_      = model.T_w
    L       = model.L
    Hsp     = model.Hsp
    u       = model.u
    p       = model.p
    ql      = model.ql
    etaw    = model.etaw
    rhow    = model.rhow
    T_w     = model.T_w
    #w       = w - m
    z       = model.x
    g       = model.g
    r       = model.r
    S       = model.S
    #omega   = model.omega
    
    psi   = TestFunction(Q)
    dH    = TrialFunction(Q)
    
    # thermal parameters
    ki  = 2.1*(rho / rhoi)**2
    
    T_v = T.vector().array()
    model.assign_variable(H,   ci(0)*T_v)
    model.assign_variable(H_1, ci(0)*T_v)
    
    self.HBc = DirichletBC(Q, model.T_surface*ci,  model.surface)
   
    # Darcy flux :
    omega = conditional(lt(H, Hsp), 0.0, (H - ci*T_w) / L)
    k     = 0.077 * r * exp(-7.8*rho/rhow)                # intrinsic perm.
    phi   = 1 - rho/rhoi                                  # porosity
    Smi   = 0.0057 / (1 - phi) + 0.017                    # irr. water content
    Se    = (omega - Smi) / (1 - Smi)                     # effective sat.
    K     = k * rhow * g / etaw
    krw   = Se**3.0 
    ql    = K * krw * (p / (rhow * g) + z).dx(0)          # water flux
    u     = - k / etaw * p.dx(0)                          # darcy velocity
    u     = - ql/phi                                      # darcy velocity

    # enthalpy residual :
    theta   = 1.0
    H_mid   = theta*H + (1 - theta)*H_1
    delta   = - ki/(rho*ci) * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
              + w * H_mid.dx(0) * psi * dx \
              - (H - H_1)/dt * psi * dx \
              + (ql * H_mid).dx(0) * psi * dx \
    
    # equation to be minimzed :
    self.J  = derivative(delta, H, dH)   # temp/density jacobian

    self.delta = delta
    self.u     = u
    self.ql    = ql
    self.omega = omega
    self.Smi   = Smi

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    params = {'solver' : {'relaxation_parameter'     : 1.0,
                           'maximum_iterations'      : 25,
                           'error_on_nonconvergence' : False,
                           'relative_tolerance'      : 1e-10,
                           'absolute_tolerance'      : 1e-10}}
    return params

  def solve(self, annotate=True):
    """
    """
    s    = "::: solving FirnEnergy :::"
    print_text(s, self.color())
    
    model = self.model

    # newton's iterative method :
    solve(self.delta == 0, model.H, self.HBc, J=self.J, 
          solver_parameters=self.solve_params['solver'],
          annotate=annotate)

    model.assign_variable(model.omega, project(self.omega))
    model.assign_variable(model.ql,    project(self.ql))
    
    T_w   = model.T_w
    rhow  = model.rhow
    rhoi  = model.rhoi
    Hsp   = model.Hsp
    g     = model.g
    ci    = model.ci

    # calculate omega :
    omegap   = model.omega.vector().array()
    omegap_1 = model.omega_1.vector().array()
    domega   = omegap - omegap_1                 # water content change
    model.assign_variable(model.omega_1, model.omega)
    model.assign_variable(model.domega,  domega)

    # update coefficients used by enthalpy :
    Hp       = model.H.vector().array()
    Hhigh    = where(Hp > Hsp(0))[0]
    Hlow     = where(Hp < Hsp(0))[0]
    
    #KcoefNew         = ones(model.dof)
    #KcoefNew[Hhigh]  = 1.0/2.0
    #KcoefNew[Hlow]   = 1.0
    #model.assign_variable(model.Kcoef, KcoefNew)
    
    # calculate T :
    Tp         = Hp / ci(0)
    Tp[Hhigh]  = T_w(0)
    model.assign_variable(model.T, Tp)

    print_min_max(model.T,     'T')
    print_min_max(model.H,     'H')
    print_min_max(model.omega, 'omega')
    print_min_max(model.ql,    'ql')

    p     = model.vert_integrate(rhow * g * model.omega)
    rho   = model.rho
    phi   = 1 - rho/rhoi                                    # porosity
    Smi   = 0.0057 / (1 - phi) + 0.017                      # irr. water content
    model.assign_variable(model.p,   p)
    model.assign_variable(model.u,   project(self.u))
    model.assign_variable(model.Smi, project(Smi))
    print_min_max(model.pp, 'p')
    print_min_max(model.up, 'u')
    print_min_max((1-model.rhop/rhoi(0))*100 - model.omegap*100, 'phi - omega')


 
