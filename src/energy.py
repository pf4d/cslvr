from fenics         import *
from dolfin_adjoint import *
from io             import print_text, print_min_max
from D3Model        import D3Model
from physics_new    import Physics
import sys


class Energy(Physics):
  """
  Abstract class outlines the structure of a momentum calculation.
  """
  
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
  
  def solve(self, annotate=True, params=None):
    """ 
    Perform the Newton solve of the energy equation.
    """
    raiseNotDefined()


class Enthalpy(Energy):
  """
  """ 
  
  def __init__(self, model, solve_params=None, mode='steady',
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

    self.model    = model
                  
    r             = model.r
    mesh          = model.mesh
    V             = model.V
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
    
    # Define test and trial functions       
    psi    = TestFunction(Q)
    dtheta = TrialFunction(Q)

    # Pressure melting point
    self.calc_T_melt()

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

    # Frictional heating :
    q_friction = 0.5 * beta**2 * inner(U,U)

    # Strain heating = stress*strain
    epsdot  = model.effective_strain_rate()
    eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
    eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
    Q_s_gnd = 2 * eta_shf * epsdot
    Q_s_shf = 2 * eta_gnd * epsdot

    # thermal conductivity (Greve and Blatter 2009) :
    ki    =  9.828 * exp(-0.0057*T)
    
    # bulk properties :
    k     =  (1 - W)*ki   + W*kw     # bulk thermal conductivity
    c     =  (1 - W)*ci   + W*cw     # bulk heat capacity
    rho   =  (1 - W)*rhoi + W*rhow   # bulk density
    kappa =  k / (rho*c)             # bulk thermal diffusivity

    # configure the module to run in steady state :
    if mode == 'steady':
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
    elif mode == 'transient':
      dt      = model.time_step
    
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
    else:
      s = ">>> Enthalpy 'mode' PARAMETER MAY BE EITHER 'steady " + \
          "OR 'transient', not %s <<<"
      print_text(s % mode, 'red', 1)
      sys.exit(1)

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
    
  def calc_T_melt(self):
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

    solve(a, model.T_melt.vector(), l, annotate=False)
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


 
