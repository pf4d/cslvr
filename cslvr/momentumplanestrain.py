from fenics               import *
from dolfin_adjoint       import *
from cslvr.io             import print_text, print_min_max
from cslvr.d2model        import D2Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
import sys


class MomentumDukowiczPlaneStrain(Momentum):
  """  
  """
  def __init__(self, model, solve_params=None, isothermal=True,
               linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING DUKOWICZ-PLANE=STRAIN PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D3Model:
      s = ">>> MomentumDukowiczPlaneStrain REQUIRES A 'D2Model' INSTANCE, " \
          + "NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    self.solve_params = solve_params
    self.linear       = linear

    # momenturm and adjoint :
    U      = Function(model.Q3, name = 'G')
    Lam    = Function(model.Q3, name = 'Lam')
    dU     = TrialFunction(model.Q3)
    Phi    = TestFunction(model.Q3)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.Q2.sub(0), model.Q3.sub(0))
    self.assz  = FunctionAssigner(model.Q2.sub(2), model.Q3.sub(1))
    self.assp  = FunctionAssigner(model.Q,         model.Q3.sub(2))

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
    dLat_to    = model.dLat_to
    dLat_d     = model.dLat_d
    dLat       = model.dLat
    dBed       = model.dBed
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
    phi, xsi, kappa = Phi
    du,  dw,  dP    = dU
    u,   w,   p     = U
    
    eps_reg    = model.eps_reg
    n          = model.n
    
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
   
    # 1) Viscous dissipation
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot_l  = self.effective_strain_rate(model.U2.copy(True))
      epsdot    = self.effective_strain_rate(as_vector([u,w]))
      eta_shf   = 0.5 * b_shf * (epsdot_l + eps_reg)**((1-n)/(2*n))
      eta_gnd   = 0.5 * b_gnd * (epsdot_l + eps_reg)**((1-n)/(2*n))
      Vd_shf    = 2 * eta_shf * epsdot
      Vd_gnd    = 2 * eta_gnd * epsdot
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot  = self.effective_strain_rate(as_vector([u,0,w]))
      eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      Vd_shf  = (2*n)/(n+1) * b_shf * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd  = (2*n)/(n+1) * b_gnd * (epsdot + eps_reg)**((n+1)/(2*n))
   
    # 2) potential energy :
    Pe     = - rhoi * g * w

    # 3) dissipation by sliding :
    Sl_gnd = - 0.5 * beta * (u**2 + w**2)

    # 4) incompressibility constraint :
    Pc     = p * (u.dx(0) + w.dx(2)) 
    
    # 5) inpenetrability constraint :
    Nc     = - p * (u*N[0] + w*N[2])

    # 6) pressure boundary :
    Pb_w   = - rhosw*g*D * (u*N[0] + w*N[2])
    Pb_l   =   rhoi*g*(S - z) * (u*N[0] + w*N[2])

    # 7) stabilization :
    f       = rhoi * Constant((0.0, 0.0, -g))
    tau_shf = h**2 / (12 * b_shf * rhoi**2)
    tau_gnd = h**2 / (12 * b_gnd * rhoi**2)
    #tau_shf = h**2 / (12 * eta_shf)
    #tau_gnd = h**2 / (12 * eta_gnd)
    Lsq_shf = tau_shf * dot( (grad(p) - f), (grad(p) - f) )
    Lsq_gnd = tau_gnd * dot( (grad(p) - f), (grad(p) - f) )
    
    # Variational principle
    #A      = + Vd_shf*dx_f + Vd_gnd*dx_g \
    A      = + (Vd_shf - Lsq_shf)*dx_f + (Vd_gnd - Lsq_gnd)*dx_g \
             - (Pe + Pc)*dx - Nc*dBed - Sl_gnd*dBed_g - Pb_w*dBed_f
    
    if (not model.use_periodic_boundaries 
        and not use_lat_bcs and use_pressure_bc):
      s = "    - using cliff-pressure boundary condition -"
      print_text(s, self.color())
      A -= Pb_w*dLat_t - Pb_l*dLat_to - Pb_l*dLat_d

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.mom_F = derivative(A, U, Phi)

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []

    # pressure is zero on the surface :
    self.mom_bcs.append(DirichletBC(model.Q4.sub(3),
                        Constant(0.0), model.ff, model.GAMMA_S_GND))
    self.mom_bcs.append(DirichletBC(model.Q4.sub(3),
                        Constant(0.0), model.ff, model.GAMMA_S_FLT))
    self.mom_bcs.append(DirichletBC(model.Q4.sub(3),
                        Constant(0.0), model.ff, model.GAMMA_U_GND))
    self.mom_bcs.append(DirichletBC(model.Q4.sub(3),
                        Constant(0.0), model.ff, model.GAMMA_U_FLT))
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using divide-lateral boundary conditions -"
      print_text(s, self.color())

      self.mom_bcs.append(DirichletBC(Q4.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(Q4.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(Q4.sub(2),
                          model.w_lat, model.ff, model.GAMMA_L_DVD))
    
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
    self.A       = A
    self.U       = U 
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
    self.epsdot  = epsdot
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.A

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
    return the strain-rate tensor of self.U.
    """
    epsdot = 0.5 * (grad(U) + grad(U).T)
    return epsdot

  def stress_tensor(self, U, p, eta):
    """
    return the Cauchy stress tensor.
    """
    s   = "::: forming the Cauchy stress tensor :::"
    print_text(s, self.color())

    I     = Identity(3)
    tau   = self.deviatoric_stress_tensor(U, eta)

    sigma = tau - p*I
    return sigma
    
  def deviatoric_stress_tensor(self, U, eta):
    """
    return the deviatoric stress tensor.
    """
    s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
    print_text(s, self.color())

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

  def solve(self, annotate=True):
    """ 
    Perform the Newton solve of the full-Stokes equations 
    """
    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s    = "::: solving Dukowicz-Brinkerhoff-full-Stokes equations" + \
           " with %i max iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v, w, p = self.U.split()
    
    self.assx.assign(model.u, u, annotate=False)
    self.assy.assign(model.v, v, annotate=False)
    self.assz.assign(model.w, w, annotate=False)
    self.assp.assign(model.p, p, annotate=False)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0], 'u',   cls=self)
    print_min_max(U3[1], 'v',   cls=self)
    print_min_max(U3[2], 'w',   cls=self)
    print_min_max(model.p, 'p', cls=self)



