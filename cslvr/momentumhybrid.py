from fenics            import *
from dolfin_adjoint    import *
from cslvr.io          import print_text, print_min_max
from cslvr.hybridmodel import HybridModel
from cslvr.physics     import Physics
from cslvr.momentum    import Momentum
from cslvr.helper      import VerticalBasis, VerticalFDBasis, \
                              VerticalIntegrator
import sys


class MomentumHybrid(Momentum):
  """
  2D hybrid model.
  """
  def initialize(self, model, solve_params=None, isothermal=True,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING HYBRID MOMENTUM PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != HybridModel:
      s = ">>> MomentumHybrid REQUIRES A 'HybridModel' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    self.linear       = linear

    self.assx  = FunctionAssigner(model.u_s.function_space(), model.Q)
    self.assy  = FunctionAssigner(model.v_s.function_space(), model.Q)
    self.assz  = FunctionAssigner(model.w_s.function_space(), model.Q)
    
    # CONSTANTS
    year    = model.spy
    rho     = model.rhoi
    g       = model.g
    n       = model.n(0)
   
    S       = model.S 
    B       = model.B
    H       = S - B
    beta    = model.beta
    eps_reg = model.eps_reg
    #H       = model.H
    #S       = B + H
    deltax  = model.deltax
    sigmas  = model.sigmas
    T0_     = model.T0_
    T_      = model.T_
    
    Bc      = 3.61e-13*year
    Bw      = 1.73e3*year # model.a0 ice hardness
    Qc      = 6e4
    Qw      = model.Q0    # ice act. energy
    R       = model.R     # gas constant
    
    # MOMENTUM
    U       = Function(model.HV, name='U')
    Lam     = Function(model.HV, name='Lam')
    Phi     = TestFunction(model.HV)
    dU      = TrialFunction(model.HV)
    
    # ANSATZ    
    coef    = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    dcoef   = [lambda s:0.0, lambda s:5*s**3]
    
    u_      = [U[0],   U[2]]
    v_      = [U[1],   U[3]]
    phi_    = [Phi[0], Phi[2]]
    psi_    = [Phi[1], Phi[3]]
    
    u       = VerticalBasis(u_,  coef, dcoef)
    v       = VerticalBasis(v_,  coef, dcoef)
    phi     = VerticalBasis(phi_,coef, dcoef)
    psi     = VerticalBasis(psi_,coef, dcoef)
    
    # energy functions :
    T       = VerticalFDBasis(T_,  deltax, coef, sigmas)
    T0      = VerticalFDBasis(T0_, deltax, coef, sigmas)

    # METRICS FOR COORDINATE TRANSFORM
    def dsdx(s):
      return 1./H*(S.dx(0) - s*H.dx(0))
    
    def dsdy(s):
      return 1./H*(S.dx(1) - s*H.dx(1))
    
    def dsdz(s):
      return -1./H

    def A_v(T):
      return model.A 
      #return conditional(le(T,263.15), Bc*exp(-Qc/(R*T)), Bw*exp(-Qw/(R*T)))
    
    def epsilon_dot(s):
      # linearize the viscosity :
      if linear:
        ue    = model.u
        ve    = model.v
      # nonlinear viscosity :
      else:
        ue    = u
        ve    = v
      return ( + (ue.dx(s,0) + ue.ds(s)*dsdx(s))**2 \
               + (ve.dx(s,1) + ve.ds(s)*dsdy(s))**2 \
               +   (ue.dx(s,0) + ue.ds(s)*dsdx(s)) \
                 * (ve.dx(s,1) + ve.ds(s)*dsdy(s)) \
               + 0.25*((ue.ds(s)*dsdz(s))**2 + (ve.ds(s)*dsdz(s))**2 \
               + (+ (ue.dx(s,1) + ue.ds(s)*dsdy(s)) \
                  + (ve.dx(s,0) + ve.ds(s)*dsdx(s)))**2) \
               + eps_reg)
    
    def eta_v(s):
      return 0.5 * A_v(T0.eval(s))**(-1/n) * epsilon_dot(s)**((1-n)/(2*n))
    
    def membrane_xx(s):
      return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*H*eta_v(s) \
             * (+ 4*(u.dx(s,0) + u.ds(s)*dsdx(s)) \
                + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))
    
    def membrane_xy(s):
      return (phi.dx(s,1) + phi.ds(s)*dsdy(s))*H*eta_v(s) \
             * (+ (u.dx(s,1) + u.ds(s)*dsdy(s)) \
                + (v.dx(s,0) + v.ds(s)*dsdx(s)))
    
    def membrane_yx(s):
      return (psi.dx(s,0) + psi.ds(s)*dsdx(s))*H*eta_v(s) \
             * (+ (u.dx(s,1) + u.ds(s)*dsdy(s)) \
                + (v.dx(s,0) + v.ds(s)*dsdx(s)))
    
    def membrane_yy(s):
      return (psi.dx(s,1) + psi.ds(s)*dsdy(s))*H*eta_v(s) \
             * (+ 2*(u.dx(s,0) + u.ds(s)*dsdx(s)) \
                + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))
    
    def shear_xz(s):
      return dsdz(s)**2*phi.ds(s)*H*eta_v(s)*u.ds(s)
    
    def shear_yz(s):
      return dsdz(s)**2*psi.ds(s)*H*eta_v(s)*v.ds(s)
    
    def tau_dx(s):
      return rho*g*H*S.dx(0)*phi(s)
    
    def tau_dy(s):
      return rho*g*H*S.dx(1)*psi(s)
    
    def w(s):
      s   = Constant(s)
      w_0 = (U[0].dx(0) + U[1].dx(1))*(s-1)
      w_2 = + (U[2].dx(0) + U[3].dx(1)) * (s**(n+2) - s)/(n+1) \
            + (n+2)/H*U[2]*(+ 1/(n+1)*(s**(n+1) - 1)*S.dx(0) \
                            - 1/(n+1)*(s**(n+2) - 1)*H.dx(0)) \
            + (n+2)/H*U[3]*(+ 1/(n+1)*(s**(n+1) - 1)*S.dx(1) \
                            - 1/(n+1)*(s**(n+2) - 1)*H.dx(1))
      return (u(1)*B.dx(0) + v(1)*B.dx(1)) - 1/dsdz(s)*(w_0 + w_2)
    
    vi = VerticalIntegrator(order=4)

    R_x = - vi.intz(membrane_xx) \
          - vi.intz(membrane_xy) \
          - vi.intz(shear_xz) \
          - phi(1)*beta*u(1) \
          - vi.intz(tau_dx)
    R_y = - vi.intz(membrane_yx) \
          - vi.intz(membrane_yy) \
          - vi.intz(shear_yz) \
          - psi(1)*beta*v(1) \
          - vi.intz(tau_dy)

    # SIA
    self.mom_F   = (R_x + R_y)*dx
    self.mom_Jac = derivative(self.mom_F, U, dU)

    self.u   = u
    self.v   = v
    self.w   = w
    self.U   = U
    self.dU  = dU
    self.Phi = Phi
    self.Lam = Lam
    
    problem = NonlinearVariationalProblem(
                self.mom_F, self.U, 
                J=self.mom_Jac,
                form_compiler_parameters=self.solve_params['ffc_params'])
    self.solver = NonlinearVariationalSolver(problem)
    self.solver.parameters.update(self.solve_params['solver'])

  def form_obj_ftn(self, integral, kind='log', g1=0.01, g2=1000):
    """
    Forms and returns an objective functional for use with adjoint.
    Saves to self.J.
    """
    #NOTE: this overides base class momentum.form_obj_ftn() due to the
    #      extra complexity of evaluating U on the surface.
    self.obj_ftn_type = kind     # need to save this for printing values.
    self.integral     = integral # this too.
    
    model    = self.model
    
    # differentiate between objective over cells or facets :
    if integral in [model.OMEGA_GND, model.OMEGA_FLT]:
      dJ = model.dx(integral)
    else:
      dJ = model.ds(integral)

    adot     = model.adot
    S        = model.S
    u_ob     = model.u_ob
    v_ob     = model.v_ob
    um       = model.u(0.0)   # surface u
    vm       = model.v(0.0)   # surface v
    u        = self.u(0.0)
    v        = self.v(0.0)
    w        = self.w(0.0)

    if kind == 'log':
      J  = 0.5 * ln(  (sqrt(u**2    + v**2   ) + 0.01) \
                    / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ 
      Jp = 0.5 * ln(  (sqrt(um**2   + vm**2  ) + 0.01) \
                    / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ 
      s   = "::: forming log objective functional :::"
    
    elif kind == 'kinematic':
      #FIXME: no wm (yet), so this don't work.
      J  = 0.5 * (u*S.dx(0) + v*S.dx(1) - (w + adot))**2 * dJ
      Jp = 0.5 * (um*S.dx(0) + vm*S.dx(1) - (wm + adot))**2 * dJ
      s   = "::: forming kinematic objective functional :::"

    elif kind == 'L2':
      J  = 0.5 * ((u  - u_ob)**2 + (v  - v_ob)**2) * dJ
      Jp = 0.5 * ((um - u_ob)**2 + (vm - v_ob)**2) * dJ
      s   = "::: forming L2 objective functional :::"

    elif kind == 'ratio':
      #NOTE: experimental
      U_n   = sqrt(u**2    + v**2    + DOLFIN_EPS)
      U_m   = sqrt(um**2   + vm**2   + DOLFIN_EPS)
      Uob_n = sqrt(u_ob**2 + v_ob**2 + DOLFIN_EPS)
      #J     = 0.5 * (+ (1 - (u + 1e-4)/(u_ob + 1e-4))
      #               + (1 - (v + 1e-4)/(v_ob + 1e-4)) ) * Uob_n/U_n * dJ
      J     = 0.5 * (1 -  (U_n + 0.01) / (Uob_n + 0.01))**2 * dJ
      Jp    = 0.5 * (1 -  (U_m + 0.01) / (Uob_n + 0.01))**2 * dJ
      s     = "::: forming ratio objective functional :::"
    
    elif kind == 'log_L2_hybrid':
      J1  = g1 * 0.5 * ((u - u_ob)**2 + (v - v_ob)**2) * dJ
      J2  = g2 * 0.5 * ln(   (sqrt(u**2    + v**2)    + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ
      self.J1  = 0.5 * ((um - u_ob)**2 + (vm - v_ob)**2) * dJ
      self.J2  = 0.5 * ln(   (sqrt(um**2   + vm**2)   + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ
      self.J1p = g1 * 0.5 * ((um - u_ob)**2 + (vm - v_ob)**2) * dJ
      self.J2p = g2 * 0.5 * ln(   (sqrt(um**2 + vm**2) + 0.01) \
                                / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ
      J  = J1 + J2
      Jp = self.J1p + self.J2p
      s   = "::: forming log/L2 hybrid objective with gamma_1 = " \
            "%.1e and gamma_2 = %.1e :::" % (g1, g2)

    else:
      s = ">>> ADJOINT OBJECTIVE FUNCTIONAL MAY BE 'L2', " + \
          "'log', 'kinematic', OR 'log_L2_hybrid', NOT '%s' <<<" % kind
      print_text(s, 'red', 1)
      sys.exit(1)
    print_text(s, self.color())
    s = "    - integrated over %s -" % model.boundaries[integral]
    print_text(s, self.color())
    self.J  = J
    self.Jp = Jp
    
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

  def get_dU(self):
    """
    Return the trial function for U.
    """
    return self.dU

  def get_Phi(self):
    """
    Return the test function for U.
    """
    return self.Phi

  def get_Lam(self):
    """
    Return the adjoint function for U.
    """
    return self.Lam

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
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-10,
                'relaxation_parameter'     : 1.0,
                'absolute_tolerance'       : 1.0,
                'maximum_iterations'       : 20,
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
    m_params  = {'solver'           : nparams,
                 'ffc_params'       : self.default_ffc_options(),
                 'project_boundary' : True}
    return m_params
  
  def solve(self, annotate=False):
    """
    Solves for hybrid velocity.
    """
    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s      = "::: solving 'MomentumHybrid' with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    #s = solve(self.mom_F == 0, self.U, J = self.mom_Jac,
    #      annotate = annotate, solver_parameters = params['solver'],
    #      form_compiler_parameters = params['ffc_params'])
    out = self.solver.solve(annotate=annotate)
    print_min_max(self.U, 'U', self.color())

    model.U3.assign(self.U, annotate=annotate)

    if params['project_boundary']:
      self.assx.assign(model.u_s, project(self.u(0.0), model.Q,
                       annotate=annotate), annotate=annotate)
      self.assy.assign(model.v_s, project(self.v(0.0), model.Q,
                       annotate=annotate), annotate=annotate)
      self.assz.assign(model.w_s, project(self.w(0.0), model.Q,
                       annotate=annotate), annotate=annotate)

      print_min_max(model.U3_s, 'U3_S', self.color())

      self.assx.assign(model.u_b, project(self.u(1.0), model.Q,
                       annotate=annotate), annotate=annotate)
      self.assy.assign(model.v_b, project(self.v(1.0), model.Q,
                       annotate=annotate), annotate=annotate)
      self.assz.assign(model.w_b, project(self.w(1.0), model.Q,
                       annotate=annotate), annotate=annotate)

      print_min_max(model.U3_b, 'U3_B', self.color())

    return out



