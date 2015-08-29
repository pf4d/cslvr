from fenics              import *
from dolfin_adjoint      import *
from varglas.io          import print_text, print_min_max
from varglas.d2model     import D2Model
from varglas.physics_new import Physics
from varglas.momentum    import Momentum
from varglas.helper      import VerticalBasis, VerticalFDBasis, \
                                VerticalIntegrator
import sys


class MomentumHybrid(Momentum):
  """
  2D hybrid model.
  """
  def __init__(self, model, solve_params=None, linear=False, isothermal=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING HYBRID MOMENTUM PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D2Model:
      s = ">>> MomentumHybrid REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params
    
    self.assx  = FunctionAssigner(model.Q3.sub(0), model.Q)
    self.assy  = FunctionAssigner(model.Q3.sub(1), model.Q)
    self.assz  = FunctionAssigner(model.Q3.sub(2), model.Q)
    
    # CONSTANTS
    year    = model.spy(0)
    rho     = model.rhoi(0)
    g       = model.g(0)
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
    Rc      = model.R     # gas constant
    
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

    if isothermal:
      s = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      def A_v(T):
        return model.b**(-model.n(0)) 
    else:
      s = "    - using temperature-dependent rate-factor -"
      print_text(s, self.color())
      def A_v(T):
        return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))
    
    def epsilon_dot(s):
      return ( + (u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
               + (v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
               + (u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
               + 0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
               + (+ (u.dx(s,1) + u.ds(s)*dsdy(s)) \
                  + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
               + eps_reg)
    
    def eta_v(s):
      return A_v(T0.eval(s))**(-1./n)/2.*epsilon_dot(s)**((1.-n)/(2*n))
    
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
      w_0 = (U[0].dx(0) + U[1].dx(1))*(s-1.)
      w_2 = + (U[2].dx(0) + U[3].dx(1))*(s**(n+2) - s)/(n+1) \
            + (n+2)/H*U[2]*(1./(n+1)*(s**(n+1) - 1.)*S.dx(0) \
            - 1./(n+1)*(s**(n+2) - 1.)*H.dx(0)) \
            + (n+2)/H*U[3]*(+ 1./(n+1)*(s**(n+1) - 1.)*S.dx(1) \
                            - 1./(n+1)*(s**(n+2) - 1.)*H.dx(1))
      return (u(1)*B.dx(0) + v(1)*B.dx(1)) - 1./dsdz(s)*(w_0 + w_2) 
    
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
    nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
                                  'relaxation_parameter'     : 0.7,
                                  'relative_tolerance'       : 1e-5,
                                  'absolute_tolerance'       : 1e7,
                                  'maximum_iterations'       : 20,
                                  'error_on_nonconvergence'  : False,
                                  'report'                   : True}}
    m_params  = {'solver'      : nparams,
                 'ffc_params'  : self.default_ffc_options()}
    return m_params
  
  def solve(self, annotate=True):
    """
    Solves for hybrid velocity.
    """
    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s      = "::: solving hybrid velocity with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac,
          annotate = annotate, solver_parameters = params['solver'],
          form_compiler_parameters = params['ffc_params'])
    print_min_max(self.U, 'U')

    model.UHV.assign(self.U)

    self.assx.assign(model.u,   project(self.u(0.0), model.Q, annotate=False))
    self.assy.assign(model.v,   project(self.v(0.0), model.Q, annotate=False))
    self.assz.assign(model.w,   project(self.w(0.0), model.Q, annotate=False))

    print_min_max(model.U3, 'U3')
    
    self.assx.assign(model.u_b, project(self.u(1.0), model.Q, annotate=False))
    self.assy.assign(model.v_b, project(self.v(1.0), model.Q, annotate=False))
    self.assz.assign(model.w_b, project(self.w(1.0), model.Q, annotate=False))

    print_min_max(model.U3_b, 'U3_b')



