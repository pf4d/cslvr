from fenics         import *
from dolfin_adjoint import *
from io             import print_text, print_min_max
from physics_new    import Physics
import sys


class Momentum(Physics):
  """
  Abstract class outlines the structure of a momentum calculation.
  """

  def __new__(self, model, *args, **kwargs):
    """
    Creates and returns a new momentum object.
    """
    instance = Physics.__new__(self, model)
    return instance
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'cyan'

  def get_residual(self):
    """
    Returns the momentum residual.
    """
    raiseNotDefined()

  def get_U(self):
    """
    Return the velocity Function.
    """
    raiseNotDefined()

  def get_dU(self):
    """
    Return the trial function for U.
    """
    raiseNotDefined()

  def get_Phi(self):
    """
    Return the test function for U.
    """
    raiseNotDefined()

  def get_Lam(self):
    """
    Return the adjoint function for U.
    """
    raiseNotDefined()
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' : {'linear_solver'            : 'cg',
                                  'preconditioner'           : 'hypre_amg',
                                  'relative_tolerance'       : 1e-8,
                                  'relaxation_parameter'     : 1.0,
                                  'maximum_iterations'       : 25,
                                  'error_on_nonconvergence'  : False}}
    m_params  = {'solver' : nparams}
    return m_params
  
  def solve(self, annotate=True, params=None):
    """ 
    Perform the Newton solve of the momentum equations 
    """
    raiseNotDefined()
  
  def form_obj_ftn(self, kind='log', integral=2, g1=0.01, g2=1000):
    """
    Forms and returns an objective functional for use with adjoint.
    Saves to self.J.
    """
    self.obj_ftn_type = kind   # need to save this for printing values.
    
    model    = self.model

    dGamma   = model.ds(integral)
    u_ob     = model.u_ob
    v_ob     = model.v_ob
    adot     = model.adot
    S        = model.S
    U        = self.get_U()

    if kind == 'log':
      J   = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                      / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dGamma 
      s   = "::: forming log objective function :::"
    
    elif kind == 'kinematic':
      J   = 0.5 * (+ U[0]*S.dx(0) + U[1]*S.dx(1) - (U[2] + adot))**2 * dGamma
      s   = "::: getting kinematic objective function :::"

    elif kind == 'linear':
      J   = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dGamma
      s   = "::: getting linear objective function :::"
    
    elif kind == 'log_lin_hybrid':
      J1  = g1 * 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dGamma
      J2  = g2 * 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dGamma
      self.J1  = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dGamma
      self.J2  = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dGamma
      J   = J1  + J2
      s   = "::: getting log/linear hybrid objective with gamma_1 = " \
            "%.1e and gamma_2 = %.1e :::" % (g1, g2)

    else:
      s = ">>> ADJOINT OBJECTION FUNCTION MAY BE 'linear', " + \
          "'log', 'kinematic', OR 'log_lin_hybrid' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    print_text(s, self.color())
    self.J = J
    return J

  def form_reg_ftn(self, c, kind='Tikhonov', integral=2, alpha=1.0):
    """
    Forms and returns regularization functional for used with adjoint, saved to 
    self.Reg.
    """
    self.alpha = alpha   # need to save this for printing values.

    dR = self.model.ds(integral)
    
    # form regularization term 'R' :
    if kind != 'TV' and kind != 'Tikhonov':
      s    =   ">>> VALID REGULARIZATIONS ARE 'TV' AND 'Tikhonov' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    elif kind == 'TV':
      R  = alpha * 0.5 * sqrt(inner(grad(c), grad(c)) + DOLFIN_EPS) * dR
    elif kind == 'Tikhonov':
      R  = alpha * 0.5 * inner(grad(c), grad(c)) * dR
    s   = "::: forming %s regularization with parameter alpha = %.2E :::"
    print_text(s % (kind, alpha), self.color())
    self.Reg = R
    return R
  
  def print_eval_ftns(self):
    """
    Used to facilitate printing the objective function in adjoint solves.
    """
    if self.obj_ftn_type == 'log_lin_hybrid':
      J1 = assemble(self.J1)
      J2 = assemble(self.J2)
      print_min_max(J1, 'J1')
      print_min_max(J2, 'J2')
    R = assemble(self.Reg)
    J = assemble(self.J)
    print_min_max(R, 'R')
    print_min_max(J, 'J')
    
  def Lagrangian(self):
    """
    Returns the Lagrangian of the momentum equations.
    """
    s  = "::: forming Lagrangian :::"
    print_text(s, self.color())
    
    R   = self.get_residual()
    Phi = self.get_Phi()
    dU  = self.get_dU()

    # this is the adjoint of the momentum residual, the Lagrangian :
    return replace(R, {Phi : dU})

  def Hamiltonian(self, I):
    """
    Returns the Hamiltonian of the momentum equations with objective function
    <I>.
    """
    s  = "::: forming Hamiltonian :::"
    print_text(s, self.color())
    # the Hamiltonian :
    return I + self.Lagrangian()

  def dHdc(self, I, L, c): 
    """
    Returns the derivative of the Hamiltonian consisting of ajoint-computed
    self.Lam values w.r.t. the control variable <c>, i.e., 

       dH    d [                 ]
       -- = -- [ I + L(self.Lam) ]
       dc   dc [                 ]

    """
    s  = "::: forming dHdc :::"
    print_text(s, self.color())
    
    dU  = self.get_dU()
    Lam = self.get_Lam()

    # we need to evaluate the Hamiltonian with the values of Lam computed from
    # self.dI in order to get the derivative of the Hamiltonian w.r.t. the 
    # control variables.  Hence we need a new Lagrangian with the trial 
    # functions replaced with the computed Lam values.
    L_lam  = replace(L, {dU : Lam})

    # the Hamiltonian with unknowns replaced with computed Lam :
    H_lam  = I + L_lam

    # the derivative of the Hamiltonian w.r.t. the control variables in the 
    # direction of a P1 test function :
    return derivative(H_lam, c, TestFunction(self.model.Q))
    
  def solve_adjoint_momentum(self, H):
    """
    Solves for the adjoint variables self.Lam from the Hamiltonian <H>.
    """
    U   = self.get_U()
    Phi = self.get_Phi()
    Lam = self.get_Lam()

    # we desire the derivative of the Hamiltonian w.r.t. the model state U
    # in the direction of the test function Phi to vanish :
    dI = derivative(H, U, Phi)
    
    s  = "::: solving adjoint momentum :::"
    print_text(s, self.color())
    
    aw = assemble(lhs(dI))
    Lw = assemble(rhs(dI))
    
    a_solver = KrylovSolver('cg', 'hypre_amg')
    a_solver.solve(aw, Lam.vector(), Lw, annotate=False)

    #lam_nx, lam_ny = model.Lam.split(True)
    #lam_ix, lam_iy = model.Lam.split()

    #if self.config['adjoint']['surface_integral'] == 'shelves':
    #  lam_nx.vector()[model.gnd_dofs] = 0.0
    #  lam_ny.vector()[model.gnd_dofs] = 0.0
    #elif self.config['adjoint']['surface_integral'] == 'grounded':
    #  lam_nx.vector()[model.shf_dofs] = 0.0
    #  lam_ny.vector()[model.shf_dofs] = 0.0

    ## function assigner translates between mixed space and P1 space :
    #U_sp = model.U.function_space()
    #assx = FunctionAssigner(U_sp.sub(0), lam_nx.function_space())
    #assy = FunctionAssigner(U_sp.sub(1), lam_ny.function_space())

    #assx.assign(lam_ix, lam_nx)
    #assy.assign(lam_iy, lam_ny)
    
    #solve(self.aw == self.Lw, model.Lam,
    #      solver_parameters = {"linear_solver"  : "cg",
    #                           "preconditioner" : "hypre_amg"},
    #      annotate=False)
    #print_min_max(norm(model.Lam), '||Lam||')
    print_min_max(Lam, 'Lam')


