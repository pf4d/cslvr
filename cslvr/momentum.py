from fenics                 import *
from dolfin_adjoint         import *
from cslvr.io               import get_text, print_text, print_min_max
from cslvr.physics          import Physics
from copy                   import deepcopy
from cslvr.helper           import raiseNotDefined
import numpy                    as np
import matplotlib.pyplot        as plt
import sys
import os
import json


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
  
  def __init__(self, model, solve_params=None,
               linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """
    """
    s = "::: INITIALIZING MOMENTUM :::"
    print_text(s, self.color())

    # save the starting values, as other algorithms might change the 
    # values to suit their requirements :
    if isinstance(solve_params, dict):
      pass
    elif solve_params == None:
      solve_params    = self.default_solve_params()
      s = "::: using default parameters :::"
      print_text(s, self.color())
      s = json.dumps(solve_params, sort_keys=True, indent=2)
      print_text(s, '230')
    else:
      s = ">>> Momentum REQUIRES A 'dict' INSTANCE OF SOLVER " + \
          "PARAMETERS, NOT %s <<<"
      print_text(s % type(solve_params) , 'red', 1)
      sys.exit(1)
    
    self.solve_params_s    = deepcopy(solve_params)
    self.linear_s          = linear
    self.use_lat_bcs_s     = use_lat_bcs
    self.use_pressure_bc_s = use_pressure_bc
    
    self.initialize(model, solve_params, linear,
                    use_lat_bcs, use_pressure_bc)
  
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.  Note that any Momentum object *must*
    call this method.  See the existing child Momentum objects for reference.
    """
    raiseNotDefined()
  
  def reset(self):
    """
    reset the momentum to the original configuration.
    """
    s = "::: RE-INITIALIZING MOMENTUM PHYSICS :::"
    print_text(s, self.color())

    s = "::: restoring desired Newton solver parameters :::"
    print_text(s, self.color())
    s = json.dumps(self.solve_params_s, sort_keys=True, indent=2)
    print_text(s, '230')
    
    self.initialize(self.model, solve_params=self.solve_params_s,
                    linear=self.linear_s,
                    use_lat_bcs=self.use_lat_bcs_s, 
                    use_pressure_bc=self.use_pressure_bc_s)

  def linearize_viscosity(self):
    """
    reset the momentum to the original configuration.
    """
    s = "::: RE-INITIALIZING MOMENTUM PHYSICS WITH LINEAR VISCOSITY :::"
    print_text(s, self.color())
   
    # deepcopy the parameters so that we can change them without changing
    # the original values we started with :
    mom_params = deepcopy(self.solve_params_s)
      
    # adjust the parameters for incomplete-adjoint :
    new_params = mom_params['solver']['newton_solver']

    # only affects non-full-stokes formulations :
    mom_params['solve_vert_velocity']     = False
    mom_params['solve_pressure']          = False

    # the linear momentum systems solve much faster :
    new_params['relaxation_parameter']    = 1.0
    new_params['maximum_iterations']      = 2
    new_params['error_on_nonconvergence'] = False

    s = "::: altering solver parameters for optimal convergence :::"
    print_text(s, self.color())
    s = json.dumps(mom_params, sort_keys=True, indent=2)
    print_text(s, '230')

    self.initialize(self.model, solve_params=mom_params,
                    linear=True,
                    use_lat_bcs=self.use_lat_bcs_s, 
                    use_pressure_bc=self.use_pressure_bc_s)
  
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
    m_params  = {'solver'         : nparams,
                 'solve_pressure' : True}
    return m_params
  
  def solve_pressure(self, annotate=False):
    """
    Solve for the hydrostatic pressure 'p'.
    """
    self.model.solve_hydrostatic_pressure(annotate)
  
  def solve(self, annotate=False, params=None):
    """ 
    Perform the Newton solve of the momentum equations 
    """
    raiseNotDefined()

  def unify_eta(self):
    """
    Unifies viscosity defined over grounded and shelves to model.eta.
    """
    s = "::: unifying viscosity on shelf and grounded areas to model.eta :::"
    print_text(s, self.color())
    
    model = self.model
    
    num_shf = MPI.sum(mpi_comm_world(), len(model.shf_dofs))
    num_gnd = MPI.sum(mpi_comm_world(), len(model.gnd_dofs))

    print_min_max(num_shf, 'number of floating vertices')
    print_min_max(num_gnd, 'number of grounded vertices')

    if num_gnd == 0 and num_shf == 0:
      s = "    - floating and grounded regions have not been marked -"
      print_text(s, self.color())

    elif num_gnd == 0:
      s = "    - all floating ice, assigning eta_shf to eta  -"
      print_text(s, self.color())
      model.init_eta(project(self.eta_shf, model.Q))

    elif num_shf == 0:
      s = "    - all grounded ice, assigning eta_gnd to eta -"
      print_text(s, self.color())
      model.init_eta(project(self.eta_gnd, model.Q))

    else: 
      s = "    - grounded and floating ice present, unifying eta -"
      print_text(s, self.color())
      eta_shf = project(self.eta_shf, model.Q)
      eta_gnd = project(self.eta_gnd, model.Q)
     
      # remove areas where viscosities overlap : 
      eta_shf.vector()[model.gnd_dofs] = 0.0
      eta_gnd.vector()[model.shf_dofs] = 0.0
      
      # unify eta to self.eta :
      model.init_eta(eta_shf.vector() + eta_gnd.vector())

  def viscosity(self, U):
    """
    calculates the viscosity saved to self.eta_shf and self.eta_gnd, for
    floating and grounded ice, respectively.  Uses velocity vector <U> with
    components u,v,w.  If <linear> == True, form viscosity from model.U3.
    """
    s  = "::: forming visosity :::"
    print_text(s, self.color())
    model    = self.model
    n        = model.n
    A_shf    = model.A_shf
    A_gnd    = model.A_gnd
    eps_reg  = model.eps_reg
    epsdot   = self.effective_strain_rate(U)
    eta_shf  = 0.5 * A_shf**(-1/n) * (epsdot + eps_reg)**((1-n)/(2*n))
    eta_gnd  = 0.5 * A_gnd**(-1/n) * (epsdot + eps_reg)**((1-n)/(2*n))
    return (eta_shf, eta_gnd)

  def calc_q_fric(self):
    """
    Solve for the friction heat term stored in model.q_fric.
    """ 
    # calculate melt-rate : 
    s = "::: solving basal friction heat :::"
    print_text(s, cls=self)
    
    model    = self.model
    u,v,w    = model.U3.split(True)

    beta_v   = model.beta.vector().array()
    u_v      = u.vector().array()
    v_v      = v.vector().array()
    w_v      = w.vector().array()
    Fb_v     = model.Fb.vector().array()

    q_fric_v = beta_v * (u_v**2 + v_v**2 + (w_v+Fb_v)**2)
    
    model.init_q_fric(q_fric_v, cls=self)
    
  def form_obj_ftn(self, integral, kind='log', g1=0.01, g2=1000):
    """
    Forms and returns an objective functional for use with adjoint.
    Saves to self.J.
    """
    self.obj_ftn_type = kind     # need to save this for printing values.
    self.integral     = integral # this too.
    
    model    = self.model
    
    # note that even if self.reset() or self.linearize_viscosity() are called,
    # this will still point to the velocity function, and hence only one call
    # to this function is required :
    U        = self.get_U()

    # differentiate between objective over cells or facets :
    if integral in [model.OMEGA_GND, model.OMEGA_FLT]:
      dJ = model.dx(integral)
    else:
      dJ = model.ds(integral)

    u_ob     = model.u_ob
    v_ob     = model.v_ob
    adot     = model.adot
    S        = model.S
    U3       = model.U3
    um       = U3[0]
    vm       = U3[1]
    wm       = U3[2]

    if kind == 'log':
      J  = 0.5 * ln(  (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                        / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ 
      Jp = 0.5 * ln(  (sqrt(um**2 + vm**2) + 0.01) \
                    / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ 
      s   = "::: forming log objective functional :::"
    
    elif kind == 'kinematic':
      J  = 0.5 * (U[0]*S.dx(0) + U[1]*S.dx(1) - (U[2] + adot))**2 * dJ
      Jp = 0.5 * (um*S.dx(0) + vm*S.dx(1) - (wm + adot))**2 * dJ
      s   = "::: forming kinematic objective functional :::"

    elif kind == 'L2':
      J  = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dJ
      Jp = 0.5 * ((um - u_ob)**2 + (vm - v_ob)**2) * dJ
      s   = "::: forming L2 objective functional :::"

    elif kind == 'ratio':
      #NOTE: experimental
      U_n   = sqrt(U[0]**2 + U[1]**2 + DOLFIN_EPS)
      U_m   = sqrt(um**2   + vm**2   + DOLFIN_EPS)
      Uob_n = sqrt(u_ob**2 + v_ob**2 + DOLFIN_EPS)
      #J     = 0.5 * (+ (1 - (U[0] + 1e-4)/(u_ob + 1e-4))
      #               + (1 - (U[1] + 1e-4)/(v_ob + 1e-4)) ) * Uob_n/U_n * dJ
      J     = 0.5 * (1 -  (U_n + 0.01) / (Uob_n + 0.01))**2 * dJ
      Jp    = 0.5 * (1 -  (U_m + 0.01) / (Uob_n + 0.01))**2 * dJ
      s     = "::: forming ratio objective functional :::"
    
    elif kind == 'log_L2_hybrid':
      J1  = g1 * 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dJ
      J2  = g2 * 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dJ
      self.J1  = 0.5 * ((um - u_ob)**2 + (vm - v_ob)**2) * dJ
      self.J2  = 0.5 * ln(   (sqrt(um**2 + vm**2) + 0.01) \
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

  def calc_misfit(self):
    """
    Calculates and returns the misfit of model and observations, 

      D = ||U - U_ob||

    over shelves or grounded depending on the paramter <integral> sent to
    the self.form_obj_ftn().
    """
    s   = "::: calculating misfit L-infty norm ||U - U_ob|| over %s :::"
    print_text(s % self.model.boundaries[self.integral], cls=self)

    model    = self.model
    integral = self.integral
    #um,vm,wm = model.U3.split(True)
    #u_ob     = model.u_ob
    #v_ob     = model.v_ob

    # FIXME: bug in FEniCS (issue #405) requires me to do this junk :
    u        = Function(model.Q)
    v        = Function(model.Q)
    assign(u, model.U3.sub(0))
    assign(v, model.U3.sub(1))
    u_ob     = Function(model.Q)
    v_ob     = Function(model.Q)
    assign(u_ob, model.u_ob)
    assign(v_ob, model.v_ob)
    
    # convert everything for low-level manipulations :
    u_v    = u.vector().array()
    v_v    = v.vector().array()
    u_ob_v = u_ob.vector().array()
    v_ob_v = v_ob.vector().array()

    # the magnitude of error :
    D_x_v  = abs(u_v - u_ob_v)
    D_y_v  = abs(v_v - v_ob_v)

    # apply to x-component :
    D_x    = Function(model.Q)
    D_x.vector().set_local(D_x_v)
    D_x.vector().apply('insert')

    # apply to y-componet : 
    D_y    = Function(model.Q)
    D_y.vector().set_local(D_y_v)
    D_y.vector().apply('insert')

    # convert to vector :
    D      = as_vector([D_x, D_y])
    
    # differentiate between objective over cells or facets :
    if integral in [model.OMEGA_GND, model.OMEGA_FLT]:
      D_s    = Function(model.Q2)
      assign(D_s.sub(0), D_x)
      assign(D_s.sub(1), D_y)
    else:
      # set up essential B.C. :
      bc_D   = DirichletBC(model.Q2, D, model.ff, integral)
    
      # apply the difference only over the surface we need :
      D_s    = Function(model.Q2)
      bc_D.apply(D_s.vector())

    # calculate L_inf vector norm :
    D      = MPI.max(mpi_comm_world(), D_s.vector().max())

    s    = "||U - U_ob|| : %.3E" % D
    print_text(s, '208', 1)
    return D
  
  def calc_functionals(self):
    """
    Used to facilitate printing the objective function in adjoint solves.
    """
    s   = "::: calculating functionals :::"
    print_text(s, cls=self)

    ftnls = []

    R = assemble(self.Rp, annotate=False)
    print_min_max(R, 'R', cls=self)
    ftnls.append(R)
    
    J = assemble(self.Jp, annotate=False)
    print_min_max(J, 'J', cls=self)
    ftnls.append(J)

    if self.obj_ftn_type == 'log_L2_hybrid':
      J1 = assemble(self.J1, annotate=False)
      print_min_max(J1, 'J1', cls=self)
      ftnls.append(J1)
      
      J2 = assemble(self.J2, annotate=False)
      print_min_max(J2, 'J2', cls=self)
      ftnls.append(J2)

    if self.reg_ftn_type == 'TV_Tik_hybrid':
      R1 = assemble(self.R1p, annotate=False)
      print_min_max(R1, 'R1', cls=self)
      ftnls.append(R1)
      
      R2 = assemble(self.R2p, annotate=False)
      print_min_max(R2, 'R2', cls=self)
      ftnls.append(R2)
    return ftnls 
    
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

  def optimize_U_ob(self, control, bounds,
                    method            = 'l_bfgs_b',
                    max_iter          = 100,
                    adj_save_vars     = None,
                    adj_callback      = None,
                    post_adj_callback = None):
    """
    """
    s    = "::: solving optimal control to minimize ||u - u_ob|| with " + \
           "control parmeter '%s' :::"
    print_text(s % control.name(), cls=self)

    model = self.model

    # reset entire dolfin-adjoint state :
    adj_reset()

    # starting time :
    t0   = time()

    # need this for the derivative callback :
    global counter
    counter = 0 
    
    # functional lists to be populated :
    global Rs, Js, Ds, J1s, J2s, R1s, R2s
    Rs     = []
    Js     = []
    Ds     = []
    if self.obj_ftn_type == 'log_L2_hybrid':
      J1s  = []
      J2s  = []
    if self.reg_ftn_type == 'TV_Tik_hybrid':
      R1s  = []
      R2s  = []
   
    # solve the momentum equations with annotation enabled :
    s    = '::: solving momentum forward problem :::'
    print_text(s, cls=self)
    self.solve(annotate=True)
    
    # now solve the control optimization problem : 
    s    = "::: starting adjoint-control optimization with method '%s' :::"
    print_text(s % method, cls=self)

    # objective function callback function : 
    def eval_cb(I, c):
      s    = '::: adjoint objective eval post callback function :::'
      print_text(s, cls=self)
      print_min_max(I,    'I',         cls=self)
      print_min_max(c,    'control',   cls=self)
    
    # objective gradient callback function :
    def deriv_cb(I, dI, c):
      global counter, Rs, Js, J1s, J2s
      if method == 'ipopt':
        s0    = '>>> '
        s1    = 'iteration %i (max %i) complete'
        s2    = ' <<<'
        text0 = get_text(s0, 'red', 1)
        text1 = get_text(s1 % (counter, max_iter), 'red')
        text2 = get_text(s2, 'red', 1)
        if MPI.rank(mpi_comm_world())==0:
          print text0 + text1 + text2
        counter += 1
      s    = '::: adjoint obj. gradient post callback function :::'
      print_text(s, cls=self)
      print_min_max(dI,    'dI/dcontrol', cls=self)
      
      # update the DA current velocity to the model for evaluation 
      # purposes only; the model.assign_variable function is 
      # annotated for purposes of linking physics models to the adjoint
      # process :
      u_opt = DolfinAdjointVariable(model.U3).tape_value()
      model.init_U(u_opt, cls=self)

      # print functional values :
      control.assign(c, annotate=False)
      ftnls = self.calc_functionals()
      D     = self.calc_misfit()

      # functional lists to be populated :
      Rs.append(ftnls[0])
      Js.append(ftnls[1])
      Ds.append(D)
      if self.obj_ftn_type == 'log_L2_hybrid':
        J1s.append(ftnls[2])
        J2s.append(ftnls[3])
      if self.reg_ftn_type == 'TV_Tik_hybrid':
        R1s.append(ftnls[4])
        R2s.append(ftnls[5])

      # call that callback, if you want :
      if adj_callback is not None:
        adj_callback(I, dI, c)
   
    # get the cost, regularization, and objective functionals :
    I = self.J + self.R
    
    # define the control parameter :
    m = Control(control, value=control)
    
    # create the reduced functional to minimize :
    F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                          derivative_cb_post=deriv_cb)

    # optimize with scipy's fmin_l_bfgs_b :
    if method == 'l_bfgs_b': 
      out = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=bounds,
                     options={"disp"    : True,
                              "maxiter" : max_iter,
                              "gtol"    : 1e-5})
      b_opt = out[0]
    
    # or optimize with IPOpt (preferred) :
    elif method == 'ipopt':
      try:
        import pyipopt
      except ImportError:
        info_red("""You do not have IPOPT and/or pyipopt installed.
                    When compiling IPOPT, make sure to link against HSL,
                    as it is a necessity for practical problems.""")
        raise
      problem = MinimizationProblem(F, bounds=bounds)
      parameters = {"tol"                : 1e-8,
                    "acceptable_tol"     : 1e-6,
                    "maximum_iterations" : max_iter,
                    "print_level"        : 1,
                    "ma97_order"         : "metis",
                    "linear_solver"      : "ma97"}
      solver = IPOPTSolver(problem, parameters=parameters)
      b_opt  = solver.solve()

    # make the optimal control parameter available :
    model.assign_variable(control, b_opt, cls=self)
    #Control(control).update(b_opt)  # FIXME: does this work?
    
    # call the post-adjoint callback function if set :
    if post_adj_callback is not None:
      s    = '::: calling optimize_u_ob() post-adjoined callback function :::'
      print_text(s, cls=self)
      post_adj_callback()
    
    # save state to unique hdf5 file :
    if isinstance(adj_save_vars, list):
      s    = '::: saving variables in list arg adj_save_vars :::'
      print_text(s, cls=self)
      out_file = model.out_dir + 'u_opt.h5'
      foutput  = HDF5File(mpi_comm_world(), out_file, 'w')
      
      for var in adj_save_vars:
        model.save_hdf5(var, f=foutput)
      
      foutput.close()

    # calculate total time to compute
    tf = time()
    s  = tf - t0
    m  = s / 60.0
    h  = m / 60.0
    s  = s % 60
    m  = m % 60
    text = "time to optimize ||u - u_ob||: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)
    
    # save all the objective functional values with rudimentary plot : 
    d    = model.out_dir + 'objective_ftnls_history/'
    s    = '::: saving objective functionals to %s :::'
    print_text(s % d, cls=self)
    if model.MPI_rank==0:
      if not os.path.exists(d):
        os.makedirs(d)
      np.savetxt(d + 'time.txt', np.array([tf - t0]))
      np.savetxt(d + 'Rs.txt',   np.array(Rs))
      np.savetxt(d + 'Js.txt',   np.array(Js))
      np.savetxt(d + 'Ds.txt',   np.array(Ds))
      if self.obj_ftn_type == 'log_L2_hybrid':
        np.savetxt(d + 'J1s.txt',  np.array(J1s))
        np.savetxt(d + 'J2s.txt',  np.array(J2s))
      if self.reg_ftn_type == 'TV_Tik_hybrid':
        np.savetxt(d + 'R1s.txt',  np.array(R1s))
        np.savetxt(d + 'R2s.txt',  np.array(R2s))

      fig = plt.figure()
      ax  = fig.add_subplot(111)
      #ax.set_yscale('log')
      ax.set_ylabel(r'$\mathscr{J}\left( \mathbf{u} \right)$')
      ax.set_xlabel(r'iteration')
      ax.plot(np.array(Js), 'r-', lw=2.0)
      plt.grid()
      plt.savefig(d + 'J.png', dpi=100)
      plt.close(fig)

      fig = plt.figure()
      ax  = fig.add_subplot(111)
      #ax.set_yscale('log')
      ax.set_ylabel(r'$\mathscr{R}\left( \beta \right)$')
      ax.set_xlabel(r'iteration')
      ax.plot(np.array(Rs), 'r-', lw=2.0)
      plt.grid()
      plt.savefig(d + 'R.png', dpi=100)
      plt.close(fig)

      fig = plt.figure()
      ax  = fig.add_subplot(111)
      #ax.set_yscale('log')
      ax.set_ylabel(r'$\mathscr{D}\left( \mathbf{u} \right)$')
      ax.set_xlabel(r'iteration')
      ax.plot(np.array(Ds), 'r-', lw=2.0)
      plt.grid()
      plt.savefig(d + 'D.png', dpi=100)
      plt.close(fig)
      
      if self.obj_ftn_type == 'log_L2_hybrid':

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        #ax.set_yscale('log')
        ax.set_ylabel(r'$\mathscr{J}_1\left( \mathbf{u} \right)$')
        ax.set_xlabel(r'iteration')
        ax.plot(np.array(J1s), 'r-', lw=2.0)
        plt.grid()
        plt.savefig(d + 'J1.png', dpi=100)
        plt.close(fig)
 
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        #ax.set_yscale('log')
        ax.set_ylabel(r'$\mathscr{J}_2\left( \mathbf{u} \right)$')
        ax.set_xlabel(r'iteration')
        ax.plot(np.array(J2s), 'r-', lw=2.0)
        plt.grid()
        plt.savefig(d + 'J2.png', dpi=100)
        plt.close(fig)
      
      if self.reg_ftn_type == 'TV_Tik_hybrid':

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        #ax.set_yscale('log')
        ax.set_ylabel(r'$\mathscr{R}_{tik}\left( \beta \right)$')
        ax.set_xlabel(r'iteration')
        ax.plot(np.array(R1s), 'r-', lw=2.0)
        plt.grid()
        plt.savefig(d + 'R1.png', dpi=100)
        plt.close(fig)
 
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        #ax.set_yscale('log')
        ax.set_ylabel(r'$\mathscr{R}_{TV}\left( \beta \right)$')
        ax.set_xlabel(r'iteration')
        ax.plot(np.array(R2s), 'r-', lw=2.0)
        plt.grid()
        plt.savefig(d + 'R2.png', dpi=100)
        plt.close(fig)



