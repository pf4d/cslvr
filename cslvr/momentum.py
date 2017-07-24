from fenics                 import *
from dolfin_adjoint         import *
from cslvr.inputoutput      import get_text, print_text, print_min_max
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
               linear=False, use_lat_bcs=False,
               use_pressure_bc=True, **kwargs):
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
    self.kwargs            = kwargs
    
    self.initialize(model, solve_params, linear,
                    use_lat_bcs, use_pressure_bc, **kwargs)
  
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False,
                 use_pressure_bc=True, **kwargs):
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
                    use_pressure_bc=self.use_pressure_bc_s,
                    **self.kwargs)

  def linearize_viscosity(self, reset_orig_config=True):
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

    # this is useful so that when you call reset(), the viscosity stays
    # linear :
    if reset_orig_config:
      s = "::: reseting the original config to use linear viscosity :::"
      print_text(s, self.color())
      self.linear_s       = True
      self.solve_params_s = mom_params

    self.initialize(self.model, solve_params=mom_params,
                    linear=True,
                    use_lat_bcs=self.use_lat_bcs_s, 
                    use_pressure_bc=self.use_pressure_bc_s,
                    **self.kwargs)
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'cyan'

  def get_residual(self):
    """
    Returns the momentum residual.
    """
    #raiseNotDefined()
    return self.mom_F

  def get_U(self):
    """
    Return the velocity Function.
    """
    #raiseNotDefined()
    return self.U

  def get_dU(self):
    """
    Return the trial function for U.
    """
    #raiseNotDefined()
    return self.dU

  def get_Phi(self):
    """
    Return the test function for U.
    """
    #raiseNotDefined()
    return self.Phi

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
    r"""
    calculates the viscosity saved to ``self.eta_shf`` and ``self.eta_gnd``, for
    floating and grounded ice, respectively.  Uses velocity vector ``U`` with
    components ``u``,``v``,``w``.  
    If ``linear == True``, form viscosity from ``model.U3``.
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
    r"""
    Solve for the friction heat term stored in ``model.q_fric``.
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

    n     = project(grad(model.B))
    n_x_v = n[0].vector().array()
    n_y_v = n[1].vector().array()
    n_z_v = n[2].vector().array()
    UdotN = u_v*n_x_v + v_v*n_y_v + w_v*n_z_v
    ut  = u_v - UdotN*n_x
    vt  = v_v - UdotN*n_y
    wt  = w_v - UdotN*n_z
    
    q_fric_v = beta_v * (ut**2 + vt**2 + wt**2)

    model.init_q_fric(q_fric_v)
    
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
    return self.J + replace(R, {Phi : dU})

  def dLdc(self, L, c): 
    """
    Returns the derivative of the Lagrangian consisting of adjoint-computed
    self.Lam values w.r.t. the control variable ``c``, i.e., 

       dL    d [             ]
       -- = -- [ L(self.Lam) ]
       dc   dc [             ]

    """
    s  = "::: forming dLdc :::"
    print_text(s, self.color())
    
    dU  = self.get_dU()
    Lam = self.get_Lam()

    # we need to evaluate the Lagrangian with the values of Lam computed from
    # self.dI in order to get the derivative of the Lagrangian w.r.t. the 
    # control variables.  Hence we need a new Lagrangian with the trial 
    # functions replaced with the computed Lam values.
    L_lam  = replace(L, {dU : Lam})

    # the Lagrangian with unknowns replaced with computed Lam :
    H_lam  = self.J + L_lam

    # the derivative of the Hamiltonian w.r.t. the control variables in the 
    # direction of a test function :
    return derivative(H_lam, c, TestFunction(self.model.Q))
    
  def solve_adjoint_momentum(self, H):
    """
    Solves for the adjoint variables self.Lam from the Hamiltonian <H>.
    """
    U   = self.get_U()
    Phi = self.get_Phi()
    Lam = self.get_Lam()

    # we desire the derivative of the Lagrangian w.r.t. the model state U
    # in the direction of the test function Phi to vanish :
    dI = derivative(H, U, Phi)
    
    s  = "::: solving adjoint momentum :::"
    print_text(s, self.color())
    
    aw = assemble(lhs(dI))
    Lw = assemble(rhs(dI))
    
    a_solver = KrylovSolver('cg', 'hypre_amg')
    a_solver.solve(aw, Lam.vector(), Lw, annotate=False)

    print_min_max(Lam, 'Lam')



