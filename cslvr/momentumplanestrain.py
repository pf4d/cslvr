from dolfin               import *
from dolfin_adjoint       import *
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d2model        import D2Model
from cslvr.latmodel       import LatModel
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
import sys


class MomentumDukowiczPlaneStrain(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    """
    s = "::: INITIALIZING DUKOWICZ-PLANE-STRAIN PHYSICS :::"
    print_text(s, cls=self)
    
    if type(model) != LatModel:
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
    self.assx  = FunctionAssigner(model.u.function_space(), model.Q3.sub(0))
    self.assz  = FunctionAssigner(model.v.function_space(), model.Q3.sub(1))
    self.assp  = FunctionAssigner(model.p.function_space(), model.Q3.sub(2))

    mesh       = model.mesh
    S          = model.S
    B          = model.B
    H          = S - B
    z          = model.x[1]
    W          = model.W
    R          = model.R
    rhoi       = model.rhoi
    rhosw      = model.rhosw
    g          = model.g
    beta       = model.beta
    h          = model.h
    N          = model.N
    D          = model.D
    A          = model.A
    n          = model.n
    eps_reg    = model.eps_reg
    Fb         = model.Fb

    dOmega     = model.dOmega()
    dGamma_b   = model.dGamma_b()
    dGamma_bw  = model.dGamma_bw()
    dGamma_ltu = model.dGamma_ltu()
    dGamma_ld  = model.dGamma_ld()
     
    #===========================================================================
    # define variational problem :
    phi, xsi, kappa = Phi
    du,  dw,  dP    = dU
    u,   w,   p     = U

    # create velocity vector :
    U2     = as_vector([u,w])
    
    # viscous dissipation :
    epsdot  = self.effective_strain_rate(U2)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      U3_c    = model.U3.copy(True)
      U3_2    = as_vector([U3_c[0], U3_c[1]])
      eta     = self.viscosity(U3_2)
      Vd      = 2 * eta * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta      = self.viscosity(U2)
      Vd       = (2*n)/(n+1) * A**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, cls=self)

    # potential energy :
    Pe   = - rhoi * g * w

    # dissipation by sliding :
    Ut   = U2 - dot(U2,N)*N
    Sl   = - 0.5 * beta * dot(Ut, Ut)

    # incompressibility constraint :
    Pc   = p * div(U2) 
    
    # inpenetrability constraint :
    sig  = self.stress_tensor(U2, p, eta)
    lam  = p#-dot(N, dot(sig, N))
    Nc   = -lam * dot(U2,N)

    # pressure boundary :
    Pb_w   = - rhosw*g*D * dot(U2,N)
    Pb_l   = - rhoi*g*(S - z) * dot(U2,N)

    # action : 
    A      = (Vd - Pe - Pc)*dOmega - Nc*dGamma_b - Sl*dGamma_b - Pb_w*dGamma_bw
    
    if (not model.use_periodic and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, cls=self)
      A -= Pb_w*dGamma_ltu
    
    if (not model.use_periodic and not use_lat_bcs):
      s = "    - using internal divide lateral pressure boundary condition -"
      print_text(s, cls=self)
      A -= Pb_l*dGamma_ld
    
    # add lateral boundary conditions :  
    elif use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, cls=self)
      U3_c   = model.U3.copy(True)
      U3_2   = as_vector([U3_c[0], U3_c[1]])
      eta_l  = self.viscosity(U3_2)
      sig_l  = self.stress_tensor(U3_2, model.p, eta_l)
      A     -= dot(dot(sig_l, N), U2) * dGamma_ld

    # the first variation of the action in the direction of a
    # test function ; the extremum :
    self.mom_F = derivative(A, U, Phi)

    # the first variation of the extremum in the direction
    # a tril function ; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []
    
    self.A       = A
    self.U       = U 
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.A

  def get_U(self):
    """
    Return the unknown Function.
    """
    return self.U

  def velocity(self):
    """
    return the velocity.
    """
    return as_vector([self.model.U3[0], self.model.U3[1]])

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
    print_text(s, cls=self)

    I     = Identity(2)
    tau   = self.deviatoric_stress_tensor(U, eta)

    sigma = tau - p*I
    return sigma
    
  def deviatoric_stress_tensor(self, U, eta):
    """
    return the deviatoric stress tensor.
    """
    s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
    print_text(s, cls=self)

    epi = self.strain_rate_tensor(U)
    tau = 2 * eta * epi
    return tau
  
  def effective_stress(self, U, eta):
    """
    return the effective stress squared.
    """
    tau    = self.deviatoric_stress_tensor(U, eta)
    tu_xx  = tau[0,0]
    tu_zz  = tau[1,1]
    tu_xz  = tau[0,1]
    
    # Second invariant of the strain rate tensor squared
    taudot = 0.5 * (tu_xx**2 + tu_zz**2) + tu_xz**2
    return taudot

  def effective_strain_rate(self, U):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    ep_xx  = epi[0,0]
    ep_zz  = epi[1,1]
    ep_xz  = epi[0,1]
    
    # Second invariant of the strain rate tensor squared
    epsdot = 0.5 * (ep_xx**2 + ep_zz**2) + ep_xz**2
    return epsdot

  def calc_q_fric(self):
    """
    Solve for the friction heat term stored in model.q_fric.
    """ 
    # calculate melt-rate : 
    s = "::: solving basal friction heat :::"
    print_text(s, cls=self)
    
    model     = self.model
    u,v,w     = model.U3.split(True)
              
    beta_v    = model.beta.vector().array()
    u_v       = u.vector().array()
    w_v       = w.vector().array()
    Fb_v      = model.Fb.vector().array()

    q_fric_v  = beta_v * (u_v**2 + (w_v+Fb_v)**2)
    
    model.init_q_fric(q_fric_v)

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
                                  'relative_tolerance'       : 1e-5,
                                  'relaxation_parameter'     : 0.7,
                                  'maximum_iterations'       : 25,
                                  'error_on_nonconvergence'  : False}}
    m_params  = {'solver'      : nparams}
    return m_params

  def solve(self, annotate=False):
    """ 
    Perform the Newton solve of the full-Stokes equations 
    """
    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s    = "::: solving plane-strain momentum equations" + \
           " with %i max iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), cls=self)
    
    # zero out self.velocity for good convergence for any subsequent solves,
    # e.g. model.L_curve() :
    model.assign_variable(self.get_U(), DOLFIN_EPS)
    
    # compute solution :
    solve(self.mom_F == 0, self.get_U(), J=self.mom_Jac, bcs=self.mom_bcs,
          annotate=annotate, solver_parameters=params['solver'])

    # update the model's momentum container :    
    self.update_model_var(self.get_U(), annotate=annotate)

  def update_model_var(self, u, annotate=False):
    """
    Update the two horizontal components of velocity in ``self.model.U3``
    to those given by ``u``.
    """
    u_x, u_y, p = u.split()
    self.assx.assign(self.model.u, u_x, annotate=False)
    self.assz.assign(self.model.v, u_y, annotate=False)
    self.assp.assign(self.model.p, p,   annotate=False)

    print_min_max(self.model.U3, 'model.U3', cls=self)
    print_min_max(self.model.p,  'model.p',  cls=self)



