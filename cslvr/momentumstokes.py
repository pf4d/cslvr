from fenics               import *
from dolfin_adjoint       import *
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d3model        import D3Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
from copy                 import deepcopy
import sys

#from block import *
#from block.dolfin_util import *
#from block.iterative import *
#from block.algebraic.petsc import *


class MomentumDukowiczStokesReduced(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    #NOTE: experimental
    if type(model) != D3Model:
      s = ">>> MomentumStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    self.linear       = linear

    s = "::: INITIALIZING DUKOWICZ REDUCED FULL-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    # NOTE: not sure why this is ever changed, but the model.assimilate_data
    #       method throws an error if I don't do this :
    parameters["adjoint"]["stop_annotating"] = False

    Q          = model.Q
    S          = model.S
    B          = model.B
    Fb         = model.Fb
    z          = model.x[2]
    W          = model.W
    R          = model.R
    rhoi       = model.rhoi
    rhosw      = model.rhosw
    g          = model.g
    beta       = model.beta
    A          = model.A
    n          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    N          = model.N
    D          = model.D

    dOmega     = model.dOmega()
    dOmega_g   = model.dOmega_g()
    dOmega_w   = model.dOmega_w()
    dGamma     = model.dGamma()
    dGamma_b   = model.dGamma_b()
    dGamma_bg  = model.dGamma_bg()
    dGamma_bw  = model.dGamma_bw()
    dGamma_ld  = model.dGamma_ld()
    dGamma_lto = model.dGamma_lto()
    dGamma_ltu = model.dGamma_ltu()
    dGamma_lt  = model.dGamma_lt()
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
    
    # system unknown function space is created now if periodic boundaries 
    # are not used (see model.generate_function_space()) :
    if model.use_periodic:
      Q2   = model.Q2
    else:
      Q2   = FunctionSpace(mesh, model.QM2e)

    # momenturm and adjoint :
    U      = Function(Q2, name = 'G')
    Lam    = Function(Q2, name = 'Lam')
    dU     = TrialFunction(Q2)
    Phi    = TestFunction(Q2)
    Lam    = Function(Q2)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx = FunctionAssigner(model.u.function_space(), Q2.sub(0))
    self.assy = FunctionAssigner(model.v.function_space(), Q2.sub(1))
    self.assz = FunctionAssigner(model.w.function_space(), Q)
    phi, psi  = Phi
    du,  dv   = dU
    u,   v    = U
    
    # vertical velocity :
    dw     = TrialFunction(Q)
    chi    = TestFunction(Q)
    w      = Function(Q, name='w_f')

    self.w_F = + (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dOmega \
               + (u*N[0] + v*N[1] + dw*N[2] - Fb)*chi*dGamma_b

    #model.calc_normal_vector()
    #n_f        = model.n_f
    #self.w_F   = (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx
    #wD         = (Fb - u*n_f[0] - v*n_f[1])/n_f[2]
    #w_bcs_g    = DirichletBC(Q, wD, model.ff, model.GAMMA_B_GND)
    #w_bcs_f    = DirichletBC(Q, wD, model.ff, model.GAMMA_B_FLT)
    #self.w_bcs = [w_bcs_g, w_bcs_f]
    
    # viscous dissipation :
    U3      = as_vector([u,v,model.w])
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      eta   = self.viscosity(model.U3.copy(True))
      Vd    = 2 * eta * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta   = self.viscosity(U3)
      Vd    = (2*n)/(n+1) * A**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())

    # potential energy :
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # dissipation by sliding :
    w_b    = (Fb - u*N[0] - v*N[1]) / N[2]
    Sl_gnd = - 0.5 * beta * (u**2 + v**2 + w_b**2)

    # pressure boundary :
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * dot(U3, N)

    # action :
    A      = + (Vd - Pe)*dOmega - Sl_gnd*dGamma_bg - Pb*dGamma_bw
    
    if (not model.use_periodic and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      A -= Pb*dGamma_lt
    
    # add lateral boundary conditions :
    # FIXME: need correct BP treatment here
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c     = model.U3.copy(True)
      eta_l    = self.viscosity(U3_c)
      sig_l    = self.stress_tensor(U3_c, model.p, eta_l)
      A -= dot(dot(sig_l, N), U3) * dGamma_ld

    # the first variation of the action in the direction of a
    # test function ; the extremum :
    self.mom_F = derivative(A, U, Phi)

    # the first variation of the extremum in the direction
    # a tril function ; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []
    self.A       = A
    self.U       = U 
    self.w       = w
    self.dU      = dU
    self.Phi     = Phi
    self.Lam     = Lam
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the unknown Function.
    """
    return self.U

  def velocity(self):
    """
    return the velocity.
    """
    return self.model.U3

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
    print_text(s, self.color())

    epi = self.strain_rate_tensor(U)
    tau = 2 * eta * epi
    return tau
  
  def effective_stress(self, U, eta):
    """
    return the effective stress squared.
    """
    tau    = self.deviatoric_stress_tensor(U, eta)
    taudot = 0.5 * tr(dot(tau, tau))
    return taudot

  def effective_strain_rate(self, U):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    epsdot = 0.5 * tr(dot(epi, epi))
    return epsdot

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'cg',
                'preconditioner'           : 'hypre_amg',
                'relative_tolerance'       : 1e-9,
                'relaxation_parameter'     : 0.7,
                'maximum_iterations'       : 50,
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
    m_params  = {'solver'               : nparams,
                 'solve_pressure'       : True,
                 'vert_solve_method'    : 'mumps'}
    return m_params
  
  def solve_vert_velocity(self, annotate=False):
    """ 
    Solve for vertical velocity w.
    """
    s    = "::: solving Dukowicz reduced vertical velocity :::"
    print_text(s, self.color())
    
    model    = self.model
    aw       = assemble(lhs(self.w_F))
    Lw       = assemble(rhs(self.w_F))
    #if self.bc_w != None:
    #  self.bc_w.apply(aw, Lw)
    w_solver = LUSolver(self.solve_params['vert_solve_method'])
    w_solver.solve(aw, self.w.vector(), Lw, annotate=annotate)
    #solve(lhs(self.R2) == rhs(self.R2), self.w, bcs = self.bc_w,
    #      solver_parameters = {"linear_solver" : sm})#,
    #                           "symmetric" : True},
    #                           annotate=False)
    
    self.assz.assign(model.w, self.w, annotate=annotate)
    #w = project(self.w, Q, annotate=annotate)
    print_min_max(self.w, 'w')

  def solve(self, annotate=False):
    """ 
    Perform the Newton solve of the reduced full-Stokes equations 
    """
    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol     = params['solver']['newton_solver']['relative_tolerance']
    maxit    = params['solver']['newton_solver']['maximum_iterations']
    alpha    = params['solver']['newton_solver']['relaxation_parameter']
    lin_slv  = params['solver']['newton_solver']['linear_solver']
    precon   = params['solver']['newton_solver']['preconditioner']
    err_conv = params['solver']['newton_solver']['error_on_nonconvergence']
    s    = "::: solving Dukowicz full-Stokes reduced equations with %i max" + \
             " iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # zero out self.velocity for good convergence for any subsequent solves,
    # e.g. model.L_curve() :
    model.assign_variable(self.get_U(), DOLFIN_EPS)

    def cb_ftn():
      self.solve_vert_velocity(annotate)
    
    # compute solution :
    #solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
    #      annotate = annotate, solver_parameters = params['solver'])
    model.home_rolled_newton_method(self.mom_F, self.U, self.mom_Jac, 
                                    self.mom_bcs, atol=1e-6, rtol=rtol,
                                    relaxation_param=alpha, max_iter=maxit,
                                    method=lin_slv, preconditioner=precon,
                                    cb_ftn=cb_ftn)
    u, v = self.U.split()
  
    self.assx.assign(model.u, u, annotate=False)
    self.assy.assign(model.v, v, annotate=False)

    # solve for pressure if desired :
    if params['solve_pressure']:
      self.solve_pressure(annotate)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0], 'u')
    print_min_max(U3[1], 'v')
    print_min_max(U3[2], 'w')


class MomentumDukowiczStokes(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING DUKOWICZ-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D3Model:
      s = ">>> MomentumDukowiczStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    self.linear       = linear

    S          = model.S
    B          = model.B
    Fb         = model.Fb
    z          = model.x[2]
    W          = model.W
    R          = model.R
    rhoi       = model.rhoi
    rhosw      = model.rhosw
    g          = model.g
    beta       = model.beta
    A          = model.A
    n          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    N          = model.N
    D          = model.D

    dOmega     = model.dOmega()
    dOmega_g   = model.dOmega_g()
    dOmega_w   = model.dOmega_w()
    dGamma     = model.dGamma()
    dGamma_b   = model.dGamma_b()
    dGamma_bg  = model.dGamma_bg()
    dGamma_bw  = model.dGamma_bw()
    dGamma_ld  = model.dGamma_ld()
    dGamma_lto = model.dGamma_lto()
    dGamma_ltu = model.dGamma_ltu()

    #===========================================================================
    # define variational problem :
      
    # system unknown function space is created now if periodic boundaries 
    # are not used (see model.generate_function_space()) :
    if model.use_periodic:
      Q4   = model.Q4
    else:
      Q4   = FunctionSpace(model.mesh, model.QM4e)

    # momenturm and adjoint :
    U      = Function(Q4, name = 'G')
    dU     = TrialFunction(Q4)
    Phi    = TestFunction(Q4)

    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), Q4.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), Q4.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), Q4.sub(2))
    self.assp  = FunctionAssigner(model.p.function_space(), Q4.sub(3))
    phi, psi, xi,  kappa = Phi
    du,  dv,  dw,  dP    = dU
    u,   v,   w,   p     = U

    # create velocity vector :
    U3      = as_vector([u,v,w])

    # viscous dissipation :
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      eta  = self.viscosity(model.U3.copy(True))
      Vd   = 2 * eta * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta  = self.viscosity(U3)
      Vd   = (2*n)/(n+1) * A**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())

    # potential energy :
    Pe     = - rhoi * g * w

    # dissipation by sliding :
    Ut     = U3 - dot(U3,N)*N
    Sl_gnd = - 0.5 * beta * dot(Ut, Ut)

    # incompressibility constraint :
    Pc     = p * div(U3)

    # impenetrability constraint :
    sig    = self.stress_tensor(U3, p, eta)
    lam    = - dot(N, dot(sig, N))
    Nc     = - lam * (dot(U3, N) - Fb)

    # pressure boundary :
    Pb_w   = - rhosw*g*D * dot(U3, N)
    Pb_l   = - rhoi*g*(S - z) * dot(U3, N)

    # action :
    A      = + (Vd - Pe - Pc)*dOmega - Nc*dGamma_b \
             - Nc*dGamma_b - Sl_gnd*dGamma_bg - Pb_w*dGamma_bw

    if (not model.use_periodic and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      A -= Pb_w*dGamma_ltu

    if (not model.use_periodic and not use_lat_bcs):
      s = "    - using internal divide lateral pressure boundary condition -"
      print_text(s, self.color())
      A -= Pb_l*dGamma_ld

    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c     = model.U3.copy(True)
      eta_l    = self.viscosity(U3_c)
      sig_l    = self.stress_tensor(U3_c, model.p, eta_l)
      A       -= dot(dot(sig_l, N), U3) * dGamma_ld

    # the first variation of the action integral A w.r.t. U in the 
    # direction of a test function Phi; the extremum :
    self.mom_F = derivative(A, U, Phi)

    # the first variation of the extremum in the direction
    # a trial function dU; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)

    self.mom_bcs = []
    self.A       = A
    self.U       = U 
    self.dU      = dU
    self.Phi     = Phi
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the unknown Function.
    """
    return self.U

  def velocity(self):
    """
    return the velocity.
    """
    return self.model.U3

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
    taudot = 0.5 * tr(dot(tau, tau))
    return taudot

  def effective_strain_rate(self, U):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    epsdot = 0.5 * inner(epi, epi)#tr(dot(epi, epi))
    return epsdot

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'mumps',
                'relative_tolerance'       : 1e-9,
                'relaxation_parameter'     : 1.0,
                'maximum_iterations'       : 25,
                'error_on_nonconvergence'  : False,
              }}
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
    s    = "::: solving Dukowicz-full-Stokes equations" + \
              " with %i max iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # zero out self.velocity for good convergence for any subsequent solves,
    # e.g. model.L_curve() :
    model.assign_variable(self.get_U(), DOLFIN_EPS)
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v, w, p = self.U.split()
    
    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)
    self.assz.assign(model.w, w, annotate=annotate)
    self.assp.assign(model.p, p, annotate=annotate)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0],   'u')
    print_min_max(U3[1],   'v')
    print_min_max(U3[2],   'w')
    print_min_max(model.p, 'p')


class MomentumNitscheStokes(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None,
                 linear=False, use_lat_bcs=False,
                 use_pressure_bc=True, stabilized=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING NITSCHE-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D3Model:
      s = ">>> MomentumNitscheStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    # save the solver parameters :
    self.solve_params = solve_params
    self.linear       = linear
    self.stabilized   = stabilized

    S          = model.S
    B          = model.B
    Fb         = model.Fb
    z          = model.x[2]
    W          = model.W
    R          = model.R
    rhoi       = model.rhoi
    rhosw      = model.rhosw
    g          = model.g
    beta       = model.beta
    lam        = model.lam
    A          = model.A
    N          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    n          = model.N
    D          = model.D

    dOmega     = model.dOmega()
    dOmega_g   = model.dOmega_g()
    dOmega_w   = model.dOmega_w()
    dGamma     = model.dGamma()
    dGamma_b   = model.dGamma_b()
    dGamma_bg  = model.dGamma_bg()
    dGamma_bw  = model.dGamma_bw()
    dGamma_ld  = model.dGamma_ld()
    dGamma_lto = model.dGamma_lto()
    dGamma_ltu = model.dGamma_ltu()
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    if stabilized:
      s  = "    - using stabilized elements -"
      # system unknown function space is created now if periodic boundaries 
      # are not used (see model.generate_function_space()) :
      if model.use_periodic:
        Q4   = model.Q4
      else:
        Q4   = FunctionSpace(model.mesh, model.QM4e)
      self.assx  = FunctionAssigner(model.u.function_space(), Q4.sub(0))
      self.assy  = FunctionAssigner(model.v.function_space(), Q4.sub(1))
      self.assz  = FunctionAssigner(model.w.function_space(), Q4.sub(2))
      self.assp  = FunctionAssigner(model.p.function_space(), Q4.sub(3))
    else:
      s  = "    - using Taylor-Hood elements -"
      # system unknown function space is created now if periodic boundaries 
      # are not used (see model.generate_function_space()) :
      if model.use_periodic:
        Q4   = model.QTH3
      else:
        Q4   = FunctionSpace(model.mesh, model.QTH3e)
      self.assx  = FunctionAssigner(model.u.function_space(),
                                    model.Q_non_periodic)
      self.assy  = FunctionAssigner(model.v.function_space(),
                                    model.Q_non_periodic)
      self.assz  = FunctionAssigner(model.w.function_space(),
                                    model.Q_non_periodic)
      self.assp  = FunctionAssigner(model.p.function_space(),
                                    model.Q_non_periodic)
    print_text(s, self.color())

    # momenturm and adjoint :
    U      = Function(Q4, name = 'G')
    Phi    = TestFunction(Q4)
    dU     = TrialFunction(Q4)

    phi_x, phi_y, phi_z, q     = Phi
    psi_x, psi_y, psi_z, psi_p = dU
    U_x,   U_y,   U_z,   p     = U
    
    # create velocity vector :
    u       = as_vector([U_x,   U_y,   U_z])
    v       = as_vector([phi_x, phi_y, phi_z])
    dpsi    = as_vector([psi_x, psi_y, psi_z])
    
    # viscosity :
    if linear:
      s  = "    - using linear form of momentum using model.U3 -"
      eta  = self.viscosity(model.U3.copy(True))
    else:
      s  = "    - using nonlinear form of momentum -"
      eta  = self.viscosity(u)
    print_text(s, self.color())

    alpha = Constant(1e-12)
    gamma = Constant(1e2)
    f     = Constant((0.0, 0.0, -rhoi * g))
    I     = Identity(3)
    u_n   = Fb

    # pressure boundary :
    ut    = u - dot(u,n)*n
    Pb_w  = - rhosw*g*D * n
    Pb_l  = - rhoi*g*(S - z) * n

    def epsilon(u): return 0.5*(grad(u) + grad(u).T)
    def sigma(u,p): return 2*eta * epsilon(u) - p*I
    def L(u,p):     return -div(sigma(u,p))
    
    t   = dot(sigma(u,p), n)
    s   = dot(sigma(v,q), n)

    B_o = + inner(sigma(u,p),grad(v))*dOmega \
          - div(u)*q*dOmega
    
    B_g = - dot(n,t) * dot(v,n) * dGamma_b \
          - (dot(u,n) - u_n) * dot(s,n) * dGamma_b \
          + gamma/h * dot(u,n) * dot(v,n) * dGamma_b \
          + beta * dot(ut, v) * dGamma_b
    
    F   = + dot(f,v) * dx \
          + gamma/h * u_n * dot(v,n) * dGamma_b

    # stabilized form is identical to TH with the addition the following terms :
    if stabilized:
      B_o += alpha * h**2 * inner(L(u,p), L(v,q)) * dOmega
      F   += alpha * h**2 * inner(f, L(v,q)) * dOmega
   
    self.mom_F = B_o + B_g - F
    
    if (not model.use_periodic and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      self.mom_F -= Pb_w*dGamma_ltu
    
    if (not model.use_periodic and not use_lat_bcs):
      s = "    - using internal divide lateral pressure boundary condition -"
      print_text(s, self.color())
      self.mom_F -= Pb_l*dGamma_ld
    
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c    = model.U3.copy(True)
      eta     = self.viscosity(U3_c)
      sig_l   = self.stress_tensor(U3_c, model.p, eta_l)
      A      += dot(dot(sig_l, n), u) * dGamma_ld

    # the first variation of the extremum in the direction
    # a trial function dU; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)

    # Form for use in constructing preconditioner matrix
    self.bp_R = inner(grad(u), grad(v))*dOmega + p*q*dOmega
    self.bp_Jac = derivative(self.bp_R, U, dU)

    self.mom_bcs = []
    self.U       = U 
    self.dU      = dU
    self.Phi     = Phi
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.mom_F

  def get_U(self):
    """
    Return the unknown Function.
    """
    return self.U

  def velocity(self):
    """
    return the velocity.
    """
    return self.model.U3

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
    taudot = 0.5 * tr(dot(tau, tau))
    return taudot

  def effective_strain_rate(self, U):
    """
    return the effective strain rate squared.
    """
    epi    = self.strain_rate_tensor(U)
    epsdot = 0.5 * tr(dot(epi, epi))
    return epsdot

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'mumps',
                'relative_tolerance'       : 1e-9,
                'relaxation_parameter'     : 1.0,
                'maximum_iterations'       : 12,
                'error_on_nonconvergence'  : False,
              }}
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
    s    = "::: solving Nitsche-full-Stokes equations" + \
              " with %i max iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # zero out self.velocity for good convergence for any subsequent solves,
    # e.g. model.L_curve() :
    model.assign_variable(self.get_U(), DOLFIN_EPS)
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    #params['solver']['newton_solver']['linear_solver'] = 'gmres'
    #precond = 'fieldsplit'
    #model.home_rolled_newton_method(self.mom_F, self.U, self.mom_Jac, 
    #                                self.mom_bcs, atol=1e-6, rtol=rtol,
    #                                relaxation_param=alpha, max_iter=maxit,
    #                                method=params['solver']['newton_solver']['linear_solver'], preconditioner=precond,
    #                                bp_Jac=self.bp_Jac,
    #                                bp_R=self.bp_R)
      
    if self.stabilized:
      u, v, w, p = self.U.split()
      self.assx.assign(model.u, u, annotate=annotate)
      self.assy.assign(model.v, v, annotate=annotate)
      self.assz.assign(model.w, w, annotate=annotate)
      self.assp.assign(model.p, p, annotate=annotate)
    
    else:
      u, v, w, p = split(self.U)
      u_n = project(u, model.Q_non_periodic, annotate=annotate)
      v_n = project(v, model.Q_non_periodic, annotate=annotate)
      w_n = project(w, model.Q_non_periodic, annotate=annotate)
      p_n = project(p, model.Q_non_periodic, annotate=annotate)
      
      self.assx.assign(model.u, u_n, annotate=annotate)
      self.assy.assign(model.v, v_n, annotate=annotate)
      self.assz.assign(model.w, w_n, annotate=annotate)
      self.assp.assign(model.p, p_n, annotate=annotate)

    print_min_max(model.U3, 'U3')
    
    #U3 = model.U3.split(True)

    #print_min_max(U3[0],   'u')
    #print_min_max(U3[1],   'v')
    #print_min_max(U3[2],   'w')
    #print_min_max(model.p, 'p')



