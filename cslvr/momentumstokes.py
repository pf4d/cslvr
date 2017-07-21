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
    A_shf      = model.A_shf
    A_gnd      = model.A_gnd
    n          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    N          = model.N
    D          = model.D

    dx_f       = model.dx_f
    dx_g       = model.dx_g
    dx         = model.dx
    dBed_g     = model.dBed_g
    dBed_f     = model.dBed_f
    dLat_t     = model.dLat_t
    dLat_d     = model.dLat_d
    dBed       = model.dBed
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :

    # momenturm and adjoint :
    U      = Function(model.Q2, name = 'G')
    Lam    = Function(model.Q2, name = 'Lam')
    dU     = TrialFunction(model.Q2)
    Phi    = TestFunction(model.Q2)
    Lam    = Function(model.Q2)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx = FunctionAssigner(model.u.function_space(), model.Q2.sub(0))
    self.assy = FunctionAssigner(model.v.function_space(), model.Q2.sub(1))
    self.assz = FunctionAssigner(model.w.function_space(), model.Q)
    phi, psi  = Phi
    du,  dv   = dU
    u,   v    = U
    
    # vertical velocity :
    dw     = TrialFunction(model.Q)
    chi    = TestFunction(model.Q)
    w      = Function(model.Q, name='w_f')

    self.w_F = + (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx \
               + (u*N[0] + v*N[1] + dw*N[2] - Fb)*chi*dBed

    #model.calc_normal_vector()
    #n_f        = model.n_f
    #self.w_F   = (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx
    #wD         = (Fb - u*n_f[0] - v*n_f[1])/n_f[2]
    #w_bcs_g    = DirichletBC(model.Q, wD, model.ff, model.GAMMA_B_GND)
    #w_bcs_f    = DirichletBC(model.Q, wD, model.ff, model.GAMMA_B_FLT)
    #self.w_bcs = [w_bcs_g, w_bcs_f]
    
    # viscous dissipation :
    U3      = as_vector([u,v,model.w])
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      eta_shf, eta_gnd = self.viscosity(model.U3.copy(True))
      Vd_shf           = 2 * eta_shf * epsdot
      Vd_gnd           = 2 * eta_gnd * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta_shf, eta_gnd = self.viscosity(U3)
      Vd_shf   = (2*n)/(n+1) * A_shf**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd   = (2*n)/(n+1) * A_gnd**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())

    # potential energy :
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # dissipation by sliding :
    w_b    = (Fb - u*N[0] - v*N[1]) / N[2]
    Sl_gnd = - 0.5 * beta * (u**2 + v**2 + w_b**2)

    # pressure boundary :
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * dot(U3, N)

    # action :
    A      = + Vd_shf*dx_f + Vd_gnd*dx_g - Pe*dx \
             - Sl_gnd*dBed_g - Pb*dBed_f
    
    if (not model.use_periodic_boundaries and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      A -= Pb*dLat_t
    
    # add lateral boundary conditions :
    # FIXME: need correct BP treatment here
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c                 = model.U3.copy(True)
      eta_shf_l, eta_gnd_l = self.viscosity(U3_c)
      sig_g_l    = self.stress_tensor(U3_c, model.p, eta_gnd_l)
      #sig_g_l    = self.stress_tensor(U2, p, eta_gnd)
      A -= dot(dot(sig_g_l, N), U3) * dLat_d

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
                'relative_tolerance'       : 1e-5,
                'relaxation_parameter'     : 0.7,
                'maximum_iterations'       : 35,
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
  
  #def solve_pressure(self, annotate=False):
  #  """
  #  Solve for pressure model.p.
  #  """
  #  s    = "::: solving Dukowicz reduced pressure :::"
  #  print_text(s, self.color())
  #  
  #  model = self.model
  #  #Q     = FunctionSpace(model.mesh, 'CG', 2)
  #  Q     = model.Q
  #  p     = TrialFunction(Q)
  #  phi   = TestFunction(Q)
  # 
  #  U     = model.U3
  #  #u3,v3,w3 = model.U3.split(True)
  #  #u     = interpolate(u3, Q)
  #  #v     = interpolate(v3, Q)
  #  #w     = interpolate(w3, Q)
  #  #U     = as_vector([u,v,w])

  #  #b_shf   = self.b_shf
  #  #b_gnd   = self.b_gnd
  #  #eps_reg = model.eps_reg
  #  #n       = model.n
  #  #  
  #  #epsdot  = self.effective_strain_rate(U)
  #  #eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
  #  #eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
  #  eta_shf  = self.eta_shf
  #  eta_gnd  = self.eta_gnd

  #  epi     = self.strain_rate_tensor(U)
  #  ep_zx   = epi[2,0]
  #  ep_zy   = epi[2,1]
  #  ep_zz   = epi[2,2]
  #  rho     = model.rhoi
  #  g       = model.g
  #  dx      = model.dx
  #  dx_g    = model.dx_g
  #  dx_f    = model.dx_f

  #  a     = p.dx(2) * phi * dx
  #  L     = + rho * g * phi * dx \
  #          - (2*eta_shf*ep_zx) * phi.dx(0) * dx_f \
  #          - (2*eta_shf*ep_zy) * phi.dx(1) * dx_f \
  #          - (2*eta_shf*ep_zz) * phi.dx(2) * dx_f \
  #          - (2*eta_gnd*ep_zx) * phi.dx(0) * dx_g \
  #          - (2*eta_gnd*ep_zy) * phi.dx(1) * dx_g \
  #          - (2*eta_gnd*ep_zz) * phi.dx(2) * dx_g \
  # 
  #  p = Function(Q) 
  #  solve(a == L, p, annotate=annotate)
  #  print_min_max(p, 'p')
  #  model.save_xdmf(p, 'p')

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
    #w = project(self.w, model.Q, annotate=annotate)
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
    A_shf      = model.A_shf
    A_gnd      = model.A_gnd
    n          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    N          = model.N
    D          = model.D

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
     
    #===========================================================================
    # define variational problem :

    # momenturm and adjoint :
    U      = Function(model.Q4, name = 'G')
    Lam    = Function(model.Q4, name = 'Lam')
    dU     = TrialFunction(model.Q4)
    Phi    = TestFunction(model.Q4)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), model.Q4.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), model.Q4.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), model.Q4.sub(2))
    self.assp  = FunctionAssigner(model.p.function_space(), model.Q4.sub(3))
    phi, psi, xi,  kappa = Phi
    du,  dv,  dw,  dP    = dU
    u,   v,   w,   p     = U
    
    # create velocity vector :
    U3      = as_vector([u,v,w])
    
    # viscous dissipation :
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      eta_shf, eta_gnd = self.viscosity(model.U3.copy(True))
      Vd_shf   = 2 * eta_shf * epsdot
      Vd_gnd   = 2 * eta_gnd * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta_shf, eta_gnd = self.viscosity(U3)
      Vd_shf   = (4*n)/(n+1) * A_shf**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd   = (4*n)/(n+1) * A_gnd**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())
   
    # potential energy :
    Pe     = - rhoi * g * w

    # dissipation by sliding :
    Ut     = U3 - dot(U3,N)*N
    Sl_gnd = - 0.5 * beta * dot(Ut, Ut)

    # incompressibility constraint :
    Pc     = p * div(U3)

    # impenetrability constraint :
    sig_f  = self.stress_tensor(U3, p, eta_shf)
    sig_g  = self.stress_tensor(U3, p, eta_gnd)
    lam_f  = p#- dot(N, dot(sig_f, N))
    lam_g  = p#- dot(N, dot(sig_g, N))
    Nc_g   = - lam_g * (dot(U3, N) - Fb)
    Nc_f   = - lam_f * (dot(U3, N) - Fb)

    # pressure boundary :
    Pb_w   = - rhosw*g*D * dot(U3, N)
    Pb_l   = - rhoi*g*(S - z) * dot(U3, N)

    # action :
    A      = + Vd_shf*dx_f + Vd_gnd*dx_g - (Pe + Pc)*dx \
             - (Nc_g + Sl_gnd)*dBed_g - (Nc_f + Pb_w)*dBed_f
    
    if (not model.use_periodic_boundaries and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      A -= Pb_w*dLat_t
    
    if (not model.use_periodic_boundaries and not use_lat_bcs):
      s = "    - using internal divide lateral pressure boundary condition -"
      print_text(s, self.color())
      A -= Pb_l*dLat_d
    
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c                 = model.U3.copy(True)
      eta_shf_l, eta_gnd_l = self.viscosity(U3_c)
      sig_g_l    = self.stress_tensor(U3_c, model.p, eta_gnd_l)
      #sig_g_l    = self.stress_tensor(U2, p, eta_gnd)
      A -= dot(dot(sig_g_l, N), U3) * dLat_d

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
                'relative_tolerance'       : 1e-5,
                'relaxation_parameter'     : 0.7,
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
  


class MomentumDukowiczStokesOpt(Momentum):
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
    lam        = model.lam
    A_shf      = model.A_shf
    A_gnd      = model.A_gnd
    n          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    N          = model.N
    D          = model.D
    p          = model.p

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
    U      = Function(model.Q3, name = 'G')
    Lam    = Function(model.Q3, name = 'Lam')
    dU     = TrialFunction(model.Q3)
    Phi    = TestFunction(model.Q3)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(), model.Q3.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(), model.Q3.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(), model.Q3.sub(2))
    
    # momenturm and adjoint :
    phi, psi, xi  = Phi
    du,  dv,  dw  = dU
    u,   v,   w   = U
    
    # create velocity vector :
    U3      = U
    dU3     = as_vector([du,dv,dw])
    dphi    = as_vector([phi,psi,xi])
    
    # viscous dissipation :
    epsdot  = self.effective_strain_rate(U3)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      eta_shf, eta_gnd = self.viscosity(model.U3.copy(True))
      Vd_shf           = 2 * eta_shf * epsdot
      Vd_gnd           = 2 * eta_gnd * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta_shf, eta_gnd = self.viscosity(U3)
      Vd_shf   = (4*n)/(n+1) * A_shf**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd   = (4*n)/(n+1) * A_gnd**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())
   
    # potential energy :
    Pe     = - rhoi * g * w

    # dissipation by sliding :
    Ut     = U3# - dot(U3,N)*N
    Sl_gnd = - 0.5 * beta * dot(Ut, Ut)

    # incompressibility constraint :
    Pc     = p * div(U3)

    # impenetrability constraint :
    #sig_f  = self.stress_tensor(U3, p, eta_shf)
    #sig_g  = self.stress_tensor(U3, p, eta_gnd)
    #lam_f  = lam#- dot(N, dot(sig_f, N))
    #lam_g  = lam#- dot(N, dot(sig_g, N))
    #Nc_g   = - lam_g * (dot(U3, N) - Fb)
    #Nc_f   = - lam_f * (dot(U3, N) - Fb)
    Nc     = - lam * dot(U3, N)

    # pressure boundary :
    Pb_w   = - rhosw*g*D * dot(U3, N)
    Pb_l   = - rhoi*g*(S - z) * dot(U3, N)

    # action :
    A      = + Vd_shf*dx_f + Vd_gnd*dx_g - (Pe + Pc)*dx \
             - Sl_gnd*dBed_g - Pb_w*dBed_f - Nc*dBed
    
    #A_lam  = + (p * dot(dphi, N) + (dot(U3, N) - Fb) * kappa) * dBed
    A_lam  = lam * dot(dphi, N) * dBed

    self.J = + 1e8 * (0.5 * (dot(U3, N) - Fb)**2 + abs(dot(U3, N))) * dBed \
             + 0.5 * div(U3)**2 * dx
    
    if (not model.use_periodic_boundaries and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      A -= Pb_w*dLat_t
    
    if (not model.use_periodic_boundaries and not use_lat_bcs):
      s = "    - using internal divide lateral pressure boundary condition -"
      print_text(s, self.color())
      A -= Pb_l*dLat_d
    
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c                 = model.U3.copy(True)
      eta_shf_l, eta_gnd_l = self.viscosity(U3_c)
      sig_g_l    = self.stress_tensor(U3_c, model.p, eta_gnd_l)
      #sig_g_l    = self.stress_tensor(U2, p, eta_gnd)
      A -= dot(dot(sig_g_l, N), U3) * dLat_d

    # the first variation of the action integral A w.r.t. U in the 
    # direction of a test function Phi; the extremum :
    self.mom_F = derivative(A, U, Phi)# + A_lam

    # the first variation of the extremum in the direction
    # a trial function dU; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)
  
    # make a form for the preconditioner :
    #mom_bp = MomentumDukowiczBP(model, solve_params=solve_params,
    #                            linear=linear, use_lat_bcs=use_lat_bcs,
    #                            use_pressure_bc=use_pressure_bc)
    #self.bp_R   = replace(mom_bp.get_residual(),
    #                      {mom_bp.get_U()   : as_vector([u,v]),
    #                       mom_bp.get_Phi() : as_vector([phi,psi])})
    #self.bp_Jac = derivative(self.bp_R, as_vector([u,v]), as_vector([du,dv]))

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
                'relative_tolerance'       : 1e-5,
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
    #params['solver']['newton_solver']['linear_solver'] = 'cg'
    #precond = 'none'
    #model.home_rolled_newton_method(self.mom_F, self.U, self.mom_Jac, 
    #                                self.mom_bcs, atol=1e-6, rtol=rtol,
    #                                relaxation_param=alpha, max_iter=maxit,
    #                                method=params['solver']['newton_solver']['linear_solver'], preconditioner=precond)
    #                                #cb_ftn=cb_ftn, bp_Jac=self.bp_Jac,
    #                                #bp_R=self.bp_R)
    #u, v, w, p = self.U.split()
    u, v, w = self.U.split()
    
    self.assx.assign(model.u, u, annotate=annotate)
    self.assy.assign(model.v, v, annotate=annotate)
    self.assz.assign(model.w, w, annotate=annotate)
    #self.assp.assign(model.p, p, annotate=annotate)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0],   'u')
    print_min_max(U3[1],   'v')
    print_min_max(U3[2],   'w')
    print_min_max(model.p, 'p')
  
  def opt(self, method            = 'l_bfgs_b',
                bounds            = None,#(0,1e16),
                max_iter          = 100,
                adj_save_vars     = None,
                adj_callback      = None,
                post_adj_callback = None):
    """
    """
    s    = "::: solving optimal control to minimize ||u.n - Fb|| with " + \
           "control parmeter '%s' :::"
    print_text(s % self.model.lam.name(), cls=self)

    model   = self.model
    control = model.lam

    # reset entire dolfin-adjoint state :
    adj_reset()

    # starting time :
    t0   = time()

    # need this for the derivative callback :
    global counter
    counter = 0 
    
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
      print_min_max(I,    'I')
      for ci in c:
        print_min_max(ci,    'control:' + ci.name())
    
    # objective gradient callback function :
    def deriv_cb(I, dI, c):
      global counter
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
      for (dIi,ci) in zip(dI,c):
        print_min_max(ci,    'control:' + ci.name())
        print_min_max(dIi,   'dI/dcontrol:' + ci.name())
      
      # call that callback, if you want :
      if adj_callback is not None:
        adj_callback(I, dI, c)
   
    # get the cost, regularization, and objective functionals :
    I = self.J
    
    # define the control parameter :
    m = Control(control, value=control)
    m2 = Control(model.p, value=model.p)
    
    # create the reduced functional to minimize :
    F = ReducedFunctional(Functional(I), [m,m2], eval_cb_post=eval_cb,
                          derivative_cb_post=deriv_cb)

    # optimize with scipy's fmin_l_bfgs_b :
    if method == 'l_bfgs_b': 
      out = minimize(F, method="L-BFGS-B", tol=1e-8, bounds=bounds,
                     options={"disp"    : True,
                              "maxiter" : max_iter,
                              "gtol"    : 1e2})
      b_opt = out
    
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
                    "print_level"        : 5,
                    "ma97_order"         : "metis",
                    "linear_solver"      : "ma97"}
      solver = IPOPTSolver(problem, parameters=parameters)
      b_opt  = solver.solve()

    # make the optimal control parameter available :
    #model.assign_variable(control, b_opt)
    #Control(control).update(b_opt)  # FIXME: does this work?
    
    # call the post-adjoint callback function if set :
    if post_adj_callback is not None:
      s    = '::: calling optimize_u_ob() post-adjoined callback function :::'
      print_text(s, cls=self)
      post_adj_callback()
    
    # calculate total time to compute
    tf = time()
    s  = tf - t0
    m  = s / 60.0
    h  = m / 60.0
    s  = s % 60
    m  = m % 60
    text = "time to optimize ||u - u_ob||: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)


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
    A_shf      = model.A_shf
    A_gnd      = model.A_gnd
    N          = model.n
    eps_reg    = model.eps_reg
    h          = model.h
    n          = model.N
    D          = model.D

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
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    if stabilized:
      s  = "    - using stabilized elements -"
      Q4 = model.Q4
      self.assx  = FunctionAssigner(model.u.function_space(), model.Q4.sub(0))
      self.assy  = FunctionAssigner(model.v.function_space(), model.Q4.sub(1))
      self.assz  = FunctionAssigner(model.w.function_space(), model.Q4.sub(2))
      self.assp  = FunctionAssigner(model.p.function_space(), model.Q4.sub(3))
    else:
      s  = "    - using Taylor-Hood elements -"
      Q4 = model.QTH3
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
    
    # viscous dissipation :
    epsdot  = self.effective_strain_rate(u)
    if linear:
      s  = "    - using linear form of momentum using model.U3 in epsdot -"
      eta_shf, eta_gnd = self.viscosity(model.U3.copy(True))
    else:
      s  = "    - using nonlinear form of momentum -"
      eta_shf, eta_gnd = self.viscosity(u)
    print_text(s, self.color())

    alpha = Constant(1.0/10.0)
    gamma = Constant(1e2)
    f     = Constant((0.0, 0.0, -rhoi * g))
    I     = Identity(3)
    u_n   = Constant(0.0)

    # pressure boundary :
    ut    = u - dot(u,n)*n
    Pb_w  = - rhosw*g*D * n
    Pb_l  = - rhoi*g*(S - z) * n

    def epsilon(u): return 0.5*(grad(u) + grad(u).T)
    def sigma(u,p): return 2*eta_gnd * epsilon(u) - p*I
    def L(u,p):     return -div(sigma(u,p))
    
    t   = dot(sigma(u,p), n)
    s   = dot(sigma(v,q), n)

    if stabilized:
      B_o = + inner(sigma(u,p),grad(v))*dx + div(u)*q*dx \
            + alpha * h**2 * inner(L(u,p), L(v,q)) * dx
      
      B_g = - dot(n,t) * dot(v,n) * dBed \
            - dot(u,n) * dot(s,n) * dBed \
            + gamma/h * dot(u,n) * dot(v,n) * dBed \
            + beta * dot(ut, v) * dBed
      
      F   = + dot(f,v) * dx \
            + gamma/h * u_n * dot(v,n) * dBed \
            + alpha * h**2 * inner(f, L(v,q)) * dx
   
    else: 
      B_o = + inner(sigma(u,p),grad(v))*dx - div(u)*q*dx
      
      B_g = - dot(n,t) * dot(v,n) * dBed \
            - dot(u,n) * dot(s,n) * dBed \
            + gamma/h * dot(u,n) * dot(v,n) * dBed \
            + beta * dot(u, v) * dBed \
      
      F   = + dot(f,v) * dx \
            + gamma/h * u_n * dot(v,n) * dBed \

    self.mom_F = B_o + B_g - F
    
    if (not model.use_periodic_boundaries and use_pressure_bc):
      s = "    - using water pressure lateral boundary condition -"
      print_text(s, self.color())
      self.mom_F -= Pb_w*dLat_t
    
    if (not model.use_periodic_boundaries and not use_lat_bcs):
      s = "    - using internal divide lateral pressure boundary condition -"
      print_text(s, self.color())
      self.mom_F -= Pb_l*dLat_d
    
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using internal divide lateral stress natural boundary" + \
          " conditions -"
      print_text(s, self.color())
      U3_c                 = model.U3.copy(True)
      eta_shf_l, eta_gnd_l = self.viscosity(U3_c)
      sig_g_l    = self.stress_tensor(U3_c, model.p, eta_gnd_l)
      #sig_g_l    = self.stress_tensor(U2, p, eta_gnd)
      A += dot(dot(sig_g_l, n), u) * dLat_d

    # the first variation of the extremum in the direction
    # a trial function dU; the Jacobian :
    self.mom_Jac = derivative(self.mom_F, U, dU)

    # Form for use in constructing preconditioner matrix
    self.bp_R = inner(grad(u), grad(v))*dx + p*q*dx
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
                'linear_solver'            : 'superlu_dist',
                'relative_tolerance'       : 1e-6,
                'relaxation_parameter'     : 0.7,
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
    
    U3 = model.U3.split(True)

    print_min_max(U3[0],   'u')
    print_min_max(U3[1],   'v')
    print_min_max(U3[2],   'w')
    print_min_max(model.p, 'p')



