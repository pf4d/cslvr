from fenics               import *
from dolfin_adjoint       import *
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d3model        import D3Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
from copy                 import deepcopy
import sys



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

    mesh       = model.mesh
    r          = model.r
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
    phi, psi = Phi
    du,  dv  = dU
    u,   v   = U
    
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

    # alternative form :
    #u,v,w = U
    #epsdot = 0.5 * (+ (u.dx(0))**2 + (v.dx(1))**2 + (u.dx(0) + v.dx(1))**2 \
    #                + 0.5*(u.dx(1) + v.dx(0))**2 \
    #                + 0.5*((u.dx(2) + w.dx(0))**2 + (v.dx(2) + w.dx(1))**2))
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
                'maximum_iterations'       : 25,
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
    print_min_max(self.w, 'w', cls=self)

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
    model.assign_variable(self.get_U(), DOLFIN_EPS, cls=self)

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

    print_min_max(U3[0], 'u', cls=self)
    print_min_max(U3[1], 'v', cls=self)
    print_min_max(U3[2], 'w', cls=self)


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

    mesh       = model.mesh
    r          = model.r
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
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
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
      Vd_shf           = 2 * eta_shf * epsdot
      Vd_gnd           = 2 * eta_gnd * epsdot
    else:
      s  = "    - using nonlinear form of momentum -"
      eta_shf, eta_gnd = self.viscosity(U3)
      Vd_shf   = (2*n)/(n+1) * A_shf**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd   = (2*n)/(n+1) * A_gnd**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
    print_text(s, self.color())
   
    # potential energy :
    Pe     = - rhoi * g * w

    # dissipation by sliding :
    Ut     = U3 - dot(U3,N)*N
    Sl_gnd = - 0.5 * beta * dot(Ut, Ut)

    # incompressibility constraint :
    Pc     = p * div(U3)
    
    # impenetrability constraint :
    #sig_f  = self.stress_tensor(U3, p, eta_shf)
    #sig_g  = self.stress_tensor(U3, p, eta_gnd)
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
    model.assign_variable(self.get_U(), DOLFIN_EPS, cls=self)
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v, w, p = self.U.split()
    
    self.assx.assign(model.u, u, annotate=False)
    self.assy.assign(model.v, v, annotate=False)
    self.assz.assign(model.w, w, annotate=False)
    self.assp.assign(model.p, p, annotate=False)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0],   'u', cls=self)
    print_min_max(U3[1],   'v', cls=self)
    print_min_max(U3[2],   'w', cls=self)
    print_min_max(model.p, 'p', cls=self)



