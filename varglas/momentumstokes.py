from fenics                 import *
from dolfin_adjoint         import *
from varglas.io             import print_text, print_min_max
from varglas.d3model        import D3Model
from varglas.physics        import Physics
from varglas.momentum       import Momentum
from copy                   import deepcopy
import sys


class MomentumStokes(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None, isothermal=True,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """
    """
    if type(model) != D3Model:
      s = ">>> MomentumStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    s = "::: INITIALIZING FULL-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    # momenturm and adjoint :
    G         = Function(model.MV, name = 'G')
    Lam       = Function(model.MV, name = 'Lam')
    dU        = TrialFunction(model.MV)
    Tst       = TestFunction(model.MV)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx = FunctionAssigner(model.u.function_space(), model.Q)
    self.assy = FunctionAssigner(model.v.function_space(), model.Q)
    self.assz = FunctionAssigner(model.w.function_space(), model.Q)
    self.assp = FunctionAssigner(model.p.function_space(), model.MV.sub(1))

    mesh      = model.mesh
    eps_reg   = model.eps_reg
    n         = model.n
    r         = model.r
    S         = model.S
    B         = model.B
    H         = S - B
    z         = model.x[2]
    W         = model.W
    R         = model.R
    rhoi      = model.rhoi
    rhosw     = model.rhosw
    g         = model.g
    beta      = model.beta
    h         = model.h
    N         = model.N
    D         = model.D

    gradS     = grad(S)
    gradB     = grad(B)
    
    dx_f      = model.dx_f
    dx_g      = model.dx_g
    dx        = model.dx
    dBed_g    = model.dBed_g
    dBed_f    = model.dBed_f
    dLat_t    = model.dLat_t
    dBed      = model.dBed
     
    # new constants :
    p0        = 101325
    T0        = 288.15
    M         = 0.0289644
    
    #===========================================================================
    # define variational problem :
    du,  dp   = split(dU)
    U,   p    = split(G)
    Phi, xi   = split(Tst)

    u,   v,   w   = U
    phi, psi, chi = Phi
   
    # form the viscosity : 
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot = self.effective_strain_rate(model.U3.copy(True))
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot   = self.effective_strain_rate(U)
    
    if isothermal:
      s   = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      b_shf = model.E_shf * model.b_shf
      b_gnd = model.E_gnd * model.b_gnd

    else:
      s   = "    - using temperature-dependent rate-factor -"
      print_text(s, self.color())
      T       = model.T
      W       = model.W
      R       = model.R
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
      W_T     = conditional( lt(W, 0.01),   W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
    
    eta_shf   = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
    eta_gnd   = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
    
    # gravity vector :
    gv        = as_vector([0, 0, -g])
    f_w       = rhoi*g*(S - z) + rhosw*g*D
    I         = Identity(3)

    epi       = self.strain_rate_tensor(U)
    tau_shf   = 2*eta_shf*epi
    tau_gnd   = 2*eta_gnd*epi
    sigma_shf = tau_shf - p*I
    sigma_gnd = tau_gnd - p*I
    tau_n     = dot(N, dot(tau_gnd, N))
    
    # conservation of momentum :
    R1 = + inner(sigma_shf, grad(Phi)) * dx_f \
         + inner(sigma_gnd, grad(Phi)) * dx_g \
         - rhoi * dot(gv, Phi) * dx \
         + beta * u * phi * dBed_g \
         + beta * v * psi * dBed_g \
         + beta * w * chi * dBed_g \
         - (2*eta_gnd*u.dx(0) - tau_n)*N[0] * phi * dBed_g \
         - eta_gnd*(u.dx(1) + v.dx(0))*N[1] * phi * dBed_g \
         - eta_gnd*(u.dx(2) + w.dx(0))*N[2] * phi * dBed_g \
         - eta_gnd*(u.dx(1) + v.dx(0))*N[0] * psi * dBed_g \
         - (2*eta_gnd*v.dx(1) - tau_n)*N[1] * psi * dBed_g \
         - eta_gnd*(v.dx(2) + w.dx(1))*N[2] * psi * dBed_g \
         #- f_w * dot(N, Phi) * dBed_f \
         #- dot(dot(N, dot(tau_gnd, N)) * N, Phi) * dBed_g \
         #- p_a * dot(N, Phi) * dSrf \
         #+ beta * dot(U, Phi) * dBed_g \
    
    if (not model.use_periodic_boundaries 
        and not use_lat_bcs and use_pressure_bc):
      s = "    - using cliff-pressure boundary condition -"
      print_text(s, self.color())
      R1 -= f_w * dot(N, Phi) * dLat_t
     
    # conservation of mass :
    R2 = + div(U)*xi*dx \
         #+ beta*(u*B.dx(0) + v*B.dx(1))*chi*dBed_g \
         #+ dot(U, N)*xi*dBed
    
    # total residual :
    self.mom_F = R1 + R2
    
    self.mom_Jac = derivative(self.mom_F, G, dU)
   
    self.mom_bcs = []
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using divide-lateral boundary conditions -"
      print_text(s, self.color())

      self.mom_bcs.append(DirichletBC(model.MV.sub(0).sub(0),
                                      model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.MV.sub(0).sub(1),
                                      model.v_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.MV.sub(0).sub(2),
                                      model.w_lat, model.ff, model.GAMMA_L_DVD))

    #self.mom_bcs.append(DirichletBC(model.MV.sub(0), 
    #                                Constant((0,0,0)), model.ff, 
    #                                model.GAMMA_B_GND))
    
    self.G       = G
    self.U       = U 
    self.p       = p
    self.dU      = dU
    self.Tst     = Tst
    self.Lam     = Lam
    self.epsdot  = epsdot
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
  
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

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params
  
  def strain_rate_tensor(self, U):
    """
    return the strain-rate tensor of <U>.
    """
    return 0.5 * (grad(U) + grad(U).T)

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

  def stress_tensor(self):
    """
    return the BP Cauchy stress tensor.
    """
    # FIXME: needs eta
    s   = "::: forming the Cauchy stress tensor :::"
    print_text(s, self.color())
    epi   = self.strain_rate_tensor(self.U)
    I     = Identity(3)

    sigma = 2*self.eta*epi - self.p*I
    return sigma

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
    s      = "::: solving BP horizontal velocity with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.mom_F == 0, self.G, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    U, p = self.G.split()
    u, v, w = split(U)

    self.assx.assign(model.u, project(u, model.Q, annotate=False),
                     annotate=False)
    self.assy.assign(model.v, project(v, model.Q, annotate=False),
                     annotate=False)
    self.assz.assign(model.w, project(w, model.Q, annotate=False),
                     annotate=False)
    self.assp.assign(model.p, p, annotate=False)

    print_min_max(model.U3, 'U', cls=self)
    print_min_max(model.p,  'p', cls=self)


class MomentumDukowiczStokesReduced(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None, isothermal=True,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    if type(model) != D3Model:
      s = ">>> MomentumStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    s = "::: INITIALIZING DUKOWICZ REDUCED FULL-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    # NOTE: not sure why this is ever changed, but the model.assimilate_data
    #       method throws an error if I don't do this :
    parameters["adjoint"]["stop_annotating"] = False

    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

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

    #w = u*B.dx(0) + v*B.dx(1) - (u.dx(0) + v.dx(1))*(z - B)
    w = - u.dx(0)*(z - B) + u*B.dx(0) - v.dx(1)*(z - B) + v*B.dx(1)
    
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
      #theta   = model.theta
      W       = model.W
      R       = model.R
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      T_c     = 263.15
      #theta_c = 146.3*T_c + 7.253/2.0*T_c**2
      #a_T     = conditional( lt(theta, theta_c), 1.1384496e-5, 5.45e10)
      #Q_T     = conditional( lt(theta, theta_c), 6e4,          13.9e4)
      #W_T     = conditional( lt(W,     0.01),    W,            0.01)
      a_T     = conditional( lt(T, T_c),  1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, T_c),  6e4,          13.9e4)
      W_T     = conditional( lt(W, 0.01), W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
   
    # 1) Viscous dissipation
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot_l  = self.effective_strain_rate(model.U3.copy(True))
      epsdot    = self.effective_strain_rate(as_vector([u,v,w]))
      eta_shf   = 0.5 * b_shf * (epsdot_l + eps_reg)**((1-n)/(2*n))
      eta_gnd   = 0.5 * b_gnd * (epsdot_l + eps_reg)**((1-n)/(2*n))
      Vd_shf    = 2 * eta_shf * epsdot
      Vd_gnd    = 2 * eta_gnd * epsdot
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot  = self.effective_strain_rate(as_vector([u,v,w]))
      eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      Vd_shf  = (2*n)/(n+1) * b_shf * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd  = (2*n)/(n+1) * b_gnd * (epsdot + eps_reg)**((n+1)/(2*n))
      
    # 2) Potential energy
    Pe     = - rhoi * g * (u*S.dx(0) + v*S.dx(1))

    # 3) Dissipation by sliding
    Sl_gnd = - 0.5 * beta * (u**2 + v**2)

    # 4) pressure boundary
    Pb     = (rhoi*g*(S - z) - rhosw*g*D) * (u*N[0] + v*N[1])

    # Variational principle
    A      = + Vd_shf*dx_f + Vd_gnd*dx_g - Pe*dx \
             - Sl_gnd*dBed_g - Pb*dBed_f
    
    if (not model.use_periodic_boundaries 
        and not use_lat_bcs and use_pressure_bc):
      s = "    - using calving-front-pressure-boundary condition -"
      print_text(s, self.color())
      A -= Pb*dLat_t

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.mom_F = derivative(A, U, Phi)

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []
      
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using divide-lateral boundary conditions -"
      print_text(s, self.color())
      self.mom_bcs.append(DirichletBC(model.Q2.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.Q2.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
    
    self.eta_shf = eta_shf
    self.eta_gnd = eta_gnd
    self.b_shf   = b_shf
    self.b_gnd   = b_gnd
    self.A       = A
    self.U       = U 
    self.w       = w
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
    print_text(s, self.color)

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
    nparams = {'newton_solver' : {'linear_solver'            : 'cg',
                                  'preconditioner'           : 'hypre_amg',
                                  'relative_tolerance'       : 1e-8,
                                  'relaxation_parameter'     : 1.0,
                                  'maximum_iterations'       : 25,
                                  'error_on_nonconvergence'  : False}}
    m_params  = {'solver'         : nparams,
                 'solve_pressure' : True}
    return m_params
  
  #def solve_pressure(self, annotate=True):
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

  def solve_vert_velocity(self, annotate=annotate):
    """ 
    Solve for vertical velocity w.
    """
    model  = self.model
    
    s    = "::: solving Dukowicz reduced vertical velocity :::"
    print_text(s, self.color())
    w = project(self.w, model.Q, annotate=annotate)

    self.assz.assign(model.w, w, annotate=False)

  def solve(self, annotate=True):
    """ 
    Perform the Newton solve of the reduced full-Stokes equations 
    """
    model  = self.model
    params = self.solve_params
    
    # solve nonlinear system :
    rtol   = params['solver']['newton_solver']['relative_tolerance']
    maxit  = params['solver']['newton_solver']['maximum_iterations']
    alpha  = params['solver']['newton_solver']['relaxation_parameter']
    s    = "::: solving Dukowicz full-Stokes reduced equations with %i max" + \
             " iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v = self.U.split()
  
    self.assx.assign(model.u, u, annotate=False)
    self.assy.assign(model.v, v, annotate=False)

    # solve for the vertical velocity :
    if params['solve_vert_velocity']:
      self.solve_vert_velocity(annotate)
    if params['solve_pressure']:
      self.solve_pressure(annotate)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0], 'u', cls=self)
    print_min_max(U3[1], 'v', cls=self)
    print_min_max(U3[2], 'w', cls=self)


class MomentumDukowiczStokes(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None, isothermal=True,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING DUKOWICZ FULL-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D3Model:
      s = ">>> MomentumStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    # momenturm and adjoint :
    U      = Function(model.Q5, name = 'G')
    Lam    = Function(model.Q5, name = 'Lam')
    dU     = TrialFunction(model.Q5)
    Phi    = TestFunction(model.Q5)
   
    # function assigner goes from the U function solve to U3 vector 
    # function used to save :
    self.assx  = FunctionAssigner(model.u.function_space(),   model.Q5.sub(0))
    self.assy  = FunctionAssigner(model.v.function_space(),   model.Q5.sub(1))
    self.assz  = FunctionAssigner(model.w.function_space(),   model.Q5.sub(2))
    self.assp  = FunctionAssigner(model.p.function_space(),   model.Q5.sub(3))
    self.assl  = FunctionAssigner(model.lam.function_space(), model.Q5.sub(4))

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
    dBed       = model.dBed
     
    # new constants :
    p0         = 101325
    T0         = 288.15
    M          = 0.0289644
    
    #===========================================================================
    # define variational problem :
    phi, psi, xsi, kappa, gamma = Phi
    du,  dv,  dw,  dP,    dl    = dU
    u,   v,   w,   p,     lam   = U
    
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
      W       = model.W
      R       = model.R
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
      W_T     = conditional( lt(W, 0.01),   W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
   
    # 1) viscous dissipation :
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot_l  = self.effective_strain_rate(model.U3.copy(True))
      epsdot    = self.effective_strain_rate(as_vector([u,v,w]))
      eta_shf   = 0.5 * b_shf * (epsdot_l + eps_reg)**((1-n)/(2*n))
      eta_gnd   = 0.5 * b_gnd * (epsdot_l + eps_reg)**((1-n)/(2*n))
      Vd_shf    = 2 * eta_shf * epsdot
      Vd_gnd    = 2 * eta_gnd * epsdot
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot  = self.effective_strain_rate(as_vector([u,v,w]))
      eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      Vd_shf  = (2*n)/(n+1) * b_shf * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd  = (2*n)/(n+1) * b_gnd * (epsdot + eps_reg)**((n+1)/(2*n))
   
    # 2) potential energy :
    Pe   = - rhoi * g * w

    # 3) dissipation by sliding :
    Sl   = - 0.5 * beta * (u**2 + v**2 + w**2)

    # 4) incompressibility constraint :
    Pc   = p * (u.dx(0) + v.dx(1) + w.dx(2))
    
    # 5) inpenetrability constraint
    Nc   = - lam * (u*N[0] + v*N[1] + w*N[2])

    # 6) pressure boundary
    Pb   = - rhosw*g*D * (u*N[0] + v*N[1] + w*N[2])

    # stabilization :
    f       = rhoi * Constant((0.0, 0.0, -g))
    tau_shf = h**2 / (12 * b_shf * rhoi**2)
    tau_gnd = h**2 / (12 * b_gnd * rhoi**2)
    #tau_shf = h**2 / (12 * eta_shf)
    #tau_gnd = h**2 / (12 * eta_gnd)
    Lsq_shf = tau_shf * dot( (grad(p) - f), (grad(p) - f) )
    Lsq_gnd = tau_gnd * dot( (grad(p) - f), (grad(p) - f) )
    
    # variational principle :
    #A      = + Vd_shf*dx_f + Vd_gnd*dx_g \
    A      = + (Vd_shf - Lsq_shf)*dx_f + (Vd_gnd - Lsq_gnd)*dx_g \
             - (Pe + Pc)*dx - Nc*dBed - Sl*dBed_g - Pb*dBed_f
    
    if (not model.use_periodic_boundaries 
        and not use_lat_bcs and use_pressure_bc):
      s = "    - using cliff-pressure boundary condition -"
      print_text(s, self.color())
      A -= Pb*dLat_t

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.mom_F = derivative(A, U, Phi)

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.mom_Jac = derivative(self.mom_F, U, dU)
    
    self.mom_bcs = []

    ## pressure is zero on the surface :
    #self.mom_bcs.append(DirichletBC(model.Q5.sub(3),
    #                    Constant(0.0), model.ff, model.GAMMA_S_GND))
    #self.mom_bcs.append(DirichletBC(model.Q5.sub(3),
    #                    Constant(0.0), model.ff, model.GAMMA_S_FLT))
    #self.mom_bcs.append(DirichletBC(model.Q5.sub(3),
    #                    Constant(0.0), model.ff, model.GAMMA_U_GND))
    #self.mom_bcs.append(DirichletBC(model.Q5.sub(3),
    #                    Constant(0.0), model.ff, model.GAMMA_U_FLT))
  
    # add lateral boundary conditions :  
    if use_lat_bcs:
      s = "    - using divide-lateral boundary conditions -"
      print_text(s, self.color())

      self.mom_bcs.append(DirichletBC(model.Q5.sub(0),
                          model.u_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.Q5.sub(1),
                          model.v_lat, model.ff, model.GAMMA_L_DVD))
      self.mom_bcs.append(DirichletBC(model.Q5.sub(2),
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
    s    = "::: solving Dukowicz-full-Stokes equations with %i max" + \
             " iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.mom_F == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
          annotate = annotate, solver_parameters = params['solver'])
    u, v, w, p, l = self.U.split()
    
    self.assx.assign(model.u,   u, annotate=False)
    self.assy.assign(model.v,   v, annotate=False)
    self.assz.assign(model.w,   w, annotate=False)
    self.assp.assign(model.p,   p, annotate=False)
    self.assl.assign(model.lam, l, annotate=False)
    
    U3 = model.U3.split(True)

    print_min_max(U3[0],     'u',   cls=self)
    print_min_max(U3[1],     'v',   cls=self)
    print_min_max(U3[2],     'w',   cls=self)
    print_min_max(model.p,   'p',   cls=self)
    print_min_max(model.lam, 'lam', cls=self)


class MomentumDukowiczBrinkerhoffStokes(Momentum):
  """  
  """
  def initialize(self, model, solve_params=None, isothermal=True,
                 linear=False, use_lat_bcs=False, use_pressure_bc=True):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING DUKOWICZ-BRINKERHOFF-FULL-STOKES PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D3Model:
      s = ">>> MomentumStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

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
    phi, psi, xsi, kappa = Phi
    du,  dv,  dw,  dP    = dU
    u,   v,   w,   p     = U
    
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
      W       = model.W
      R       = model.R
      E_shf   = model.E_shf
      E_gnd   = model.E_gnd
      a_T     = conditional( lt(T, 263.15), 1.1384496e-5, 5.45e10)
      Q_T     = conditional( lt(T, 263.15), 6e4,          13.9e4)
      W_T     = conditional( lt(W, 0.01),   W,            0.01)
      b_shf   = ( E_shf*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
      b_gnd   = ( E_gnd*a_T*(1 + 181.25*W_T)*exp(-Q_T/(R*T)) )**(-1/n)
   
    # 1) Viscous dissipation
    if linear:
      s   = "    - using linear form of momentum using model.U3 in epsdot -"
      print_text(s, self.color())
      epsdot_l  = self.effective_strain_rate(model.U3.copy(True))
      epsdot    = self.effective_strain_rate(as_vector([u,v,w]))
      eta_shf   = 0.5 * b_shf * (epsdot_l + eps_reg)**((1-n)/(2*n))
      eta_gnd   = 0.5 * b_gnd * (epsdot_l + eps_reg)**((1-n)/(2*n))
      Vd_shf    = 2 * eta_shf * epsdot
      Vd_gnd    = 2 * eta_gnd * epsdot
    else:
      s   = "    - using nonlinear form of momentum -"
      print_text(s, self.color())
      epsdot  = self.effective_strain_rate(as_vector([u,v,w]))
      eta_shf = 0.5 * b_shf * (epsdot + eps_reg)**((1-n)/(2*n))
      eta_gnd = 0.5 * b_gnd * (epsdot + eps_reg)**((1-n)/(2*n))
      Vd_shf  = (2*n)/(n+1) * b_shf * (epsdot + eps_reg)**((n+1)/(2*n))
      Vd_gnd  = (2*n)/(n+1) * b_gnd * (epsdot + eps_reg)**((n+1)/(2*n))
   
    # 2) potential energy :
    Pe     = - rhoi * g * w

    # 3) dissipation by sliding :
    Sl_gnd = - 0.5 * beta * (u**2 + v**2 + w**2)

    # 4) incompressibility constraint :
    Pc     = p * (u.dx(0) + v.dx(1) + w.dx(2)) 
    
    # 5) inpenetrability constraint :
    Nc     = - p * (u*N[0] + v*N[1] + w*N[2])

    # 6) pressure boundary :
    Pb_w   = - rhosw*g*D * (u*N[0] + v*N[1] + w*N[2])
    Pb_l   =   rhoi*g*(S - z) * (u*N[0] + v*N[1] + w*N[2])

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
    #nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
    #                              'relative_tolerance'       : 1e-8,
    #                              'relaxation_parameter'     : 1.0,
    #                              'maximum_iterations'       : 25,
    #                              'error_on_nonconvergence'  : False}}
    nparams = {'newton_solver' :
              {
                'linear_solver'            : 'tfqmr',
                'preconditioner'           : 'petsc_amg',
                'relative_tolerance'       : 1e-8,
                'relaxation_parameter'     : 1.0,
                'maximum_iterations'       : 25,
                'error_on_nonconvergence'  : False,
                'krylov_solver'            :
                {
                  'monitor_convergence'   : False,
                  'preconditioner' :
                  {
                    'structure' : 'same'
                  }
                }
              }}
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

    print_min_max(U3[0],   'u', cls=self)
    print_min_max(U3[1],   'v', cls=self)
    print_min_max(U3[2],   'w', cls=self)
    print_min_max(model.p, 'p', cls=self)



