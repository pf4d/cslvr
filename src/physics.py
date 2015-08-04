r"""
This module contains the physics objects for the flowline ice model, does all 
of the finite element work, and Newton solves for non-linear equations.

**Classes**

:class:`~src.physics.AdjointVelocityBP` -- Linearized adjoint of 
Blatter-Pattyn, and functions for using it to calculate the gradient and 
objective function

:class:`~src.physics.Age` -- Solves the pure advection equation for ice 
age with a age zero Dirichlet boundary condition at the ice surface above the 
ELA. 
Stabilized with Streamline-upwind-Petrov-Galerking/GLS.

:class:`~src.physics.Enthalpy` -- Advection-Diffusion equation for ice sheets

:class:`~src.physics.FreeSurface` -- Calculates the change in surface 
elevation, and updates the mesh and surface function

:class:`~src.physics.SurfaceClimate` -- PDD and surface temperature model 
based on lapse rates

:class:`~src.physics.VelocityStokes` -- Stokes momentum balance

:class:`~src.physics.VelocityBP` -- Blatter-Pattyn momentum balance
"""

from pylab      import ndarray, where
from fenics     import *
from termcolor  import colored, cprint
from helper     import raiseNotDefined, VerticalBasis, VerticalFDBasis, \
                       VerticalIntegrator
from io         import print_text, print_min_max
import numpy as np
import numpy.linalg as linalg


class Physics(object):
  """
  This abstract class outlines the structure of a physics calculation.
  """
  def solve(self):
    """
    Solves the physics calculation.
    """
    raiseNotDefined()
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'cyan'


class VelocityDukowiczStokes(Physics):
  """  
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING DUKOWICZ FULL-STOKES PHYSICS :::"
    print_text(s, self.color())

    self.model    = model
    self.config   = config

    mesh          = model.mesh
    r             = config['velocity']['r']
    V             = model.V
    Q             = model.Q
    Q4            = model.Q4
    U             = model.U
    dU            = model.dU
    Phi           = model.Phi
    n             = model.n
    b_shf         = model.b_shf
    b_gnd         = model.b_gnd
    eta_shf       = model.eta_shf
    eta_gnd       = model.eta_gnd
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    W             = model.W
    R             = model.R
    rhoi          = model.rhoi
    rhow          = model.rhow
    g             = model.g
    beta          = model.beta
    h             = model.h
    N             = model.N
    D             = model.D

    gradS         = grad(S)
    gradB         = grad(B)

    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    ds       = model.ds  
    dGnd     = ds(3)         # grounded bed
    dFlt     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dFlt   # bed
    
    #===========================================================================
    # define variational problem :
   
    phi, psi, xsi, kappa = Phi
    du,  dv,  dw,  dP    = dU
    u,   v,   w,   P     = U
      
    # 1) Viscous dissipation
    Vd_shf   = model.Vd_shf
    Vd_gnd   = model.Vd_gnd

    # 2) Potential energy
    Pe     = rhoi * g * w

    # 3) Dissipation by sliding
    Sl_gnd = 0.5 * beta**2 * H**r * (u**2 + v**2 + w**2)
    Sl_shf = 0.5 * Constant(1e-10) * (u**2 + v**2 + w**2)

    # 4) Incompressibility constraint
    Pc     = -P * (u.dx(0) + v.dx(1) + w.dx(2)) 
    
    # 5) Impenetrability constraint
    Nc     = P * (u*N[0] + v*N[1] + w*N[2])

    # 6) pressure boundary
    Pb     = - (rhoi*g*(S - x[2]) + rhow*g*D) * (u*N[0] + v*N[1] + w*N[2]) 

    f       = rhoi * Constant((0.0, 0.0, g))
    tau_shf = h**2 / (12 * b_shf * rhoi**2)
    tau_gnd = h**2 / (12 * b_gnd * rhoi**2)
    Lsq_shf = -tau_shf * dot( (grad(P) + f), (grad(P) + f) )
    Lsq_gnd = -tau_gnd * dot( (grad(P) + f), (grad(P) + f) )
    
    # Variational principle
    A      = + (Vd_shf + Lsq_shf)*dx_s + (Vd_gnd + Lsq_gnd)*dx_g \
             + (Pe + Pc)*dx + Sl_gnd*dGnd + Sl_shf*dFlt + Nc*dBed
    if (not config['periodic_boundary_conditions']
        and not config['velocity']['use_lat_bcs']
        and config['use_pressure_boundary']):
      A += Pb*dSde

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.F = derivative(A, U, Phi)   

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.J = derivative(self.F, U, dU)
    
    self.bcs = []
      
    # add lateral boundary conditions :  
    if config['velocity']['use_lat_bcs']:
      self.bcs.append(DirichletBC(Q3.sub(0), model.u_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(Q3.sub(1), model.v_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(Q3.sub(2), model.w_lat_bc, model.ff, 4))
    
    # keep the residual for adjoint solves :
    model.A       = A

  def solve(self):
    """ 
    Perform the Newton solve of the full-Stokes equations 
    """
    model  = self.model
    config = self.config
    
    # Solve the nonlinear equations via Newton's method
    s    = "::: solving Dukowicz full-Stokes equations :::"
    print_text(s, self.color())
    solve(self.F == 0, model.U, bcs=self.bcs, J = self.J, 
          solver_parameters = config['velocity']['newton_params'])
    u, v, w, P = model.U.split(True)

    model.assign_variable(model.u, u)
    model.assign_variable(model.v, v)
    model.assign_variable(model.w, w)
    model.assign_variable(model.P, P)
     
    print_min_max(model.u, 'u')
    print_min_max(model.v, 'v')
    print_min_max(model.w, 'w')
    print_min_max(model.P, 'P')


class VelocityStokes(Physics):
  """  
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING FULL-STOKES PHYSICS :::"
    print_text(s, self.color())

    self.model    = model
    self.config   = config

    mesh          = model.mesh
    r             = config['velocity']['r']
    V             = model.MV
    Q             = model.Q
    G             = model.U
    dU            = model.dU
    Tst           = model.Phi
    n             = model.n
    eta_shf       = model.eta_shf
    eta_gnd       = model.eta_gnd
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    W             = model.W
    R             = model.R
    epsdot        = model.epsdot
    eps_reg       = model.eps_reg
    rhoi          = model.rhoi
    rhow          = model.rhow
    g             = model.g
    beta          = model.beta
    h             = model.h
    N             = model.N
    D             = model.D

    gradS         = grad(S)
    gradB         = grad(B)
     
    # new constants :
    p0     = 101325
    T0     = 288.15
    M      = 0.0289644
    ci     = model.ci

    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    ds       = model.ds  
    dGnd     = ds(3)         # grounded bed
    dShf     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dShf   # bed
    dSrf     = ds(2) + ds(6)
    
    #===========================================================================
    # define variational problem :
    du,  dp = split(dU)
    U,   P  = split(G)
    Phi, xi = split(Tst)
    
    # gravity vector :
    gv   = as_vector([0, 0, -g])
    f_w  = rhoi*g*(S - x[2]) + rhow*g*D
    I    = Identity(3)

    epi       = model.strain_rate_tensor(U)
    sigma_shf = 2*eta_shf*epi - P*I
    sigma_gnd = 2*eta_gnd*epi - P*I
    
    # conservation of momentum :
    R1 = + inner(sigma_shf, grad(Phi)) * dx_s \
         + inner(sigma_gnd, grad(Phi)) * dx_g \
         - rhoi * dot(gv, Phi) * dx \
         + beta**2 * dot(U, Phi) * dGnd \
         #- p_a * dot(N, Phi) * dSrf \
    
    if (not config['periodic_boundary_conditions']
        and not config['velocity']['use_lat_bcs']
        and config['use_pressure_boundary']):
      R1 -= f_w * dot(N, Phi) * dSde \
    
    # conservation of mass :
    R2 = div(U)*xi*dx + dot(U, N)*xi*dBed
    
    # total residual :
    self.A = R1 + R2
    
    self.J = derivative(self.A, G, dU)
   
    self.bcs = []
      
    # add lateral boundary conditions :  
    if config['velocity']['use_lat_bcs']:
      self.bcs.append(DirichletBC(V.sub(0), model.u_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(V.sub(1), model.v_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(V.sub(2), model.w_lat_bc, model.ff, 4))

    # keep the residual for adjoint solves :
    model.A       = self.A

  def solve(self):
    """ 
    Perform the Newton solve of the full-Stokes equations 
    """
    model  = self.model
    config = self.config
    
    # solve nonlinear system :
    params = config['velocity']['newton_params']
    rtol   = params['newton_solver']['relative_tolerance']
    maxit  = params['newton_solver']['maximum_iterations']
    alpha  = params['newton_solver']['relaxation_parameter']
    s      = "::: solving full-Stokes equations with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.A == 0, model.U, bcs=self.bcs, J = self.J, 
          solver_parameters = config['velocity']['newton_params'])
    U, P    = model.U.split()
    u, v, w = U.split(True)

    model.assign_variable(model.u, u)
    model.assign_variable(model.v, v)
    model.assign_variable(model.w, w)
    model.assign_variable(model.P, P)
     
    print_min_max(model.u, 'u')
    print_min_max(model.v, 'v')
    print_min_max(model.w, 'w')
    print_min_max(model.P, 'P')


class VelocityDukowiczBP(Physics):
  """				
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING DUKOWICZ BP VELOCITY PHYSICS :::"
    print_text(s, self.color())

    self.model    = model
    self.config   = config

    mesh     = model.mesh
    r        = config['velocity']['r']
    V        = model.V
    Q        = model.Q
    Q2       = model.Q2
    U        = model.U
    Phi      = model.Phi
    dU       = model.dU
    S        = model.S
    B        = model.B
    H        = S - B
    x        = model.x
    R        = model.R
    rhoi     = model.rhoi
    rhow     = model.rhow
    g        = model.g
    beta     = model.beta
    w        = model.w
    N        = model.N
    D        = model.D
    
    gradS    = grad(S)
    gradB    = grad(B)

    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    ds       = model.ds  
    dGnd     = ds(3)         # grounded bed
    dFlt     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dFlt   # bed
    
    #===========================================================================
    # define variational problem :
    du,  dv  = dU
    u,   v   = U

    # vertical velocity components :
    chi      = TestFunction(Q)
    dw       = TrialFunction(Q)

    # 1) viscous dissipation :
    Vd_shf   = model.Vd_shf 
    Vd_gnd   = model.Vd_gnd

    # 2) potential energy :
    Pe       = rhoi * g * (u*gradS[0] + v*gradS[1])

    # 3) dissipation by sliding :
    #Ne       = H + rhow/rhoi * D
    #p        = 1.0#-0.383
    #q        = -0.349
    #Unorm    = sqrt(inner(U,U) + DOLFIN_EPS)
    #coef     = beta**(-2/p) * Ne**(-q/p) * p/(p+1)
    #coef     = beta * Ne**(-q/p) * p/(p+1)
    #Sl_gnd   = + coef * abs(u + DOLFIN_EPS)**(1/p + 1) * u/Unorm \
    #           + coef * abs(v + DOLFIN_EPS)**(1/p + 1) * v/Unorm
    Sl_shf   = 0.5 * Constant(DOLFIN_EPS) * H**r * (u**2 + v**2)
    Sl_gnd   = 0.5 * beta**2 * H**r * (u**2 + v**2)
    
    # 4) pressure boundary :
    Pb       = - (rhoi*g*(S - x[2]) + rhow*g*D) * (u*N[0] + v*N[1]) 

    # 5) tangential velocity to divide :
    Db       = u**2*N[0] + v**2*N[1]

    # Variational principle
    A        = Vd_shf*dx_s + Vd_gnd*dx_g + Pe*dx + Sl_gnd*dGnd
    if (not config['periodic_boundary_conditions']
        and not config['velocity']['use_lat_bcs']
        and config['use_pressure_boundary']):
      A += Pb*dSde #+ Db*ds(7)

    # Calculate the first variation of the action 
    # in the direction of the test function
    self.F   = derivative(A, U, Phi)

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.J   = derivative(self.F, U, dU)
   
    self.w_R = (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx - \
               (u*N[0] + v*N[1] + dw*N[2])*chi*dBed
    
    # Set up linear solve for vertical velocity.
    self.aw = lhs(self.w_R)
    self.Lw = rhs(self.w_R)
    
    # list of boundary conditions
    self.bcs  = []
    self.bc_w = None
      
    # add lateral boundary conditions :  
    if config['velocity']['use_lat_bcs']:
      self.bcs.append(DirichletBC(Q2.sub(0), model.u_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(Q2.sub(1), model.v_lat_bc, model.ff, 4))
      self.bc_w = DirichletBC(Q, model.w_lat_bc, model.ff, 4)
      
    # add boundary condition for the divide :
    # FIXME: this is a hack
    #self.bcs.append(DirichletBC(Q2.sub(0), 0.0, model.ff, 7))
    #self.bcs.append(DirichletBC(Q2.sub(1), 0.0, model.ff, 7))

    # keep the residual for adjoint solves :
    model.A       = A

  def solve_vert_velocity(self):
    """ 
    Solve for the vertical velocity 'w'. 
    """
    model  = self.model
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving Dukowicz BP vertical velocity :::"
    print_text(s, self.color())
    
    sm = config['velocity']['vert_solve_method']
    
    aw       = assemble(self.aw)
    Lw       = assemble(self.Lw)
    if self.bc_w != None:
      self.bc_w.apply(aw, Lw)
    w_solver = LUSolver(sm)
    w_solver.solve(aw, model.w.vector(), Lw)
    #solve(self.aw == self.Lw, model.w, bcs = self.bc_w,
    #      solver_parameters = {"linear_solver" : sm})
    print_min_max(model.w, 'w')

  def solve_pressure(self):
    """
    Solve for the BP pressure 'p'.
    """
    model  = self.model
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving Dukowicz BP pressure :::"
    print_text(s, self.color())
    
    Q       = model.Q
    rhoi    = model.rhoi
    g       = model.g
    S       = model.S
    x       = model.x
    w       = model.w
    p       = model.p
    eta_shf = model.eta_shf
    eta_gnd = model.eta_gnd

    p_shf   = project(rhoi*g*(S - x[2]) + 2*eta_shf*w.dx(2), Q)
    p_gnd   = project(rhoi*g*(S - x[2]) + 2*eta_gnd*w.dx(2), Q)
    
    # unify the enhancement factor over shelves and grounded ice : 
    p_v     = p.vector().array()
    p_gnd_v = p_gnd.vector().array()
    p_shf_v = p_shf.vector().array()
    p_v[self.gnd_dofs] = p_gnd_v[self.gnd_dofs]
    p_v[self.shf_dofs] = p_shf_v[self.shf_dofs]
    self.assign_variable(p, p_v)
    
    print_min_max(p, 'p')

  def solve(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    config = self.config
    
    # solve nonlinear system :
    params = config['velocity']['newton_params']
    rtol   = params['newton_solver']['relative_tolerance']
    maxit  = params['newton_solver']['maximum_iterations']
    alpha  = params['newton_solver']['relaxation_parameter']
    s      = "::: solving Dukowicz BP horizontal velocity with %i max " + \
             "iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    solve(self.F == 0, model.U, J = self.J, bcs = self.bcs,
          solver_parameters = params)
    u, v = model.U.split(True)
    
    model.assign_variable(model.u, u)
    model.assign_variable(model.v, v)
    print_min_max(model.u, 'u')
    print_min_max(model.v, 'v')
    
    # solve for vertical velocity :
    if config['velocity']['solve_vert_velocity']:
      self.solve_vert_velocity()
     
    # solve for pressure :
    if config['velocity']['calc_pressure']:
      self.solve_pressure()


class VelocityBP(Physics):
  """				
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING BP VELOCITY PHYSICS :::"
    print_text(s, self.color())
    
    self.model  = model
    self.config = config

    mesh          = model.mesh
    r             = config['velocity']['r']
    V             = model.Q2
    Q             = model.Q
    U             = model.U
    dU            = model.dU
    Phi           = model.Phi
    eta_shf       = model.eta_shf
    eta_gnd       = model.eta_gnd
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    rhoi          = model.rhoi
    rhow          = model.rhow
    R             = model.R
    g             = model.g
    beta          = model.beta
    w             = model.w
    N             = model.N
    D             = model.D
    
    gradS         = grad(S)
    gradB         = grad(B)
     
    # new constants :
    p0     = 101325
    T0     = 288.15
    M      = 0.0289644
    ci     = model.ci

    dx     = model.dx
    dx_s   = dx(1)
    dx_g   = dx(0)
    dx     = dx(1) + dx(0) # entire internal
    ds     = model.ds  
    dGnd   = ds(3)         # grounded bed
    dFlt   = ds(5)         # floating bed
    dSde   = ds(4)         # sides
    dBed   = dGnd + dFlt   # bed
    
    #===========================================================================
    # define variational problem :

    # horizontal velocity :
    u, v      = U
    phi, psi  = Phi
    
    # vertical velocity :
    dw        = TrialFunction(Q)
    chi       = TestFunction(Q)
    
    epi_1  = as_vector([   2*u.dx(0) + v.dx(1), 
                        0.5*(u.dx(1) + v.dx(0)),
                        0.5* u.dx(2)            ])
    epi_2  = as_vector([0.5*(u.dx(1) + v.dx(0)),
                             u.dx(0) + 2*v.dx(1),
                        0.5* v.dx(2)            ])
   
    # boundary integral terms : 
    f_w    = rhoi*g*(S - x[2]) + rhow*g*D               # lateral
    p_a    = p0 * (1 - g*x[2]/(ci*T0))**(ci*M/R)        # surface pressure
    
    #Ne       = H + rhow/rhoi * D
    #P        = -0.383
    #Q        = -0.349
    #Unorm    = sqrt(inner(U,U) + DOLFIN_EPS)
    #Coef     = 1/(beta * Ne**(q/p))
    
    # residual :
    R1 = + 2 * eta_shf * dot(epi_1, grad(phi)) * dx_s \
         + 2 * eta_shf * dot(epi_2, grad(psi)) * dx_s \
         + 2 * eta_gnd * dot(epi_1, grad(phi)) * dx_g \
         + 2 * eta_gnd * dot(epi_2, grad(psi)) * dx_g \
         + rhoi * g * gradS[0] * phi * dx \
         + rhoi * g * gradS[1] * psi * dx \
         + beta**2 * u * phi * dGnd \
         + beta**2 * v * psi * dGnd \
         + Constant(DOLFIN_EPS) * u * phi * dFlt \
         + Constant(DOLFIN_EPS) * v * psi * dFlt \
    
    if (not config['periodic_boundary_conditions']
        and not config['velocity']['use_lat_bcs']
        and config['use_pressure_boundary']):
      R1 -= f_w * (N[0]*phi + N[1]*psi) * dSde \
    
    R2 = + (u.dx(0) + v.dx(1) + dw.dx(2)) * chi * dx \
         + (u*N[0] + v*N[1] + dw*N[2]) * chi * dBed \
  
    # residuals :  
    self.R1 = R1
    self.R2 = R2
    
    # Jacobian :
    self.J = derivative(self.R1, U, dU)

    # list of boundary conditions
    self.bcs  = []
    self.bc_w = None
      
    # add lateral boundary conditions :  
    if config['velocity']['use_lat_bcs']:
      self.bcs.append(DirichletBC(V.sub(0), model.u_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(V.sub(1), model.v_lat_bc, model.ff, 4))
      self.bc_w = DirichletBC(Q, model.w_lat_bc, model.ff, 4)

    # add boundary condition for the divide :
    self.bcs.append(DirichletBC(V.sub(0), 0.0, model.ff, 7))
    self.bcs.append(DirichletBC(V.sub(1), 0.0, model.ff, 7))

    # keep the residual for adjoint solves :
    model.A     = self.R1

  def solve_pressure(self):
    """
    Solve for the BP pressure 'p'.
    """
    model  = self.model
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving BP pressure :::"
    print_text(s, self.color())
    
    Q       = model.Q
    rhoi    = model.rhoi
    g       = model.g
    S       = model.S
    x       = model.x
    w       = model.w
    p       = model.p
    eta_shf = model.eta_shf
    eta_gnd = model.eta_gnd

    p_shf   = project(rhoi*g*(S - x[2]) + 2*eta_shf*w.dx(2), Q)
    p_gnd   = project(rhoi*g*(S - x[2]) + 2*eta_gnd*w.dx(2), Q)
    
    # unify the enhancement factor over shelves and grounded ice : 
    p_v     = p.vector().array()
    p_gnd_v = p_gnd.vector().array()
    p_shf_v = p_shf.vector().array()
    p_v[self.gnd_dofs] = p_gnd_v[self.gnd_dofs]
    p_v[self.shf_dofs] = p_shf_v[self.shf_dofs]
    self.assign_variable(p, p_v)
    
    print_min_max(p, 'p')

  def solve_vert_velocity(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving BP vertical velocity :::"
    print_text(s, self.color())
    
    sm = config['velocity']['vert_solve_method']
    
    aw       = assemble(lhs(self.R2))
    Lw       = assemble(rhs(self.R2))
    if self.bc_w != None:
      self.bc_w.apply(aw, Lw)
    w_solver = LUSolver(sm)
    w_solver.solve(aw, model.w.vector(), Lw)
    #solve(lhs(self.R2) == rhs(self.R2), model.w, bcs = self.bc_w,
    #      solver_parameters = {"linear_solver" : sm})#,
    #                           "symmetric" : True})
    print_min_max(model.w, 'w')
    
  def solve(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    config = self.config
    
    # solve nonlinear system :
    params = config['velocity']['newton_params']
    rtol   = params['newton_solver']['relative_tolerance']
    maxit  = params['newton_solver']['maximum_iterations']
    alpha  = params['newton_solver']['relaxation_parameter']
    s      = "::: solving BP horizontal velocity with %i max iterations" + \
             " and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.R1 == 0, model.U, J = self.J, bcs = self.bcs, 
          solver_parameters = params)
    u, v = model.U.split(True)

    model.assign_variable(model.u, u)
    model.assign_variable(model.v, v)
    print_min_max(model.u, 'u')
    print_min_max(model.v, 'v')
    
    # solve for vertical velocity :
    if config['velocity']['solve_vert_velocity']:
      self.solve_vert_velocity()
    
    # solve for pressure :
    if config['velocity']['calc_pressure']:
      self.solve_pressure()


class VelocityBPFull(Physics):
  """				
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    s = "::: INITIALIZING FULL BP VELOCITY PHYSICS :::"
    print_text(s, self.color())
    
    self.model  = model
    self.config = config

    mesh          = model.mesh
    r             = config['velocity']['r']
    V             = model.Q3
    Q             = model.Q
    U             = model.U
    dU            = model.dU
    Phi           = model.Phi
    eta_shf       = model.eta_shf
    eta_gnd       = model.eta_gnd
    gamma         = model.gamma
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    R             = model.R
    rhoi          = model.rhoi
    rhow          = model.rhow
    g             = model.g
    beta          = model.beta
    N             = model.N
    D             = model.D
    
    gradS         = grad(S)
    gradB         = grad(B)
     
    # new constants :
    p0     = 101325
    T0     = 288.15
    M      = 0.0289644
    ci     = model.ci

    dx     = model.dx
    dx_s   = dx(1)
    dx_g   = dx(0)
    dx     = dx(1) + dx(0) # entire internal
    ds     = model.ds  
    dGnd   = ds(3)         # grounded bed
    dFlt   = ds(5)         # floating bed
    dSde   = ds(4)         # sides
    dBed   = dGnd + dFlt   # bed
    dSrf   = ds(2) + ds(6) # surface
    
    #===========================================================================
    # define variational problem :

    # horizontal velocity :
    u, v, w        = U
    phi, psi, chi  = Phi
    
    epi_1  = as_vector([     u.dx(0) - w.dx(2), 
                        0.5*(u.dx(1) + v.dx(0)),
                        0.5*(u.dx(2) + w.dx(0)) ])
    epi_2  = as_vector([0.5*(v.dx(0) + u.dx(1)),
                             v.dx(1) - w.dx(2),
                        0.5*(v.dx(2) + w.dx(1)) ])
    epi_3  = as_vector([0.5*(w.dx(0) + u.dx(2)),
                        0.5*(w.dx(1) + v.dx(2)),
                        0.0                     ])
   
    # boundary integral terms : 
    f_w    = rhoi*g*(S - x[2]) + rhow*g*D               # lateral
    p_a    = p0 * (1 - g*x[2]/(ci*T0))**(ci*M/R)        # surface pressure
    
    #Ne     = H - rhow/rhoi * D
    #lnC    = ln(-0.383)
    #Ce     = ln(beta**2 * Ne**(-0.349)) / lnC
    #Sl_gnd = (ln(1/u)/lnC + Ce)*u + (ln(1/v)/lnC + Ce)*v + (ln(1/w)/lnC + Ce)*w
    
    # residual :
    R1 = + 2 * eta_shf * dot(epi_1, grad(phi)) * dx_s \
         + 2 * eta_shf * dot(epi_2, grad(psi)) * dx_s \
         + 2 * eta_gnd * dot(epi_1, grad(phi)) * dx_g \
         + 2 * eta_gnd * dot(epi_2, grad(psi)) * dx_g \
         + rhoi * g * gradS[0] * phi * dx \
         + rhoi * g * gradS[1] * psi * dx \
         + beta**2 * u * phi * dGnd \
         + beta**2 * v * psi * dGnd \
         #+ 2 * eta_shf * dot(epi_3, grad(chi)) * dx_s \
         #+ 2 * eta_gnd * dot(epi_3, grad(chi)) * dx_g \
         #+ rhoi * g * dot(gradS, Phi) * dx \
         #+ beta**2 * dot(U, Phi) * dGnd \
    
    if (not config['periodic_boundary_conditions']
        and not config['velocity']['use_lat_bcs']
        and config['use_pressure_boundary']):
      R1 -= f_w * dot(N, Phi) * dSde \
    
    R2 = div(U)*chi*dx + dot(U, N)*chi*dBed
    #R2 = -dot(U, grad(chi))*dx + dot(U, N)*chi*dSrf
     
    # residual :
    self.A = R1 + R2
    
    # Jacobian :
    self.J = derivative(self.A, U, dU)

    # list of boundary conditions
    self.bcs  = []
      
    # add lateral boundary conditions :  
    if config['velocity']['use_lat_bcs']:
      self.bcs.append(DirichletBC(V.sub(0), model.u_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(V.sub(1), model.v_lat_bc, model.ff, 4))
      self.bcs.append(DirichletBC(V.sub(2), model.w_lat_bc, model.ff, 4))

    # add boundary condition for the divide :
    self.bcs.append(DirichletBC(V.sub(0), 0.0, model.ff, 7))
    self.bcs.append(DirichletBC(V.sub(1), 0.0, model.ff, 7))

    # keep the residual for adjoint solves :
    model.A       = self.A

  def solve_pressure(self):
    """
    Solve for the BP pressure 'p'.
    """
    model  = self.model
    config = self.config
    
    # solve for vertical velocity :
    s  = "::: solving full BP pressure :::"
    print_text(s, self.color())
    
    Q       = model.Q
    rhoi    = model.rhoi
    g       = model.g
    S       = model.S
    x       = model.x
    w       = model.w
    p       = model.p
    eta_shf = model.eta_shf
    eta_gnd = model.eta_gnd

    p_shf   = project(rhoi*g*(S - x[2]) + 2*eta_shf*w.dx(2), Q)
    p_gnd   = project(rhoi*g*(S - x[2]) + 2*eta_gnd*w.dx(2), Q)
    
    # unify the enhancement factor over shelves and grounded ice : 
    p_v     = p.vector().array()
    p_gnd_v = p_gnd.vector().array()
    p_shf_v = p_shf.vector().array()
    p_v[self.gnd_dofs] = p_gnd_v[self.gnd_dofs]
    p_v[self.shf_dofs] = p_shf_v[self.shf_dofs]
    self.assign_variable(p, p_v)
    
    print_min_max(p, 'p')

  def solve(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    config = self.config
    
    # solve nonlinear system :
    params = config['velocity']['newton_params']
    rtol   = params['newton_solver']['relative_tolerance']
    maxit  = params['newton_solver']['maximum_iterations']
    alpha  = params['newton_solver']['relaxation_parameter']
    s      = "::: solving full BP velocity with %i max " + \
             "iterations and step size = %.1f :::"
    print_text(s % (maxit, alpha), self.color())
    
    # compute solution :
    solve(self.A == 0, model.U, J = self.J, bcs = self.bcs, 
          solver_parameters = params)
    u, v, w = model.U.split(True)

    model.assign_variable(model.u, u)
    model.assign_variable(model.v, v)
    model.assign_variable(model.w, w)
    print_min_max(model.u, 'u')
    print_min_max(model.v, 'v')
    print_min_max(model.w, 'w')
    
    # solve for pressure :
    if config['velocity']['calc_pressure']:
      self.solve_pressure()
    

class Enthalpy(Physics):
  """ 
  """
  def __init__(self, model, config):
    """ 
    Set up equation, memory allocation, etc. 
    """
    s    = "::: INITIALIZING ENTHALPY PHYSICS :::"
    print_text(s, self.color())

    self.config = config
    self.model  = model

    r           = config['velocity']['r']
    mesh        = model.mesh
    V           = model.V
    Q           = model.Q
    theta       = model.theta
    theta0      = model.theta0
    n           = model.n
    eta_gnd     = model.eta_gnd
    eta_shf     = model.eta_shf
    T           = model.T
    T_melt      = model.T_melt
    Mb          = model.Mb
    L           = model.L
    ci          = model.ci
    cw          = model.cw
    T_w         = model.T_w
    gamma       = model.gamma
    S           = model.S
    B           = model.B
    H           = S - B
    x           = model.x
    W           = model.W
    R           = model.R
    epsdot      = model.epsdot
    eps_reg     = model.eps_reg
    rhoi        = model.rhoi
    rhow        = model.rhow
    g           = model.g
    beta        = model.beta
    u           = model.u
    v           = model.v
    w           = model.w
    kappa       = model.kappa
    Kcoef       = model.Kcoef
    ki          = model.ki
    kw          = model.kw
    T_surface   = model.T_surface
    theta_surface = model.theta_surface
    theta_float   = model.theta_float
    q_geo         = model.q_geo
    thetahat      = model.thetahat
    uhat        = model.uhat
    vhat        = model.vhat
    what        = model.what
    mhat        = model.mhat
    spy         = model.spy
    h           = model.h
    ds          = model.ds
    dSrf        = ds(2)         # surface
    dGnd        = ds(3)         # grounded bed
    dFlt        = ds(5)         # floating bed
    dSde        = ds(4)         # sides
    dBed        = dGnd + dFlt   # bed
    dx          = model.dx
    dx_s        = dx(1)
    dx_g        = dx(0)
    dx          = dx(1) + dx(0) # entire internal
    
    ## second invariant of the strain-rate tensor squared :
    #term   = + 0.5*(   (u.dx(1) + v.dx(0))**2  \
    #                 + (u.dx(2) + w.dx(0))**2  \
    #                 + (v.dx(2) + w.dx(1))**2) \
    #         + u.dx(0)**2 + v.dx(1)**2 + w.dx(2)**2 
    #epsdot = 0.5 * term + eps_reg
    
    # initialize the conductivity coefficient for entirely cold ice :
    model.assign_variable(Kcoef, 1.0)

    # Define test and trial functions       
    psi    = TestFunction(Q)
    dtheta = TrialFunction(Q)

    # Pressure melting point
    s    = "::: calculating pressure-melting temperature :::"
    print_text(s, self.color())
    model.assign_variable(T_melt, project(T_w - gamma * (S - x[2]), Q))
    print_min_max(T_melt, 'T_melt')
   
    # Surface boundary condition
    model.assign_variable(theta_surface, project(T_surface * ci))
    model.assign_variable(theta_float,   project(ci*T_melt))

    # For the following heat sources, note that they differ from the 
    # oft-published expressions, in that they are both multiplied by constants.
    # I think that this is the correct form, as they must be this way in order 
    # to conserve energy.  This also implies that heretofore, models have been 
    # overestimating frictional heat, and underestimating strain heat.

    # Frictional heating :
    #q_friction = 0.5 * beta**2 * H**r * (u**2 + v**2)
    q_friction = 0.5 * beta**2 * H**r * (u**2 + v**2 + w**2)

    # Strain heating = stress*strain
    Q_s_gnd = model.Vd_gnd
    Q_s_shf = model.Vd_shf

    # thermal conductivity (Greve and Blatter 2009) :
    ki    =  9.828 * exp(-0.0057*T)
    
    # bulk properties :
    k     =  (1 - W)*ki   + W*kw     # bulk thermal conductivity
    c     =  (1 - W)*ci   + W*cw     # bulk heat capacity
    rho   =  (1 - W)*rhoi + W*rhow   # bulk density
    kappa =  k / (rho*c)             # bulk thermal diffusivity

    # configure the module to run in steady state :
    if config['mode'] == 'steady':
      U       = as_vector([u, v, w])
      epi     = model.strain_rate_tensor(U)
      ep_xx   = epi[0,0]
      ep_yy   = epi[1,1]
      ep_zz   = epi[2,2]
      ep_xy   = epi[0,1]
      ep_xz   = epi[0,2]
      ep_yz   = epi[1,2]
      epsdot  = 0.5 * (+ ep_xx**2 + ep_yy**2 + ep_zz**2) \
                       + ep_xy**2 + ep_xz**2 + ep_yz**2
      #Q_s_gnd = 2 * eta_gnd * tr(dot(epi,epi))
      #Q_s_shf = 2 * eta_shf * tr(dot(epi,epi))
      #Q_s_gnd = 4 * eta_gnd * epsdot
      #Q_s_shf = 4 * eta_shf * epsdot

      # skewed test function in areas with high velocity :
      Unorm  = sqrt(dot(U, U) + DOLFIN_EPS)
      PE     = Unorm*h/(2*kappa)
      tau    = 1/tanh(PE) - 1/PE
      #T_c    = conditional( lt(Unorm, 4), 0.0, 1.0 )
      psihat = psi + h*tau/(2*Unorm) * dot(U, grad(psi))

      # residual of model :
      self.a = + rho * dot(U, grad(dtheta)) * psihat * dx \
               + rho * spy * kappa * dot(grad(psi), grad(dtheta)) * dx \
      
      self.L = + (q_geo + q_friction) * psihat * dGnd \
               + Q_s_gnd * psihat * dx_g \
               + Q_s_shf * psihat * dx_s
      
    # configure the module to run in transient mode :
    elif config['mode'] == 'transient':
      dt = config['time_step']
    
      U       = as_vector([uhat, vhat, what - mhat])
      epi     = model.strain_rate_tensor(U)
      ep_xx   = epi[0,0]
      ep_yy   = epi[1,1]
      ep_zz   = epi[2,2]
      ep_xy   = epi[0,1]
      ep_xz   = epi[0,2]
      ep_yz   = epi[1,2]
      epsdot  = 0.5 * (+ ep_xx**2 + ep_yy**2 + ep_zz**2) \
                       + ep_xy**2 + ep_xz**2 + ep_yz**2
      #Q_s_gnd = 2 * eta_gnd * tr(dot(epi,epi))
      #Q_s_shf = 2 * eta_shf * tr(dot(epi,epi))
      Q_s_gnd = 4 * eta_gnd * epsdot
      Q_s_shf = 4 * eta_shf * epsdot

      # Skewed test function.  Note that vertical velocity has 
      # the mesh velocity subtracted from it.
      Unorm  = sqrt(dot(U, U) + 1.0)
      PE     = Unorm*h/(2*kappa)
      tau    = 1/tanh(PE) - 1/PE
      #T_c    = conditional( lt(Unorm, 4), 0.0, 1.0 )
      psihat = psi + h*tau/(2*Unorm) * dot(U, grad(psi))

      nu = 0.5
      # Crank Nicholson method
      thetamid = nu*dtheta + (1 - nu)*theta0
      
      # implicit system (linearized) for enthalpy at time theta_{n+1}
      self.a = + rho * (dtheta - theta0) / dt * psi * dx \
               + rho * dot(U, grad(thetamid)) * psihat * dx \
               + rho * spy * kappa * dot(grad(psi), grad(thetamid)) * dx \
      
      self.L = + (q_geo + q_friction) * psi * dGnd \
               + Q_s_gnd * psihat * dx_g \
               + Q_s_shf * psihat * dx_s
    
    # surface boundary condition : 
    self.bc_theta = []
    self.bc_theta.append( DirichletBC(Q, theta_surface, model.ff, 2) )
    self.bc_theta.append( DirichletBC(Q, theta_surface, model.ff, 6) )
    
    # apply T_w conditions of portion of ice in contact with water :
    self.bc_theta.append( DirichletBC(Q, theta_float,   model.ff, 5) )
    
    # apply lateral boundaries if desired : 
    if config['enthalpy']['lateral_boundaries'] == 'surface':
      self.bc_theta.append( DirichletBC(Q, theta_surface, model.ff, 4) )

    self.c          = c
    self.k          = k
    self.rho        = rho
    self.kappa      = kappa
    self.q_friction = q_friction
    self.dBed       = dBed
     
  
  def solve(self, theta0=None, thetahat=None, uhat=None, 
            vhat=None, what=None, mhat=None):
    """ 
    """
    model  = self.model
    config = self.config

    # Assign values for H0,u,w, and mesh velocity
    if theta0 is not None:
      model.assign_variable(model.theta0,   theta0)
      model.assign_variable(model.thetahat, thetahat)
      model.assign_variable(model.uhat, uhat)
      model.assign_variable(model.vhat, vhat)
      model.assign_variable(model.what, what)
      model.assign_variable(model.mhat, mhat)
      
    mesh       = model.mesh
    V          = model.V
    Q          = model.Q
    T_melt     = model.T_melt
    theta      = model.theta
    T          = model.T
    Mb         = model.Mb
    W          = model.W
    W0         = model.W0
    L          = model.L
    Kcoef      = model.Kcoef
    q_geo      = model.q_geo
    B          = model.B
    ci         = model.ci
    rhoi       = model.rhoi
    dBed       = self.dBed
    q_friction = self.q_friction
    rho        = self.rho
    kappa      = self.kappa
    
    # solve the linear equation for enthalpy :
    s    = "::: solving enthalpy :::"
    print_text(s, self.color())
    sm = config['enthalpy']['solve_method']
    aw        = assemble(self.a)
    Lw        = assemble(self.L)
    for bc in self.bc_theta:
      bc.apply(aw, Lw)
    theta_solver = LUSolver(sm)
    theta_solver.solve(aw, theta.vector(), Lw)
    #solve(self.a == self.L, theta, self.bc_theta,
    #      solver_parameters = {"linear_solver" : sm})
    print_min_max(theta, 'theta')

    # temperature solved diagnostically : 
    s = "::: calculating temperature :::"
    print_text(s, self.color())
    T_n  = project(theta/ci, Q)
    
    # update temperature for wet/dry areas :
    T_n_v        = T_n.vector().array()
    T_melt_v     = T_melt.vector().array()
    warm         = T_n_v >= T_melt_v
    cold         = T_n_v <  T_melt_v
    T_n_v[warm]  = T_melt_v[warm]
    model.assign_variable(T, T_n_v)
    print_min_max(T,  'T')
    
    # update kappa coefficient for wet/dry areas :
    #Kcoef_v       = Kcoef.vector().array()
    #Kcoef_v[warm] = 1.0/10.0              # wet ice
    #Kcoef_v[cold] = 1.0                   # cold ice
    #model.assign_variable(Kcoef, Kcoef_v)

    # water content solved diagnostically :
    s = "::: calculating water content :::"
    print_text(s, self.color())
    W_n  = project((theta - ci*T_melt)/L, Q)
    
    # update water content :
    W_v             = W_n.vector().array()
    W_v[cold]       = 0.0
    W_v[W_v < 0.0]  = 0.0
    W_v[W_v > 0.01] = 0.01  # for rheology; instant water run-off
    model.assign_variable(W0, W)
    model.assign_variable(W,  W_v)
    print_min_max(W,  'W')
    
    # calculate melt-rate : 
    s = "::: calculating basal melt-rate :::"
    print_text(s, self.color())
    B_mag = sqrt(inner(grad(B), grad(B)) + DOLFIN_EPS)
    dHdn  = rho * kappa * dot(grad(theta), grad(B) / B_mag)
    nMb   = project((q_geo + q_friction - dHdn) / (L*rhoi))
    nMb_v = nMb.vector().array()
    #nMb_v[nMb_v < 0.0]  = 0.0
    #nMb_v[nMb_v > 10.0] = 10.0
    model.assign_variable(Mb,  nMb_v)
    print_min_max(Mb, 'Mb')

    # calculate bulk density :
    s = "::: calculating bulk density :::"
    print_text(s, self.color())
    rho       = project(self.rho)
    model.rho = rho
    print_min_max(rho,'rho')


class EnthalpyDG(Physics):
  r""" 
  """
  def __init__(self, model, config):
    """ 
    Set up equation, memory allocation, etc. 
    """
    s    = "::: INITIALIZING DG ENTHALPY PHYSICS :::"
    print_text(s, self.color())

    self.config = config
    self.model  = model

    r           = config['velocity']['r']
    mesh        = model.mesh
    Q           = model.Q
    DQ          = model.DQ
    theta       = model.theta
    theta0      = model.theta0
    n           = model.n
    b_gnd       = model.b_gnd
    b_gnd       = model.b_gnd
    b_shf       = model.b_shf
    T           = model.T
    T_melt          = model.T_melt
    Mb          = model.Mb
    L           = model.L
    ci          = model.ci
    cw          = model.cw
    T_w         = model.T_w
    gamma       = model.gamma
    S           = model.S
    B           = model.B
    x           = model.x
    W           = model.W
    R           = model.R
    epsdot      = model.epsdot
    eps_reg     = model.eps_reg
    rhoi        = model.rhoi
    rhow        = model.rhow
    g           = model.g
    beta        = model.beta
    u           = model.u
    v           = model.v
    w           = model.w
    kappa       = model.kappa
    Kcoef       = model.Kcoef
    ki          = model.ki
    kw          = model.kw
    T_surface   = model.T_surface
    theta_surface   = model.theta_surface
    theta_float     = model.theta_float
    q_geo       = model.q_geo
    thetahat        = model.thetahat
    uhat        = model.uhat
    vhat        = model.vhat
    what        = model.what
    mhat        = model.mhat
    spy         = model.spy
    h           = model.h
    N           = model.N
    ds          = model.ds
    dSrf        = ds(2)         # surface
    dGnd        = ds(3)         # grounded bed
    dFlt        = ds(5)         # floating bed
    dSde        = ds(4)         # sides
    dBed        = dGnd + dFlt   # bed
    dGamma      = ds(2) + ds(3) + ds(5) + ds(4)
    dx          = model.dx
    dx_s        = dx(1)
    dx_g        = dx(0)
    dx          = dx(1) + dx(0) # entire internal
    
    # second invariant of the strain-rate tensor squared :
    term   = + 0.5*(   (u.dx(1) + v.dx(0))**2  \
                     + (u.dx(2) + w.dx(0))**2  \
                     + (v.dx(2) + w.dx(1))**2) \
             + u.dx(0)**2 + v.dx(1)**2 + w.dx(2)**2 
    epsdot = 0.5 * term + eps_reg
    
    # Define test and trial functions       
    psi     = TestFunction(DQ)
    dtheta  = TrialFunction(DQ)

    # Pressure melting point
    T_melt = Function(DQ)
    model.assign_variable(T_melt, project(T_w - gamma * (S - x[2]), DQ))
   
    # Surface boundary condition
    theta_surface = Function(DQ)
    theta_float   = Function(DQ)
    model.assign_variable(theta_surface, project(T_surface * ci, DQ))
    model.assign_variable(theta_float,   project(ci*T_melt, DQ))

    # For the following heat sources, note that they differ from the 
    # oft-published expressions, in that they are both multiplied by constants.
    # I think that this is the correct form, as they must be this way in order 
    # to conserve energy.  This also implies that heretofore, models have been 
    # overestimating frictional heat, and underestimating strain heat.

    # Frictional heating :
    q_friction = 0.5 * beta**2 * (S - B)**r * (u**2 + v**2)

    # Strain heating = stress*strain
    Q_s_gnd = (2*n)/(n+1) * b_gnd * epsdot**((n+1)/(2*n))
    Q_s_shf = (2*n)/(n+1) * b_shf * epsdot**((n+1)/(2*n))

    # thermal conductivity (Greve and Blatter 2009) :
    ki    =  9.828 * exp(-0.0057*T)
    
    # bulk properties :
    k     =  (1 - W)*ki   + W*kw     # bulk thermal conductivity
    c     =  (1 - W)*ci   + W*cw     # bulk heat capacity
    rho   =  (1 - W)*rhoi + W*rhow   # bulk density
    kappa =  k / (rho*c)             # bulk thermal diffusivity

    # configure the module to run in steady state :
    if config['mode'] == 'steady':
      try:
        U    = as_vector([u, v, w])
      except NameError:
        print "No velocity field found.  Defaulting to no velocity"
        U    = 0.0

      h_avg  = (h('+') + h('-'))/2.0
      un     = (dot(U, N) + abs(dot(U, N)))/2.0
      alpha  = Constant(5.0)

      # residual of model :
      a_int  = rho * dot(grad(psi), spy * kappa*grad(dtheta) - U*dtheta)*dx
             
      a_fac  = + rho * spy * kappa * (alpha/h_avg)*jump(psi)*jump(dtheta) * dS \
               - rho * spy * kappa * dot(avg(grad(psi)), jump(dtheta, N)) * dS \
               - rho * spy * kappa * dot(jump(psi, N), avg(grad(dtheta))) * dS
                 
      a_vel  = jump(psi)*jump(un*dtheta)*dS  + psi*un*dtheta*dGamma
      
      self.a = a_int + a_fac + a_vel

      #self.a = + rho * dot(U, grad(dtheta)) * psihat * dx \
      #         + rho * spy * kappa * dot(grad(psi), grad(dtheta)) * dx \
      
      self.L = + (q_geo + q_friction) * psi * dGnd \
               + Q_s_gnd * psi * dx_g \
               + Q_s_shf * psi * dx_s
      

    # configure the module to run in transient mode :
    elif config['mode'] == 'transient':
      dt = config['time_step']
    
      # Skewed test function.  Note that vertical velocity has 
      # the mesh velocity subtracted from it.
      U = as_vector([uhat, vhat, what - mhat])

      Unorm  = sqrt(dot(U, U) + 1.0)
      PE     = Unorm*h/(2*kappa)
      tau    = 1/tanh(PE) - 1/PE
      T_c    = conditional( lt(Unorm, 4), 0.0, 1.0 )
      psihat = psi + T_c*h*tau/(2*Unorm) * dot(U, grad(psi))

      nu = 0.5
      # Crank Nicholson method
      thetamid = nu*dtheta + (1 - nu)*theta0
      
      # implicit system (linearized) for enthalpy at time theta_{n+1}
      self.a = + rho * (dtheta - theta0) / dt * psi * dx \
               + rho * dot(U, grad(thetamid)) * psihat * dx \
               + rho * spy * kappa * dot(grad(psi), grad(thetamid)) * dx \
      
      self.L = + (q_geo + q_friction) * psi * dGnd \
               + Q_s_gnd * psihat * dx_g \
               + Q_s_shf * psihat * dx_s

    model.theta         = Function(DQ)
    model.T_surface = T_surface
    model.q_geo     = q_geo
    model.T_melt        = T_melt
    model.theta_surface = theta_surface
    model.theta_float   = theta_float
    
    self.c          = c
    self.k          = k
    self.rho        = rho
    self.kappa      = kappa
    self.q_friction = q_friction
    self.dBed       = dBed
     
  
  def solve(self, theta0=None, thetahat=None, uhat=None, 
            vhat=None, what=None, mhat=None):
    r""" 
    """
    model  = self.model
    config = self.config
    
    # Assign values for theta0,u,w, and mesh velocity
    if theta0 is not None:
      model.assign_variable(model.theta0,   theta0)
      model.assign_variable(model.thetahat, thetahat)
      model.assign_variable(model.uhat, uhat)
      model.assign_variable(model.vhat, vhat)
      model.assign_variable(model.what, what)
      model.assign_variable(model.mhat, mhat)
      
    lat_bc     = config['enthalpy']['lateral_boundaries']
    mesh       = model.mesh
    V          = model.V
    Q          = model.Q
    DQ         = model.DQ
    T_melt         = model.T_melt
    theta          = model.theta
    theta_surface  = model.theta_surface
    theta_float    = model.theta_float  
    T          = model.T
    Mb         = model.Mb
    W          = model.W
    W0         = model.W0
    W_r        = model.W_r
    L          = model.L
    Kcoef      = model.Kcoef
    q_geo      = model.q_geo
    B          = model.B
    ci         = model.ci
    rhoi       = model.rhoi
    dBed       = self.dBed
    q_friction = self.q_friction
    rho        = self.rho
    kappa      = self.kappa

    # surface boundary condition : 
    self.bc_theta = []
    self.bc_theta.append( DirichletBC(DQ, theta_surface, model.ff, 2) )
    
    # apply T_w conditions of portion of ice in contact with water :
    self.bc_theta.append( DirichletBC(DQ, theta_float,   model.ff, 5) )
    self.bc_theta.append( DirichletBC(DQ, theta_surface, model.ff, 6) )
    
    # apply lateral boundaries if desired : 
    if config['enthalpy']['lateral_boundaries'] is not None:
      self.bc_theta.append( DirichletBC(DQ, lat_bc, model.ff, 4) )
    
    # solve the linear equation for enthalpy :
    if self.model.MPI_rank==0:
      s    = "::: solving DG internal energy :::"
      text = colored(s, 'cyan')
      print text
    sm = config['enthalpy']['solve_method']
    solve(self.a == self.L, theta, self.bc_theta, 
          solver_parameters = {"linear_solver" : sm})

    # calculate temperature and water content :
    s = "::: calculating temperature, water content, and basal melt-rate :::"
    print_text(s, self.color())
    
    # temperature solved diagnostically : 
    T_n  = project(theta/ci, Q)
    
    # update temperature for wet/dry areas :
    T_n_v        = T_n.vector().array()
    T_melt_v         = T_melt.vector().array()
    warm         = T_n_v >= T_melt_v
    cold         = T_n_v <  T_melt_v
    T_n_v[warm]  = T_melt_v[warm]
    model.assign_variable(T, T_n_v)
    
    # water content solved diagnostically :
    W_n  = project((theta - ci*T_melt)/L, Q)
    
    # update water content :
    W_v        = W_n.vector().array()
    W_v[cold]  = 0.0
    model.assign_variable(W0, W)
    model.assign_variable(W,  W_v)
    
    # update capped variable for rheology : 
    W_v[W_v > 0.01] = 0.01
    model.assign_variable(W_r, W_v)
    
    # calculate melt-rate : 
    nMb   = project((q_geo + q_friction) / (L*rhoi))
    model.assign_variable(Mb,  nMb)

    # calculate bulk density :
    rho       = project(self.rho)
    model.rho = rho
    
    # print the min/max values to the screen :    
    print_min_max(theta,  'theta')
    print_min_max(T,  'T')
    print_min_max(Mb, 'Mb')
    print_min_max(W,  'W')
    print_min_max(rho,'rho')


class FreeSurface(Physics):
  r""" 
  Class for evolving the free surface of the ice through time.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
                 	attributes such as velocties, age, and surface climate

  **Stabilization** 

  The free surface equation is hyperbolic so, a modified Galerkin test 
  function is used for stabilization.
  
  :Equation:
     .. math::
      \hat{\phi} = \phi + \frac{\alpha h}{2}\frac{u_i u_j}{| |u| |}
      \cdot\nabla_{| |}\phi
     
  A shock-capturing artificial viscosity is applied in order to smooth the 
  sharp discontinuities that occur at the ice boundaries where the model
  domain switches from ice to ice-free regimes.  The additional term is
  given by
  
  :Equation:
     .. math::
      D_{shock} = \nabla \cdot C \nabla S

  +-----------------------------------+-----------------------------------+
  |Term                               |Description                        |
  +===================================+===================================+
  |.. math::                          |Nonlinear residual-dependent scalar|
  |   C = \frac{h}{2| |u| |}\left[    |                                   |
  |   \nabla_{| |}S\cdot\nabla        |                                   |
  |   _{| |}S\right]^{-1}\mathcal{R}  |                                   |
  |   ^{2}                            |                                   |
  +-----------------------------------+-----------------------------------+
  |.. math::                          |Residual of the original free      |
  |   \mathcal{R}                     |surface equation                   |
  +-----------------------------------+-----------------------------------+

  For the Stokes' equations to remain stable, it is necessary to either
  satisfy or circumvent the Ladyzehnskaya-Babuska-Brezzi (LBB) condition.
  We circumvent this condition by using a Galerkin-least squares (GLS)
  formulation of the Stokes' functional:
    
  :Equation:
     .. math::
      \mathcal{A}'\left[\textbf{u},P\right] = \mathcal{A} - \int
      \limits_\Omega\tau_{gls}\left(\nabla P - \rho g\right)\cdot
      \left(\nabla P - \rho g\right)d\Omega
      
  +----------------------------------------+------------------------------+
  |Term                                    |Description                   |
  +========================================+==============================+
  |.. math::                               |Variational principle for     |
  |   \mathcal{A}                          |power law rheology            |
  +----------------------------------------+------------------------------+
  |.. math::                               |Pressure                      |
  |   P                                    |                              |
  +----------------------------------------+------------------------------+
  |.. math::                               |Ice density                   |
  |   \rho                                 |                              |
  +----------------------------------------+------------------------------+
  |.. math::                               |Force of gravity              |
  |   g                                    |                              |
  +----------------------------------------+------------------------------+
  |.. math::                               |Stabilization parameter. Since|
  |   \tau_{gls} = \frac{h^2}              |it is a function of ice       |
  |   {12\rho b(T)}                        |viscosity, the stabilization  |
  |                                        |parameter is nonlinear        |
  +----------------------------------------+------------------------------+
  |.. math::                               |Temperature dependent rate    |
  |   b(T)                                 |factor                        |
  +----------------------------------------+------------------------------+
  """

  def __init__(self, model, config):
    """
    """
    s    = "::: INITIALIZING FREE-SURFACE PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    Q_flat = model.Q_flat
    Q      = model.Q

    phi    = TestFunction(Q_flat)
    dS     = TrialFunction(Q_flat)

    self.Shat   = model.Shat           # surface elevation velocity 
    self.ahat   = model.ahat           # accumulation velocity
    self.uhat   = model.uhat_f         # horizontal velocity
    self.vhat   = model.vhat_f         # horizontal velocity perp. to uhat
    self.what   = model.what_f         # vertical velocity
    mhat        = model.mhat           # mesh velocity
    dSdt        = model.dSdt           # surface height change
    M           = model.M
    ds          = model.ds_flat
    dSurf       = ds(2)
    dBase       = ds(3)
    
    self.static_boundary = DirichletBC(Q, 0.0, model.ff_flat, 4)
    h = CellSize(model.flat_mesh)

    # upwinded trial function :
    unorm       = sqrt(self.uhat**2 + self.vhat**2 + 1e-1)
    upwind_term = h/(2.*unorm)*(self.uhat*phi.dx(0) + self.vhat*phi.dx(1))
    phihat      = phi + upwind_term

    mass_matrix = dS * phihat * dSurf
    lumped_mass = phi * dSurf

    stiffness_matrix = - self.uhat * self.Shat.dx(0) * phihat * dSurf \
                       - self.vhat * self.Shat.dx(1) * phihat * dSurf\
                       + (self.what + self.ahat) * phihat * dSurf
    
    # Calculate the nonlinear residual dependent scalar
    term1            = self.Shat.dx(0)**2 + self.Shat.dx(1)**2 + 1e-1
    term2            = + self.uhat*self.Shat.dx(0) \
                       + self.vhat*self.Shat.dx(1) \
                       - (self.what + self.ahat)
    C                = 10.0*h/(2*unorm) * term1 * term2**2
    diffusion_matrix = C * dot(grad(phi), grad(self.Shat)) * dSurf
    
    # Set up the Galerkin-least squares formulation of the Stokes' functional
    A_pro         = - phi.dx(2)*dS*dx - dS*phi*dBase + dSdt*phi*dSurf 
    M.vector()[:] = 1.0
    self.M        = M*dx

    self.newz                   = Function(model.Q)
    self.mass_matrix            = mass_matrix
    self.stiffness_matrix       = stiffness_matrix
    self.diffusion_matrix       = diffusion_matrix
    self.lumped_mass            = lumped_mass
    self.A_pro                  = A_pro
    
  def solve(self):
    """
    :param uhat : Horizontal velocity
    :param vhat : Horizontal velocity perpendicular to :attr:`uhat`
    :param what : Vertical velocity 
    :param Shat : Surface elevation velocity
    :param ahat : Accumulation velocity

    """
    model  = self.model
    config = self.config
   
    model.assign_variable(self.Shat, model.S) 
    model.assign_variable(self.ahat, model.adot) 
    model.assign_variable(self.uhat, model.u) 
    model.assign_variable(self.vhat, model.v) 
    model.assign_variable(self.what, model.w) 

    m = assemble(self.mass_matrix,      keep_diagonal=True)
    r = assemble(self.stiffness_matrix, keep_diagonal=True)

    s    = "::: solving free-surface :::"
    print_text(s, self.color())
    if config['free_surface']['lump_mass_matrix']:
      m_l = assemble(self.lumped_mass)
      m_l = m_l.get_local()
      m_l[m_l==0.0]=1.0
      m_l_inv = 1./m_l

    if config['free_surface']['static_boundary_conditions']:
      self.static_boundary.apply(m,r)

    if config['free_surface']['use_shock_capturing']:
      k = assemble(self.diffusion_matrix)
      r -= k
      print_min_max(r, 'D')

    if config['free_surface']['lump_mass_matrix']:
      model.assign_variable(model.dSdt, m_l_inv * r.get_local())
    else:
      m.ident_zeros()
      solve(m, model.dSdt.vector(), r)

    A = assemble(lhs(self.A_pro))
    p = assemble(rhs(self.A_pro))
    q = Vector()  
    solve(A, q, p)
    model.assign_variable(model.dSdt, q)


class AdjointDukowiczVelocity(Physics):
  """ 
  Complete adjoint of the BP momentum balance.  Now updated to calculate
  the adjoint model and gradient using automatic differentiation.  Changing
  the form of the objective function and the differentiation variables now
  automatically propgates through the machinery.  This means that switching
  to topography optimization, or minimization of dHdt is now straightforward,
  and requires no math.
    
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
                  attributes such as velocties, age, and surface climate
  """
  def __init__(self, model, config):
    """ 
    Setup.
    """
    s   = "::: INITIALIZING DUKOWICZ ADJOINT VELOCITY PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    u_ob     = model.u_ob
    v_ob     = model.v_ob
    adot     = model.adot
    ds       = model.ds
    S        = model.S
    A        = model.A
    U        = model.U
    N        = model.N

    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    dSrf_s   = ds(6)         # surface
    dSrf_g   = ds(2)         # surface
    dGnd     = ds(3)         # grounded bed 
    dFlt     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dFlt   # bed

    if config['adjoint']['surface_integral'] == 'shelves':
      dSrf     = ds(6)
      s   = "    - integrating over shelves -"
    elif config['adjoint']['surface_integral'] == 'grounded':
      dSrf     = ds(2)
      s   = "    - integrating over grounded ice -"
    
    print_text(s, self.color())

    control = config['adjoint']['control_variable']
    alpha   = config['adjoint']['alpha']

    Q_adj   = U.function_space()

    # Objective function; least squares over the surface.
    if config['adjoint']['objective_function'] == 'log':
      self.I = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 1.0) \
                         / (sqrt(u_ob**2 + v_ob**2) + 1.0))**2 * dSrf
      s   = "    - using log objective function -"
    
    elif config['adjoint']['objective_function'] == 'kinematic':
      self.I = 0.5 * (+ U[0]*S.dx(0) + U[1]*S.dx(1) \
                      - (U[2] + adot))**2 * dSrf
      s   = "    - using kinematic objective function -"

    elif config['adjoint']['objective_function'] == 'linear':
      self.I = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dSrf
      s   = "    - using linear objective function -"
    
    elif config['adjoint']['objective_function'] == 'log_lin_hybrid':
      g1      = config['adjoint']['gamma1']
      g2      = config['adjoint']['gamma2']
      self.I1 = + g1 * 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dSrf
      self.I2 = + g2 * 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 1.0) \
                                 / (sqrt(u_ob**2 + v_ob**2) + 1.0))**2 * dSrf
      self.I  = self.I1 + self.I2
      s   = "    - using log/linear hybrid objective -"

    else:
      s = "    - WARNING: adjoint objection function may be 'linear', " + \
          "'log', 'kinematic', or 'log_lin_hybrid'.  Defaulting to 'log' -"
      print_text(s, 'red', 1)
      self.I = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 1.0) \
                         / (sqrt(u_ob**2 + v_ob**2) + 1.0))**2 * dSrf
      s   = "    - using log objective function -"
    print_text(s, self.color())
    
    # form regularization term 'R' :
    for a,c in zip(alpha,control):
      if isinstance(a, int) or isinstance(a, float):
        s   = "    - regularization parameter alpha = %.2E -" % a
        print_text(s, self.color())
      if a == 0:
        s = "    - using no regularization -"
        self.R = Constant(DOLFIN_EPS) * dGnd
      else:
        if config['adjoint']['regularization_type'] == 'TV':
          s      = "    - using total variation regularization -"
          self.R = a/2.0 * sqrt( + (c.dx(0)*N[2] - c.dx(1)*N[0])**2 \
                              + (c.dx(1)*N[2] - c.dx(2)*N[1])**2 + 1e-3) * dGnd
        elif config['adjoint']['regularization_type'] == 'Tikhonov':
          s   = "    - using Tikhonov regularization -"
          self.R = a/2.0 * ( + (c.dx(0)*N[2] - c.dx(1)*N[0])**2 \
                             + (c.dx(1)*N[2] - c.dx(2)*N[1])**2 ) * dGnd
        else:
          s = "    - Valid regularizations are 'TV' and 'Tikhonov';" + \
              + " defaulting to Tikhonov regularization -"
          self.R = a/2.0 * ( + (c.dx(0)*N[2] - c.dx(1)*N[0])**2 \
                             + (c.dx(1)*N[2] - c.dx(2)*N[1])**2 ) * dGnd
      self.I += self.R
      print_text(s, self.color())
    
    # Derivative, with trial function dU.  These are the momentum equations 
    # in weak form multiplied by dU and integrated by parts
    F_adjoint  = derivative(A, U, model.dU)
    
    # Objective function constrained to obey the forward model
    I_adjoint  = self.I + F_adjoint

    # Gradient of this with respect to U in the direction of a test 
    # function yields a bilinear residual, which when solved yields the 
    # value of the adjoint variable
    self.dI    = derivative(I_adjoint, U, model.Phi)

    # Instead of treating the Lagrange multiplier as a trial function, treat 
    # it as a function.
    F_gradient = derivative(A, U, model.Lam)

    # This is a scalar quantity when discretized, as it contains no test or 
    # trial functions
    I_gradient = self.I + F_gradient

    # Differentiation wrt to the control variable in the direction of a test 
    # function yields a vector.  Assembly of this vector yields dJ/dbeta
    self.J = []
    rho    = TestFunction(model.Q)
    for c in control:
      self.J.append(derivative(I_gradient, c, rho))
    
    self.aw = lhs(self.dI)
    self.Lw = rhs(self.dI)
    
    # FIXME: this is a hack.
    self.bcs = []
    U_sp     = model.U.function_space()
    self.bcs.append(DirichletBC(U_sp.sub(0), 0.0, model.ff, 7))
    self.bcs.append(DirichletBC(U_sp.sub(1), 0.0, model.ff, 7))
    
  def solve(self):
    """
    Solves the bilinear residual created by differentiation of the 
    variational principle in combination with an objective function.
   """
    model  = self.model
    config = self.config

    s    = "::: solving Dukowicz adjoint velocity :::"
    print_text(s, self.color())
      
    aw = assemble(self.aw)
    Lw = assemble(self.Lw)
    for bc in self.bcs:
      bc.apply(aw, Lw)

    if config['model_order'] == 'stokes':
      a_solver = LUSolver('mumps')
    else:
      a_solver = KrylovSolver('cg', 'hypre_amg')

    a_solver.solve(aw, model.Lam.vector(), Lw)

    #if config['model_order'] == 'stokes':
    #  lam_nx, lam_ny, lam_nz, lam_np = model.Lam.split(True)
    #  lam_ix, lam_iy, lam_iz, lam_ip = model.Lam.split()
    #elif config['model_order'] == 'BP':
    #  lam_nx, lam_ny = model.Lam.split(True)
    #  lam_ix, lam_iy = model.Lam.split()

    #if config['adjoint']['surface_integral'] == 'shelves':
    #  lam_nx.vector()[model.gnd_dofs] = 0.0
    #  lam_ny.vector()[model.gnd_dofs] = 0.0
    #elif config['adjoint']['surface_integral'] == 'grounded':
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
    #                           "preconditioner" : "hypre_amg"})
    print_min_max(model.Lam, 'Lam')


class AdjointVelocity(Physics):
  """ 
  """
  def __init__(self, model, config):
    """ 
    Setup.
    """
    s   = "::: INITIALIZING ADJOINT VELOCITY PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    Q        = model.Q
    u_ob     = model.u_ob
    v_ob     = model.v_ob
    adot     = model.adot
    ds       = model.ds
    S        = model.S
    A        = model.A
    dU       = model.dU
    Phi      = model.Phi
    U        = model.U
    N        = model.N
    
    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    dSrf_s   = ds(6)         # surface
    dSrf_g   = ds(2)         # surface
    dGnd     = ds(3)         # grounded bed 
    dFlt     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dFlt   # bed

    if config['adjoint']['surface_integral'] == 'shelves':
      dSrf     = ds(6)
      s   = "    - integrating over shelves -"
    elif config['adjoint']['surface_integral'] == 'grounded':
      dSrf     = ds(2)
      s   = "    - integrating over grounded ice -"
    
    print_text(s, self.color())

    control = config['adjoint']['control_variable']
    alpha   = config['adjoint']['alpha']

    # Objective function; least squares over the surface.
    if config['adjoint']['objective_function'] == 'log':
      self.J = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                         / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dSrf
      s   = "    - using log objective function -"
    
    elif config['adjoint']['objective_function'] == 'kinematic':
      self.J = 0.5 * (+ U[0]*S.dx(0) + U[1]*S.dx(1) \
                      - (U[2] + adot))**2 * dSrf
      s   = "    - using kinematic objective function -"

    elif config['adjoint']['objective_function'] == 'linear':
      self.J = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dSrf
      s   = "    - using linear objective function -"
    
    elif config['adjoint']['objective_function'] == 'log_lin_hybrid':
      g1       = config['adjoint']['gamma1']
      g2       = config['adjoint']['gamma2']
      self.J1  = g1 * 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dSrf
      self.J2  = g2 * 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                                / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dSrf
      self.J1p = 0.5 * ((U[0] - u_ob)**2 + (U[1] - v_ob)**2) * dSrf
      self.J2p = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                           / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dSrf
      self.J   = self.J1  + self.J2
      self.Jp  = self.J1p + self.J2p
      s   = "    - using log/linear hybrid objective with gamma_1 = " \
            "%.1e and gamma_2 = %.1e -" % (g1, g2)

    else:
      s = "    - WARNING: adjoint objection function may be 'linear', " + \
          "'log', 'kinematic', or 'log_lin_hybrid'.  Defaulting to 'log' -"
      print_text(s, 'red', 1)
      self.J = 0.5 * ln(   (sqrt(U[0]**2 + U[1]**2) + 0.01) \
                         / (sqrt(u_ob**2 + v_ob**2) + 0.01))**2 * dSrf
      s   = "    - using log objective function -"
    print_text(s, self.color())

    if config['adjoint']['control_domain'] == 'bed':
      dR  = dBed
      s   = "    - regularizing over bed -"
    elif config['adjoint']['control_domain'] == 'surface':
      dR  = dSrf
      s   = "    - regularizing over surface -"
    elif config['adjoint']['control_domain'] == 'complete':
      if config['adjoint']['surface_integral'] == 'shelves':
        dR  = dx_s
      elif config['adjoint']['surface_integral'] == 'grounded':
        dR  = dx_g
      s   = "    - regularizing over entire domain -"
    
    # form regularization term 'R' :
    for a,c in zip(alpha,control):
      s   = "    - regularization parameter alpha = %.2E -" % a
      print_text(s, self.color())
      if config['adjoint']['regularization_type'] == 'TV':
        s       = "    - using total variation regularization -"
        self.R  = a * 0.5 * sqrt(inner(grad(c), grad(c)) + DOLFIN_EPS) * dR
        self.Rp = 0.5 * sqrt(inner(grad(c), grad(c)) + DOLFIN_EPS) * dR
      elif config['adjoint']['regularization_type'] == 'Tikhonov':
        s       = "    - using Tikhonov regularization -"
        self.R  = a * 0.5 * inner(grad(c), grad(c)) * dR
        self.Rp = 0.5 * inner(grad(c), grad(c)) * dR
      else:
        s       = "    - Valid regularizations are 'TV' and 'Tikhonov';" + \
                  + " defaulting to Tikhonov regularization -"
        self.R  = a * 0.5 * inner(grad(c), grad(c)) * dR
        self.Rp = 0.5 * inner(grad(c), grad(c)) * dR
      self.I = self.J + self.R
      print_text(s, self.color())

    # this is the adjoint of the momentum residual, the Lagrangian :
    L          = replace(A, {Phi:dU})

    # the Hamiltonian :
    H_U        = self.I + L

    # we desire the derivative of the Hamiltonian w.r.t. the model state U
    # in the direction of the test function Phi to vanish :
    self.dI    = derivative(H_U, U, Phi)

    # we need to evaluate the Hamiltonian with the values of Lam computed from
    # self.dI in order to get the derivative of the Hamiltonian w.r.t. the 
    # control variables.  Hence we need a new Lagrangian with the trial 
    # functions replaced with the computed Lam values.
    L_lam      = replace(L, {dU:model.Lam})

    # the Hamiltonian with unknowns replaced with computed Lam :
    self.H_lam = self.I + L_lam

    # the derivative of the Hamiltonian w.r.t. the control variables in the 
    # direction of a P1 test function :
    self.dHdc = []
    phi       = TestFunction(Q)
    for c in control:
      self.dHdc.append(derivative(self.H_lam, c, phi))
    
    self.aw = lhs(self.dI)
    self.Lw = rhs(self.dI)
    
    # FIXME: this is a hack.
    self.bcs = []
    U_sp     = model.U.function_space()
    self.bcs.append(DirichletBC(U_sp.sub(0), 0.0, model.ff, 7))
    self.bcs.append(DirichletBC(U_sp.sub(1), 0.0, model.ff, 7))

  def solve(self):
    """
    Solves the bilinear residual created by differentiation of the 
    variational principle in combination with an objective function.
    """
    model  = self.model
    config = self.config

    s    = "::: solving adjoint velocity :::"
    print_text(s, self.color())
      
    aw = assemble(self.aw)
    Lw = assemble(self.Lw)
    for bc in self.bcs:
      bc.apply(aw, Lw)
    
    if config['model_order'] == 'stokes':
      a_solver = LUSolver('mumps')
    else:
      a_solver = KrylovSolver('cg', 'hypre_amg')

    a_solver.solve(aw, model.Lam.vector(), Lw)

    #lam_nx, lam_ny = model.Lam.split(True)
    #lam_ix, lam_iy = model.Lam.split()

    #if config['adjoint']['surface_integral'] == 'shelves':
    #  lam_nx.vector()[model.gnd_dofs] = 0.0
    #  lam_ny.vector()[model.gnd_dofs] = 0.0
    #elif config['adjoint']['surface_integral'] == 'grounded':
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
    #                           "preconditioner" : "hypre_amg"})
    #print_min_max(norm(model.Lam), '||Lam||')
    print_min_max(model.Lam, 'Lam')


class SurfaceClimate(Physics):

  """
  Class which specifies surface mass balance, surface temperature using a 
  PDD model.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
                  attributes such as velocties, age, and surface climate
  """

  def __init__(self, model, config):
    self.model  = model
    self.config = config

  def solve(self):
    """
    Calculates PDD, surface temperature given current model geometry

    """
    s    = "::: solving surface climate :::"
    print_text(s, self.color())
    model  = self.model
    config = self.config

    T_ma  = config['surface_climate']['T_ma']
    T_w   = model.T_w
    S     = model.S.vector().array()
    lat   = model.lat.vector().array()
    
    # Apply the lapse rate to the surface boundary condition
    model.assign_variable(model.T_surface, T_ma(S, lat) + T_w)


class Age(Physics):
  r"""
  Class for calculating the age of the ice in steady state.

  :Very simple PDE:
     .. math::
      \vec{u} \cdot \nabla A = 1

  This equation, however, is numerically challenging due to its being 
  hyperbolic.  This is addressed by using a streamline upwind Petrov 
  Galerkin (SUPG) weighting.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
                  attributes such as velocties, age, and surface climate
  """

  def __init__(self, model, config):
    """ 
    Set up the equations 
    """
    s    = "::: INITIALIZING AGE PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    h = model.h
    
    Bub = FunctionSpace(model.mesh, "B", 4, constrained_domain=model.pBC)
    model.MQ  = model.Q + Bub

    # Trial and test
    a   = TrialFunction(model.MQ)
    phi = TestFunction(model.MQ)
    self.age = Function(model.MQ)

    # Steady state
    if config['mode'] == 'steady':
      s    = "    - using steady-state -"
      print_text(s, self.color())
      
      ## SUPG method :
      U      = as_vector([model.u, model.v, model.w])
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))
      
      # Residual 
      R = dot(U,grad(a)) - Constant(1.0)
      self.a = dot(U,grad(a)) * phi * dx
      self.L = Constant(1.0) * phi * dx

      # Weak form of residual
      self.F = R * phihat * dx

    else:
      s    = "    - using transient -"
      print_text(s, self.color())
      
      # Starting and midpoint quantities
      ahat   = model.ahat
      a0     = model.a0
      uhat   = model.uhat
      vhat   = model.vhat
      what   = model.what
      mhat   = model.mhat

      # Time step
      dt     = config['time_step']

      # SUPG method (note subtraction of mesh velocity) :
      U      = as_vector([uhat, vhat, what-mhat])
      Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
      phihat = phi + h/(2*Unorm) * dot(U,grad(phi))

      # Midpoint value of age for Crank-Nicholson
      a_mid = 0.5*(a + self.ahat)
      
      # Weak form of time dependent residual
      self.F = + (a - a0)/dt * phi * dx \
               + dot(U, grad(a_mid)) * phihat * dx \
               - 1.0 * phihat * dx

    # form the boundary conditions :
    if config['age']['use_smb_for_ela']:
      s    = "    - using adot (SMB) boundary condition -"
      print_text(s, self.color())
      self.bc_age = DirichletBC(model.MQ, 0.0, model.ff_acc, 1)
    
    else:
      s    = "    - using ELA boundary condition -"
      print_text(s, self.color())
      def above_ela(x,on_boundary):
        return x[2] > config['age']['ela'] and on_boundary
      self.bc_age = DirichletBC(model.Q, 0.0, above_ela)

  def solve(self, ahat=None, a0=None, uhat=None, what=None, vhat=None):
    """ 
    Solve the system
    
    :param ahat   : Observable estimate of the age
    :param a0     : Initial age of the ice
    :param uhat   : Horizontal velocity
    :param vhat   : Horizontal velocity perpendicular to :attr:`uhat`
    :param what   : Vertical velocity
    """
    model  = self.model
    config = self.config

    # Assign values to midpoint quantities and mesh velocity
    if ahat:
      model.assign_variable(model.ahat, ahat)
      model.assign_variable(model.a0,   a0)
      model.assign_variable(model.uhat, uhat)
      model.assign_variable(model.vhat, vhat)
      model.assign_variable(model.what, what)

    # Solve!
    s    = "::: solving age :::"
    print_text(s, self.color())
    #solve(lhs(self.F) == rhs(self.F), model.age, self.bc_age)
    solve(self.a == self.L, self.age, self.bc_age)
    model.age.interpolate(self.age)
    print_min_max(model.age, 'age')


class VelocityBalance(Physics):
  
  def __init__(self, model, config):
    """
    """ 
    s    = "::: INITIALIZING VELOCITY-BALANCE PHYSICS :::"
    print_text(s, self.color())


    self.model  = model
    self.config = config
    
    Q           = model.Q
    g           = model.g
    rho         = model.rhoi
    S           = model.S
    B           = model.B
    H           = S - B
    h           = model.h
    dSdx        = model.dSdx
    dSdy        = model.dSdy
    d_x         = model.d_x
    d_y         = model.d_y
    adot        = model.adot
   
    # assign the variables something that the user specifies : 
    kappa = config['balance_velocity']['kappa']
        
    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :

    phi  = TestFunction(Q)
    Ubar = TrialFunction(Q)
    Nx   = TrialFunction(Q)
    Ny   = TrialFunction(Q)
    
    # calculate horizontally smoothed driving stress :
    a_dSdx = + Nx * phi * dx \
             + (kappa*H)**2 * (phi.dx(0)*Nx.dx(0) + phi.dx(1)*Nx.dx(1)) * dx
    L_dSdx = rho * g * H * dSdx * phi * dx \
    
    a_dSdy = + Ny * phi * dx \
             + (kappa*H)**2 * (phi.dx(0)*Ny.dx(0) + phi.dx(1)*Ny.dx(1)) * dx
    L_dSdy = rho * g * H * dSdy * phi*dx \
    
    # SUPG method :
    if model.mesh.ufl_cell().topological_dimension() == 3:
      dS      = as_vector([d_x, d_y, 0.0])
    elif model.mesh.ufl_cell().topological_dimension() == 2:
      dS      = as_vector([d_x, d_y])
    phihat  = phi + h/(2*H) * ((H*dS[0]*phi).dx(0) + (H*dS[1]*phi).dx(1))
    #phihat  = phi + h/(2*H) * (H*dS[0]*phi.dx(0) + H*dS[1]*phi.dx(1))
    
    def L(u, uhat):
      if model.mesh.ufl_cell().topological_dimension() == 3:
        return div(uhat)*u + dot(grad(u), uhat)
      elif model.mesh.ufl_cell().topological_dimension() == 2:
        l = + (uhat[0].dx(0) + uhat[1].dx(1))*u \
            + u.dx(0)*uhat[0] + u.dx(1)*uhat[1]
        return l
    
    B = L(Ubar*H, dS) * phihat * dx
    a = adot * phihat * dx

    self.a_dSdx = a_dSdx
    self.a_dSdy = a_dSdy
    self.L_dSdx = L_dSdx
    self.L_dSdy = L_dSdy
    self.B      = B
    self.a      = a
    
  
  def solve(self):
    """
    Solve the balance velocity.
    """
    model = self.model

    s    = "::: calculating surface gradient :::"
    print_text(s, self.color())
    
    dSdx   = project(model.S.dx(0), model.Q)
    dSdy   = project(model.S.dx(1), model.Q)
    model.assign_variable(model.dSdx, dSdx)
    model.assign_variable(model.dSdy, dSdy)
    print_min_max(model.dSdx, 'dSdx')
    print_min_max(model.dSdy, 'dSdy')
    
    # update velocity direction from driving stress :
    s    = "::: solving for smoothed x-component of driving stress :::"
    print_text(s, self.color())
    solve(self.a_dSdx == self.L_dSdx, model.Nx)
    print_min_max(model.Nx, 'Nx')
    
    s    = "::: solving for smoothed y-component of driving stress :::"
    print_text(s, self.color())
    solve(self.a_dSdy == self.L_dSdy, model.Ny)
    print_min_max(model.Ny, 'Ny')
    
    # normalize the direction vector :
    s    =   "::: calculating normalized velocity direction" \
           + " from driving stress :::"
    print_text(s, self.color())
    d_x_v = model.Nx.vector().array()
    d_y_v = model.Ny.vector().array()
    d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + 1e-16)
    model.assign_variable(model.d_x, -d_x_v / d_n_v)
    model.assign_variable(model.d_y, -d_y_v / d_n_v)
    print_min_max(model.d_x, 'd_x')
    print_min_max(model.d_y, 'd_y')
    
    # calculate balance-velocity :
    s    = "::: solving velocity balance magnitude :::"
    print_text(s, self.color())
    solve(self.B == self.a, model.Ubar)
    print_min_max(model.Ubar, 'Ubar')
    
    # enforce positivity of balance-velocity :
    s    = "::: removing negative values of balance velocity :::"
    print_text(s, self.color())
    Ubar_v = model.Ubar.vector().array()
    Ubar_v[Ubar_v < 0] = 0
    model.assign_variable(model.Ubar, Ubar_v)
    print_min_max(model.Ubar, 'Ubar')


class SurfaceMassBalance(Physics):
  
  def __init__(self, model, config):
    """
    """ 
    s    = "::: INITIALIZING SURFACE-MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())


    self.model  = model
    self.config = config
    
    Q           = model.Q
    S           = model.S
    B           = model.B
    H           = S - B
    Mb          = model.Mb
    ubar        = model.ubar
    vbar        = model.vbar
    wbar        = model.wbar
   
    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :

    phi  = TestFunction(Q)
    adot = TrialFunction(Q)

    Ubar = as_vector([ubar, vbar, wbar])
    
    self.B = (div(Ubar*H) + Mb) * phi * dx
    self.a = adot * phi * dx
  
  def solve(self):
    """
    Solve for the surface mass balance.
    """
    model = self.model

    # calculate balance-velocity :
    s    = "::: solving for surface mass balance :::"
    print_text(s, self.color())
    solve(self.a == self.B, model.adot)
    print_min_max(model.adot, 'adot')
    

class SSA_Balance(Physics):

  def __init__(self, model, config):
    """
    """
    s    = "::: INITIALIZING SSA-BALANCE PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    mesh     = model.mesh
    Q2       = model.Q2
    ubar     = model.ubar
    vbar     = model.vbar
    S        = model.S
    B        = model.B
    H        = S - B
    beta     = model.beta
    rhoi     = model.rhoi
    g        = model.g
    etabar   = model.etabar
    
    Phi      = TestFunction(Q2)
    phi, psi = split(Phi)
    dU       = TrialFunction(Q2)
    du, dv   = split(dU)
    
    U        = as_vector([ubar, vbar])
    U_nm     = model.normalize_vector(U)
    U_n      = as_vector([U_nm[0],  U_nm[1]])
    U_t      = as_vector([U_nm[1], -U_nm[0]])
    
    s        = dot(dU, U_n)
    t        = dot(dU, U_t)
    U_s      = as_vector([s, t])
    grads    = as_vector([s.dx(0), s.dx(1)])
    gradt    = as_vector([t.dx(0), t.dx(1)])
    dsdi     = dot(grads, U_n)
    dsdj     = dot(grads, U_t)
    dtdi     = dot(gradt, U_n)
    dtdj     = dot(gradt, U_t)
    gradphi  = as_vector([phi.dx(0), phi.dx(1)]) 
    gradpsi  = as_vector([psi.dx(0), psi.dx(1)]) 
    gradS    = as_vector([S.dx(0),   S.dx(1)  ]) 
    dphidi   = dot(gradphi, U_n)
    dphidj   = dot(gradphi, U_t)
    dpsidi   = dot(gradpsi, U_n)
    dpsidj   = dot(gradpsi, U_t)
    dSdi     = dot(gradS,   U_n)
    dSdj     = dot(gradS,   U_t)
    gradphi  = as_vector([dphidi,    dphidj])
    gradpsi  = as_vector([dpsidi,    dpsidj])
    gradS    = as_vector([dSdi,      dSdj  ])
    
    epi_1  = as_vector([2*dsdi + dtdj, 
                        0.5*(dsdj + dtdi) ])
    epi_2  = as_vector([0.5*(dsdj + dtdi),
                        2*dtdj + dsdi     ])
    
    tau_id = phi * rhoi * g * H * gradS[0] * dx
    tau_jd = psi * rhoi * g * H * gradS[1] * dx

    tau_ib = - beta**2 * s * phi * dx
    tau_jb = - beta**2 * t * psi * dx

    tau_1  = - 2 * etabar * H * dot(epi_1, gradphi) * dx
    tau_2  = - 2 * etabar * H * dot(epi_2, gradpsi) * dx
    
    delta_1  = tau_1 + tau_ib - tau_id
    delta_2  = tau_2 + tau_jb - tau_jd
    
    delta  = delta_1 + delta_2
    U_s    = Function(Q2)

    # make the variables available to solve :
    self.delta = delta
    self.U_s   = U_s
    self.U_n   = U_n
    self.U_t   = U_t
    
  def solve(self):
    """
    """
    model = self.model
    
    s    = "::: solving 'SSA_Balance' for flow direction :::"
    print_text(s, self.color())
    solve(lhs(self.delta) == rhs(self.delta), self.U_s)
    u_s, u_t = self.U_s.split(True)
    model.assign_variable(model.u_s, u_s)
    model.assign_variable(model.u_t, u_t)
    print_min_max(model.u_s, 'u_s')
    print_min_max(model.u_t, 'u_t')

  def solve_component_stress(self):  
    """
    """
    model  = self.model
    config = self.config
    
    s    = "solving 'SSA_Balance' for stresses :::" 
    print_text(s, self.color())

    Q       = model.Q
    beta    = model.beta
    S       = model.S
    B       = model.B
    H       = S - B
    rhoi    = model.rhoi
    g       = model.g
    etabar  = model.etabar
    
    # solve with corrected velociites :
    model   = self.model
    config  = self.config

    Q       = model.Q
    U_s     = self.U_s
    U_n     = self.U_n
    U_t     = self.U_t

    phi     = TestFunction(Q)
    dtau    = TrialFunction(Q)
    
    s       = dot(U_s, U_n)
    t       = dot(U_s, U_t)
    grads   = as_vector([s.dx(0), s.dx(1)])
    gradt   = as_vector([t.dx(0), t.dx(1)])
    dsdi    = dot(grads, U_n)
    dsdj    = dot(grads, U_t)
    dtdi    = dot(gradt, U_n)
    dtdj    = dot(gradt, U_t)
    gradphi = as_vector([phi.dx(0), phi.dx(1)]) 
    gradS   = as_vector([S.dx(0),   S.dx(1)  ]) 
    dphidi  = dot(gradphi, U_n)
    dphidj  = dot(gradphi, U_t)
    dSdi    = dot(gradS,   U_n)
    dSdj    = dot(gradS,   U_t)
    gradphi = as_vector([dphidi, dphidj])
    gradS   = as_vector([dSdi,   dSdj  ])
    
    epi_1  = as_vector([2*dsdi + dtdj, 
                        0.5*(dsdj + dtdi) ])
    epi_2  = as_vector([0.5*(dsdj + dtdi),
                        2*dtdj + dsdi     ])
    
    tau_id_s = phi * rhoi * g * H * gradS[0] * dx
    tau_jd_s = phi * rhoi * g * H * gradS[1] * dx

    tau_ib_s = - beta**2 * s * phi * dx
    tau_jb_s = - beta**2 * t * phi * dx

    tau_ii_s = - 2 * etabar * H * epi_1[0] * gradphi[0] * dx
    tau_ij_s = - 2 * etabar * H * epi_1[1] * gradphi[1] * dx

    tau_ji_s = - 2 * etabar * H * epi_2[0] * gradphi[0] * dx
    tau_jj_s = - 2 * etabar * H * epi_2[1] * gradphi[1] * dx
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # solve the linear system :
    solve(M, model.tau_id.vector(), assemble(tau_id_s))
    print_min_max(model.tau_id, 'tau_id')
    solve(M, model.tau_jd.vector(), assemble(tau_jd_s))
    print_min_max(model.tau_jd, 'tau_jd')
    solve(M, model.tau_ib.vector(), assemble(tau_ib_s))
    print_min_max(model.tau_ib, 'tau_ib')
    solve(M, model.tau_jb.vector(), assemble(tau_jb_s))
    print_min_max(model.tau_jb, 'tau_jb')
    solve(M, model.tau_ii.vector(), assemble(tau_ii_s))
    print_min_max(model.tau_ii, 'tau_ii')
    solve(M, model.tau_ij.vector(), assemble(tau_ij_s))
    print_min_max(model.tau_ij, 'tau_ij')
    solve(M, model.tau_ji.vector(), assemble(tau_ji_s))
    print_min_max(model.tau_ji, 'tau_ji')
    solve(M, model.tau_jj.vector(), assemble(tau_jj_s))
    print_min_max(model.tau_jj, 'tau_jj')
   

class BP_Balance(Physics):

  def __init__(self, model, config):
    """
    """
    s    = "::: INITIALIZING BP-BALANCE PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    mesh     = model.mesh
    Q        = model.Q
    Q2       = model.Q2
    u        = model.u
    v        = model.v
    S        = model.S
    B        = model.B
    H        = S - B
    beta     = model.beta
    rhoi     = model.rhoi
    rhow     = model.rhow
    g        = model.g
    x        = model.x
    N        = model.N
    D        = model.D
    eta      = model.eta
    
    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    ds       = model.ds  
    dGnd     = ds(3)         # grounded bed
    dFlt     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dFlt   # bed
    
    f_w      = rhoi*g*(S - x[2]) + rhow*g*D
    
    Phi      = TestFunction(Q2)
    phi, psi = split(Phi)
    dU       = TrialFunction(Q2)
    du, dv   = split(dU)
    
    U        = as_vector([u, v])
    U_nm     = model.normalize_vector(U)
    U_n      = as_vector([U_nm[0],  U_nm[1]])
    U_t      = as_vector([U_nm[1], -U_nm[0]])
    
    s        = dot(dU, U_n)
    t        = dot(dU, U_t)
    U_s      = as_vector([s,       t      ])
    grads    = as_vector([s.dx(0), s.dx(1)])
    gradt    = as_vector([t.dx(0), t.dx(1)])
    dsdi     = dot(grads, U_n)
    dsdj     = dot(grads, U_t)
    dsdz     = s.dx(2)
    dtdi     = dot(gradt, U_n)
    dtdj     = dot(gradt, U_t)
    dtdz     = t.dx(2)
    gradphi  = as_vector([phi.dx(0), phi.dx(1)])
    gradpsi  = as_vector([psi.dx(0), psi.dx(1)])
    gradS    = as_vector([S.dx(0),   S.dx(1)  ])
    dphidi   = dot(gradphi, U_n)
    dphidj   = dot(gradphi, U_t)
    dpsidi   = dot(gradpsi, U_n)
    dpsidj   = dot(gradpsi, U_t)
    dSdi     = dot(gradS,   U_n)
    dSdj     = dot(gradS,   U_t)
    gradphi  = as_vector([dphidi,    dphidj,  phi.dx(2)])
    gradpsi  = as_vector([dpsidi,    dpsidj,  psi.dx(2)])
    gradS    = as_vector([dSdi,      dSdj,    S.dx(2)  ])
    
    epi_1  = as_vector([2*dsdi + dtdj, 
                        0.5*(dsdj + dtdi),
                        0.5*dsdz             ])
    epi_2  = as_vector([0.5*(dsdj + dtdi),
                             dsdi + 2*dtdj,
                        0.5*dtdz             ])
    
    F_id = phi * rhoi * g * gradS[0] * dx
    F_jd = psi * rhoi * g * gradS[1] * dx
    
    F_ib = - beta**2 * s * phi * dBed
    F_jb = - beta**2 * t * psi * dBed
    
    F_ip = f_w * N[0] * phi * dSde
    F_jp = f_w * N[1] * psi * dSde
    
    F_1  = - 2 * eta * dot(epi_1, gradphi) * dx
    F_2  = - 2 * eta * dot(epi_2, gradpsi) * dx
    
    delta_1  = F_1 + F_ib + F_ip - F_id
    delta_2  = F_2 + F_jb + F_jp - F_jd
    
    delta  = delta_1 + delta_2
    U_s    = Function(Q2)

    # make the variables available to solve :
    self.delta = delta
    self.U_nm  = U_nm
    self.U_s   = U_s
    self.U_n   = U_n
    self.U_t   = U_t
    self.f_w   = f_w
    
  def solve(self):
    """
    """
    model = self.model
    model.calc_eta()
    
    s    = "::: solving 'BP_Balance' for flow direction :::"
    print_text(s, self.color())
    solve(lhs(self.delta) == rhs(self.delta), self.U_s)
    u_s, u_t = self.U_s.split(True)
    model.assign_variable(model.u_s, u_s)
    model.assign_variable(model.u_t, u_t)
    print_min_max(model.u_s, 'u_s')
    print_min_max(model.u_t, 'u_t')

  def solve_component_stress(self):  
    """
    """
    model  = self.model
    config = self.config
    
    s    = "solving 'BP_Balance' for internal forces :::" 
    print_text(s, self.color())

    Q       = model.Q
    N       = model.N
    beta    = model.beta
    S       = model.S
    B       = model.B
    H       = S - B
    rhoi    = model.rhoi
    g       = model.g
    eta     = model.eta
    
    dx      = model.dx
    dx_s    = dx(1)
    dx_g    = dx(0)
    dx      = dx(1) + dx(0) # entire internal
    ds      = model.ds  
    dGnd    = ds(3)         # grounded bed
    dFlt    = ds(5)         # floating bed
    dSde    = ds(4)         # sides
    dBed    = dGnd + dFlt   # bed
    
    # solve with corrected velociites :
    model   = self.model
    config  = self.config

    Q       = model.Q
    f_w     = self.f_w
    U_s     = self.U_s
    U_n     = self.U_n
    U_t     = self.U_t
    U_nm    = self.U_nm

    phi     = TestFunction(Q)
    dtau    = TrialFunction(Q)
    
    s       = dot(U_s, U_n)
    t       = dot(U_s, U_t)
    U_s     = as_vector([s,       t      ])
    grads   = as_vector([s.dx(0), s.dx(1)])
    gradt   = as_vector([t.dx(0), t.dx(1)])
    dsdi    = dot(grads, U_n)
    dsdj    = dot(grads, U_t)
    dsdz    = s.dx(2)
    dtdi    = dot(gradt, U_n)
    dtdj    = dot(gradt, U_t)
    dtdz    = t.dx(2)
    dwdz    = -(dsdi + dtdj)
    gradphi = as_vector([phi.dx(0), phi.dx(1)])
    gradS   = as_vector([S.dx(0),   S.dx(1)  ])
    dphidi  = dot(gradphi, U_n)
    dphidj  = dot(gradphi, U_t)
    dSdi    = dot(gradS,   U_n)
    dSdj    = dot(gradS,   U_t)
    gradphi = as_vector([dphidi, dphidj, phi.dx(2)])
    gradS   = as_vector([dSdi,   dSdj,   S.dx(2)  ])
    
    epi_1   = as_vector([dsdi, 
                         0.5*(dsdj + dtdi),
                         0.5*dsdz             ])
    epi_2   = as_vector([0.5*(dtdi + dsdj),
                         dtdj,
                         0.5*dtdz             ])
    
    F_id_s = + phi * rhoi * g * gradS[0] * dx \
             - 2 * eta * dwdz * dphidi * dx #\
    #         + 2 * eta * dwdz * phi * N[0] * U_n[0] * ds
    F_jd_s = + phi * rhoi * g * gradS[1] * dx \
             - 2 * eta * dwdz * dphidj * dx #\
    #         + 2 * eta * dwdz * phi * N[1] * U_n[1] * ds
    
    F_ib_s = - beta**2 * s * phi * dBed
    F_jb_s = - beta**2 * t * phi * dBed
    
    F_ip_s = f_w * N[0] * phi * dSde
    F_jp_s = f_w * N[1] * phi * dSde
    
    F_pn_s = f_w * N[0] * phi * dSde
    F_pt_s = f_w * N[1] * phi * dSde
     
    F_ii_s = - 2 * eta * epi_1[0] * gradphi[0] * dx# \
    #         + 2 * eta * epi_1[0] * phi * N[0] * U_n[0] * ds
    #         + f_w * N[0] * phi * U_n[0] * dSde
    F_ij_s = - 2 * eta * epi_1[1] * gradphi[1] * dx# \
    #         + 2 * eta * epi_1[1] * phi * N[1] * U_n[1] * ds
    #         + f_w * N[1] * phi * U_n[1] * dSde
    F_iz_s = - 2 * eta * epi_1[2] * gradphi[2] * dx + F_ib_s #\
    #        + 2 * eta * epi_1[2] * phi * N[2] * ds
     
    F_ji_s = - 2 * eta * epi_2[0] * gradphi[0] * dx# \
    #         + 2 * eta * epi_2[0] * phi * N[0] * U_t[0] * ds
    #         + f_w * N[0] * phi * U_t[0] * dSde
    F_jj_s = - 2 * eta * epi_2[1] * gradphi[1] * dx# \
    #         + 2 * eta * epi_2[0] * phi * N[1] * U_t[1] * ds
    #         + f_w * N[1] * phi * U_t[1] * dSde
    F_jz_s = - 2 * eta * epi_2[2] * gradphi[2] * dx + F_jb_s #\
    #         + 2 * eta * epi_2[2] * phi * N[2] * ds
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # solve the linear system :
    solve(M, model.F_id.vector(), assemble(F_id_s))
    print_min_max(model.F_id, 'F_id')
    solve(M, model.F_jd.vector(), assemble(F_jd_s))
    print_min_max(model.F_jd, 'F_jd')
    solve(M, model.F_ib.vector(), assemble(F_ib_s))
    print_min_max(model.F_ib, 'F_ib')
    solve(M, model.F_jb.vector(), assemble(F_jb_s))
    print_min_max(model.F_jb, 'F_jb')
    solve(M, model.F_ip.vector(), assemble(F_ip_s))
    print_min_max(model.F_ip, 'F_ip')
    solve(M, model.F_jp.vector(), assemble(F_jp_s))
    print_min_max(model.F_jp, 'F_jp')
    solve(M, model.F_ii.vector(), assemble(F_ii_s))
    print_min_max(model.F_ii, 'F_ii')
    solve(M, model.F_ij.vector(), assemble(F_ij_s))
    print_min_max(model.F_ij, 'F_ij')
    solve(M, model.F_iz.vector(), assemble(F_iz_s))
    print_min_max(model.F_iz, 'F_iz')
    solve(M, model.F_ji.vector(), assemble(F_ji_s))
    print_min_max(model.F_ji, 'F_ji')
    solve(M, model.F_jj.vector(), assemble(F_jj_s))
    print_min_max(model.F_jj, 'F_jj')
    solve(M, model.F_jz.vector(), assemble(F_jz_s))
    print_min_max(model.F_jz, 'F_jz')
   
    if config['stokes_balance']['vert_integrate']: 
      s    = "::: vertically integrating 'BP_Balance' internal forces :::"
      print_text(s, self.color())
      
      tau_ii   = model.vert_integrate(model.F_ii, d='down')
      tau_ij   = model.vert_integrate(model.F_ij, d='down')
      tau_iz   = model.vert_integrate(model.F_iz, d='down')
                                                
      tau_ji   = model.vert_integrate(model.F_ji, d='down')
      tau_jj   = model.vert_integrate(model.F_jj, d='down')
      tau_jz   = model.vert_integrate(model.F_jz, d='down')
                                                
      tau_id   = model.vert_integrate(model.F_id, d='down')
      tau_jd   = model.vert_integrate(model.F_jd, d='down')
                                                
      tau_ip   = model.vert_integrate(model.F_ip, d='down')
      tau_jp   = model.vert_integrate(model.F_jp, d='down')
      
      tau_ib   = model.vert_extrude(model.F_ib, d='up')
      tau_jb   = model.vert_extrude(model.F_jb, d='up')
     
      model.assign_variable(model.tau_id, tau_id)
      model.assign_variable(model.tau_jd, tau_jd)
      model.assign_variable(model.tau_ib, tau_ib)
      model.assign_variable(model.tau_jb, tau_jb)
      model.assign_variable(model.tau_ip, tau_ip)
      model.assign_variable(model.tau_jp, tau_jp)
      model.assign_variable(model.tau_ii, tau_ii)
      model.assign_variable(model.tau_ij, tau_ij)
      model.assign_variable(model.tau_iz, tau_iz)
      model.assign_variable(model.tau_ji, tau_ji)
      model.assign_variable(model.tau_jj, tau_jj)
      model.assign_variable(model.tau_jz, tau_jz)
    
      print_min_max(model.tau_id, 'tau_id')
      print_min_max(model.tau_jd, 'tau_jd')
      print_min_max(model.tau_ib, 'tau_ib')
      print_min_max(model.tau_jb, 'tau_jb')
      print_min_max(model.tau_ip, 'tau_ip')
      print_min_max(model.tau_jp, 'tau_jp')
      print_min_max(model.tau_ii, 'tau_ii')
      print_min_max(model.tau_ij, 'tau_ij')
      print_min_max(model.tau_iz, 'tau_iz')
      print_min_max(model.tau_ji, 'tau_ji')
      print_min_max(model.tau_jj, 'tau_jj')
      print_min_max(model.tau_jz, 'tau_jz')


class VelocityHybrid(Physics):
  """
  New 2D hybrid model.
  """
  def __init__(self, model, config):
    """
    """
    s = "::: INITIALIZING HYBRID VELOCITY PHYSICS :::"
    print_text(s, self.color())
    
    self.model  = model
    self.config = config
    
    # CONSTANTS
    year = model.spy
    rho  = model.rhoi
    g    = model.g
    n    = model.n
    
    B       = model.B
    beta    = model.beta
    eps_reg = model.eps_reg
    H       = model.H
    S       = B + H
    deltax  = model.deltax
    sigmas  = model.sigmas
    T0_     = model.T0_
    T_      = model.T_
    U       = model.U
    
    Bc    = 3.61e-13*year
    Bw    = 1.73e3*year #model.a0 ice hardness
    Qc    = 6e4
    Qw    = model.Q0 # ice act. energy
    Rc    = model.R  # gas constant

    # FUNCTION SPACES
    Q  = model.Q
    HV = model.HV
    
    # MOMENTUM
    Phi = TestFunction(HV)
    dU  = TrialFunction(HV)
    
    # ANSATZ    
    coef  = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    dcoef = [lambda s:0.0, lambda s:5*s**3]
    
    u_   = [U[0],   U[2]]
    v_   = [U[1],   U[3]]
    phi_ = [Phi[0], Phi[2]]
    psi_ = [Phi[1], Phi[3]]
    
    u    = VerticalBasis(u_,  coef, dcoef)
    v    = VerticalBasis(v_,  coef, dcoef)
    phi  = VerticalBasis(phi_,coef, dcoef)
    psi  = VerticalBasis(psi_,coef, dcoef)
    
    # energy functions :
    T    = VerticalFDBasis(T_,  deltax, coef, sigmas)
    T0   = VerticalFDBasis(T0_, deltax, coef, sigmas)

    # METRICS FOR COORDINATE TRANSFORM
    def dsdx(s):
      return 1./H*(S.dx(0) - s*H.dx(0))
    
    def dsdy(s):
      return 1./H*(S.dx(1) - s*H.dx(1))
    
    def dsdz(s):
      return -1./H

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
      return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*H*eta_v(s)*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))
    
    def membrane_xy(s):
      return (phi.dx(s,1) + phi.ds(s)*dsdy(s))*H*eta_v(s)*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))
    
    def membrane_yx(s):
      return (psi.dx(s,0) + psi.ds(s)*dsdx(s))*H*eta_v(s)*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))
    
    def membrane_yy(s):
      return (psi.dx(s,1) + psi.ds(s)*dsdy(s))*H*eta_v(s)*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))
    
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
    
    # O(4)
    points  = np.array([0.0,       0.4688, 0.8302, 1.0   ])
    weights = np.array([0.4876/2., 0.4317, 0.2768, 0.0476])
    # O(6)
    #points  = np.array([1.0,     0.89976,   0.677186, 0.36312,   0.0        ])
    #weights = np.array([0.02778, 0.1654595, 0.274539, 0.3464285, 0.371519/2.])
    # O(8)
    #points  = np.array([1,         0.934001, 0.784483, 0.565235, 0.295758, 0          ])
    #weights = np.array([0.0181818, 0.10961,  0.18717,  0.248048, 0.28688,  0.300218/2.])
    
    vi = VerticalIntegrator(points, weights)

    R_x = - vi.intz(membrane_xx) \
          - vi.intz(membrane_xy) \
          - vi.intz(shear_xz) \
          - phi(1)*beta**2*u(1) \
          - vi.intz(tau_dx)
    R_y = - vi.intz(membrane_yx) \
          - vi.intz(membrane_yy) \
          - vi.intz(shear_yz) \
          - psi(1)*beta**2*v(1) \
          - vi.intz(tau_dy)

    # SIA
    self.R = (R_x + R_y)*dx
    #R = replace(R,{U:dU})
    self.J = derivative(self.R, U, dU)

    self.u = u
    self.v = v
    self.w = w

    #Define variational solver for the momentum problem
    ffc_options = config['velocity']['ffc_options']
    m_problem   = NonlinearVariationalProblem(self.R, U, J=self.J,
                                         form_compiler_parameters=ffc_options)
    self.m_solver = NonlinearVariationalSolver(m_problem)
    
    self.m_solver.parameters['nonlinear_solver']                      = 'newton'
    self.m_solver.parameters['newton_solver']['relaxation_parameter']    = 0.7
    self.m_solver.parameters['newton_solver']['relative_tolerance']      = 1e-5
    self.m_solver.parameters['newton_solver']['absolute_tolerance']      = 1e7
    self.m_solver.parameters['newton_solver']['maximum_iterations']      = 20
    self.m_solver.parameters['newton_solver']['error_on_nonconvergence'] = False
    self.m_solver.parameters['newton_solver']['linear_solver']        = 'mumps'
    self.m_solver.parameters['newton_solver']['report']        = True
  
  def solve(self):
    """
    Solves for hybrid velocity.
    """
    s    = "::: solving hybrid velocity :::"
    print_text(s, self.color())
    
    model = self.model
    Q     = model.Q

    solver_return = self.m_solver.solve()

    model.assign_variable(model.u_s, project(self.u(0.0), Q))
    model.assign_variable(model.v_s, project(self.v(0.0), Q))
    model.assign_variable(model.w_s, project(self.w(0.0), Q))

    model.assign_variable(model.u, model.u_s)
    model.assign_variable(model.v, model.v_s)
    model.assign_variable(model.w, model.w_s)

    print_min_max(model.u, 'u_s')
    print_min_max(model.v, 'v_s')
    print_min_max(model.w, 'w_s')

    model.assign_variable(model.u_b, project(self.u(1.0), Q))
    model.assign_variable(model.v_b, project(self.v(1.0), Q))
    model.assign_variable(model.w_b, project(self.w(1.0), Q))

    print_min_max(model.u_b, 'u_b')
    print_min_max(model.v_b, 'v_b')
    print_min_max(model.w_b, 'w_b')

    return solver_return


class MassTransportHybrid(Physics):
  """
  New 2D hybrid model.
  """
  def __init__(self, model, config):
    """
    """
    s = "::: INITIALIZING HYBRID MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    self.model  = model
    self.config = config
    
    # CONSTANTS
    year = model.spy
    rho  = model.rhoi
    g    = model.g
    n    = model.n
    A    = config['velocity']['A']
    
    Q      = model.Q
    B      = model.B
    beta   = model.beta
    adot   = model.adot
    ubar_c = model.ubar_c 
    vbar_c = model.vbar_c
    H      = model.H
    H0     = model.H0
    U      = model.U
    T_     = model.T_
    deltax = model.deltax
    sigmas = model.sigmas
    h      = model.h
    S      = B + H
    coef   = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    T      = VerticalFDBasis(T_, deltax, coef, sigmas)
    
    Bc    = 3.61e-13*year
    Bw    = 1.73e3*year #model.a0 ice hardness
    Qc    = 6e4
    Qw    = model.Q0 # ice act. energy
    Rc    = model.R  # gas constant
    
    # TIME STEP AND REGULARIZATION
    eps_reg = model.eps_reg
    self.dt = config['time_step']
    thklim  = config['free_surface']['thklim']
   
    # function spaces : 
    dH  = TrialFunction(Q)
    xsi = TestFunction(Q)
    
    def A_v(T):
      return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))
    
    # SIA DIFFUSION COEFFICIENT INTEGRAL TERM.
    def sia_int(s):
      return A_v(T.eval(s))*s**(n+1)
    
    # O(4)
    points  = np.array([0.0,       0.4688, 0.8302, 1.0   ])
    weights = np.array([0.4876/2., 0.4317, 0.2768, 0.0476])
    # O(6)
    #points  = np.array([1.0,     0.89976,   0.677186, 0.36312,   0.0        ])
    #weights = np.array([0.02778, 0.1654595, 0.274539, 0.3464285, 0.371519/2.])
    # O(8)
    #points  = np.array([1,         0.934001, 0.784483, 0.565235, 0.295758, 0          ])
    #weights = np.array([0.0181818, 0.10961,  0.18717,  0.248048, 0.28688,  0.300218/2.])
    
    vi = VerticalIntegrator(points, weights)
    
    #D = 2.*(rho*g)**n*A/(n+2.)*H**(n+2)*dot(grad(S),grad(S))**((n-1.)/2.)
    D = 2.0*(rho*g)**n*H**(n+2)*dot(grad(S),grad(S))**((n-1.0)/2.0)*vi.intz(sia_int) + rho*g*H**2/beta**2
    
    ubar = U[0]
    vbar = U[1]
    
    ubar_si = -D/H*S.dx(0)
    vbar_si = -D/H*S.dx(1)
    
    self.ubar_proj = (ubar-ubar_si)*xsi*dx
    self.vbar_proj = (vbar-vbar_si)*xsi*dx

    # mass term :
    self.M  = dH*xsi*dx
    
    # residual :
    R_thick = + (H-H0) / self.dt * xsi * dx \
              + D * dot(grad(S), grad(xsi)) * dx \
              + (Dx(ubar_c*H,0) + Dx(vbar_c*H,1)) * xsi * dx \
              - adot * xsi * dx

    # Jacobian :
    J_thick = derivative(R_thick, H, dH)

    bc = []#DirichletBC(Q, thklim, 'on_boundary')
    
    #Define variational solver for the mass problem
    ffc_options  = config['free_surface']['ffc_options']
    mass_problem = NonlinearVariationalProblem(R_thick, H, J=J_thick, bcs=bc,
                                        form_compiler_parameters=ffc_options)
    self.mass_solver  = NonlinearVariationalSolver(mass_problem)
    self.mass_solver.parameters['nonlinear_solver']                  = 'snes'
    self.mass_solver.parameters['snes_solver']['method']             = 'vinewtonrsls'
    self.mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-6
    self.mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-6
    self.mass_solver.parameters['snes_solver']['maximum_iterations'] = 20
    self.mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
    self.mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'
    self.mass_solver.parameters['snes_solver']['report'] = True

  def solve(self):
    """
    Solves for hybrid conservation of mass.
    """
    config = self.config
    model  = self.model

    ffc_options = config['free_surface']['ffc_options']

    # Find corrective velocities
    s    = "::: solving for corrective velocities :::"
    print_text(s, self.color())

    solve(self.M == self.ubar_proj, model.ubar_c, 
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=ffc_options)

    solve(self.M == self.vbar_proj, model.vbar_c, 
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=ffc_options)

    print_min_max(model.ubar_c, 'ubar_c')
    print_min_max(model.vbar_c, 'vbar_c')

    # SOLVE MASS CONSERVATION bounded by (H_max, H_min) :
    s    = "::: solving hybrid mass-balance with time step "+str(self.dt(0))+" :::"
    print_text(s, self.color())
    solver_return = self.mass_solver.solve(model.H_min, model.H_max)

    print_min_max(model.H, 'H')
    
    # update the surface :
    s    = "::: updating surface :::"
    print_text(s, self.color())
    S    = project(model.B + model.H, model.Q)
    model.assign_variable(model.S, S)
    model.assign_variable(model.H0,model.H)
    print_min_max(model.S, 'S')
    return solver_return

class EnergyHybrid(Physics):
  """
  New 2D hybrid model.
  """
  def __init__(self, model, config):
    """
    """
    s = "::: INITIALIZING HYBRID ENERGY PHYSICS :::"
    print_text(s, self.color())
    
    self.model  = model
    self.config = config
    
    # CONSTANTS
    year    = model.spy
    g       = model.g
    n       = model.n
    A       = config['velocity']['A']
            
    k       = model.ki
    rho     = model.rhoi
    Cp      = model.ci
    kappa   = year*k/(rho*Cp)
            
    q_geo   = model.q_geo
    S       = model.S
    B       = model.B
    beta    = model.beta
    T_s     = model.T_surface
    T_w     = model.T_w
    U       = model.U
    H       = model.H
    H0      = model.H0
    T_      = model.T_
    T0_     = model.T0_
    deltax  = model.deltax
    sigmas  = model.sigmas
    eps_reg = model.eps_reg
    h       = model.h
    dt      = config['time_step']
    N_T     = config['enthalpy']['N_T']
    
    Bc      = 3.61e-13*year
    Bw      = 1.73e3*year  # model.a0 ice hardness
    Qc      = 6e4
    Qw      = model.Q0     # ice act. energy
    Rc      = model.R      # gas constant
    gamma   = model.gamma  # pressure melting point depth dependence
   
    # get velocity components : 
    # ANSATZ    
    coef  = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    dcoef = [lambda s:0.0, lambda s:5*s**3]
    
    u_   = [U[0], U[2]]
    v_   = [U[1], U[3]]
    
    u    = VerticalBasis(u_, coef, dcoef)
    v    = VerticalBasis(v_, coef, dcoef)
    
    # FUNCTION SPACES
    Q = model.Q
    Z = model.Z
    
    # ENERGY BALANCE 
    Psi = TestFunction(Z)
    dT  = TrialFunction(Z)

    # initialize surface temperature :
    model.assign_variable(T0_, project(as_vector([T_s]*N_T), Z))

    T  = VerticalFDBasis(T_,  deltax, coef, sigmas)
    T0 = VerticalFDBasis(T0_, deltax, coef, sigmas)

    # METRICS FOR COORDINATE TRANSFORM
    def dsdx(s):
      return 1./H*(S.dx(0) - s*H.dx(0))
    
    def dsdy(s):
      return 1./H*(S.dx(1) - s*H.dx(1))
    
    def dsdz(s):
      return -1./H
    
    def epsilon_dot(s):
      return ( + (u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
               + (v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
               + (u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
               + 0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
               + (+ (u.dx(s,1) + u.ds(s)*dsdy(s)) \
                  + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
               + eps_reg)
    
    def A_v(T):
      return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))
    
    def eta_v(s):
      return A_v(T0.eval(s))**(-1./n)/2.*epsilon_dot(s)**((1.-n)/(2*n))
    
    def w(s):
      w_0 = (U[0].dx(0) + U[1].dx(1))*(s-1.)
      w_2 = + (U[2].dx(0) + U[3].dx(1))*(s**(n+2) - s)/(n+1) \
            + (n+2)/H*U[2]*(1./(n+1)*(s**(n+1) - 1.)*S.dx(0) \
            - 1./(n+1)*(s**(n+2) - 1.)*H.dx(0)) \
            + (n+2)/H*U[3]*(+ 1./(n+1)*(s**(n+1) - 1.)*S.dx(1) \
                            - 1./(n+1)*(s**(n+2) - 1.)*H.dx(1))
      return (u(1)*B.dx(0) + v(1)*B.dx(1)) - 1./dsdz(s)*(w_0 + w_2) 
    
    R_T = 0

    for i in range(N_T):
      # SIGMA COORDINATE
      s = i/(N_T-1.0)
    
      # EFFECTIVE VERTICAL VELOCITY
      w_eff = + u(s)*dsdx(s) + v(s)*dsdy(s) + w(s)*dsdz(s) \
              + 1.0/H*(1.0 - s)*(H - H0)/dt
    
      # STRAIN HEAT
      Phi_strain = (2*n)/(n+1)*2*eta_v(s)*epsilon_dot(s)
    
      # STABILIZATION SCHEME
      Umag   = sqrt(u(s)**2 + v(s)**2 + 1e-3)
      tau    = h/(2*Umag)
      Psihat = Psi[i] + tau*(u(s)*Psi[i].dx(0) + v(s)*Psi[i].dx(1))
    
      # TIME DERIVATIVE
      dTdt = (T(i) - T0(i))/dt
    
      # SURFACE BOUNDARY
      if i==0:
        R_T += Psi[i]*(T(i) - T_s)*dx
      # BASAL BOUNDARY
      elif i==(N_T-1):
        R_T += dTdt*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 
        R_T += -w_eff*q_geo/(rho*Cp*kappa*dsdz(s))*Psi[i]*dx
        f    = (q_geo + 0.5*beta**2*(u(s)**2 + v(s)**2))/(rho*Cp*kappa*dsdz(s))
        R_T += -2.*kappa*dsdz(s)**2*(+ (T(N_T-2) - T(N_T-1)) / deltax**2 \
                                     - f/deltax)*Psi[i]*dx
      # INTERIOR
      else:
        R_T += dTdt*Psi[i]*dx
        R_T += -kappa*dsdz(s)**2.*T.d2s(i)*Psi[i]*dx
        R_T += w_eff*T.ds(i)*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 
    
    # PRETEND THIS IS LINEAR (A GOOD APPROXIMATION IN THE TRANSIENT CASE)
    self.R_T = replace(R_T, {T_:dT})

    # pressure melting point stuff :
    self.Tm  = as_vector([T_w - gamma*sigma*H for sigma in sigmas])

  def solve(self):
    """
    Solves for hybrid energy.
    """
    s    = "::: solving hybrid energy :::"
    print_text(s, self.color())
    
    config = self.config
    model  = self.model
    
    Q      = model.Q
    T_     = model.T_
    rhoi   = model.rhoi
    L      = model.L
    q_geo  = model.q_geo
    beta   = model.beta
    u_b    = model.u_b
    v_b    = model.v_b
    w_b    = model.w_b
    q_fric = 0.5 * beta**2 * (u_b**2 + v_b**2 + w_b**2)

    ffc_options = config['enthalpy']['ffc_options']

    # SOLVE TEMPERATURE
    solve(lhs(self.R_T) == rhs(self.R_T), model.T_,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=ffc_options)    
    print_min_max(model.T_, 'T_')

    # Update temperature field
    s    = "::: calculating pressure-melting point :::"
    print_text(s, self.color())
    T_melt  = project(self.Tm)
    Tb_m    = T_melt.split(True)[-1]  # deepcopy avoids projections
    model.assign_variable(model.T_melt, Tb_m)
    print_min_max(T_melt, 'T_melt')
    
    #  correct for pressure melting point :
    T_v                 = T_.vector().array()
    T_melt_v            = T_melt.vector().array()
    T_v[T_v > T_melt_v] = T_melt_v[T_v > T_melt_v]
    model.assign_variable(T_, T_v)
    
    out_T = T_.split(True)            # deepcopy avoids projections
    
    model.assign_variable(model.Ts, out_T[0])
    model.assign_variable(model.Tb, out_T[-1]) 

    print_min_max(model.Ts, 'T_S')
    print_min_max(model.Tb, 'T_B')
    
    # calculate melt-rate : 
    s = "::: calculating basal melt-rate :::"
    print_text(s, self.color())
    nMb   = project((q_geo + q_fric) / (L*rhoi))
    model.assign_variable(model.Mb,  nMb)
    print_min_max(model.Mb, 'Mb')
