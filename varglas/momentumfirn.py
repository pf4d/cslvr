from fenics                 import *
from dolfin_adjoint         import *
from varglas.io             import print_text, print_min_max
from varglas.d1model        import D1Model
from varglas.physics_new    import Physics
from varglas.momentum       import Momentum
import numpy                    as np
import sys


class MomentumFirn(Momentum):

  def __init__(self, model, solve_params=None):
    """
    """
    s = "::: INITIALIZING FIRN MOMENTUM PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D1Model:
      s = ">>> FirnDensity REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params

    mesh    = model.mesh
    Q       = model.Q
    Q3      = model.Q3

    kcHh    = model.kcHh
    kcLw    = model.kcLw
   
    H       = model.H                         # enthalpy
    T       = model.T                         # temperature
    w       = model.w                         # velocity
    w_1     = model.w_1                       # previous step's velocity
    m       = model.m                         # mesh velocity
    bdot    = model.bdot                      # average annual accumulation
    rhoCoef = model.rhoCoef                   # density ceofficient
    dt      = model.time_step                 # timestep
    g       = model.g                         # gravitational acceleration
    kg      = model.kg                        # grain growth coefficient
    Ec      = model.Ec                        # act. energy for water in ice
    Eg      = model.Eg                        # act. energy for grain growth
    R       = model.R                         # universal gas constant
    rhoi    = model.rhoi                      # density of ice
    rhom    = model.rhom                      # critical density
    rho     = model.rho                       # density

    eta     = TestFunction(Q)
    dw      = TrialFunction(Q)
  
    dQ      = TrialFunction(Q3)
    U       = Function(Q3)
    U_1     = Function(Q3)
    Phi     = TestFunction(Q3)
    
    self.assrho   = FunctionAssigner(model.Q, model.Q3.sub(0))
    self.asssig   = FunctionAssigner(model.Q, model.Q3.sub(1))
    self.assrss   = FunctionAssigner(model.Q, model.Q3.sub(2))

    rho,   sigma,   r    = U
    rho_1, sigma_1, r_1  = U_1
    phi,   psi,     xi   = Phi

    # initialize :
    U_i = project(as_vector([model.rho, model.sigma, model.r]), Q3)
    model.assign_variable(U,   U_i)
    model.assign_variable(U_1, U_i)

    # rho residual :
    theta     = 0.878
    rho_mid   = theta*rho + (1 - theta)*rho_1
    
    drhodt    = rhoCoef * exp( -Ec/(R*T) ) * (rhoi - rho_mid) * sigma / r
    d_rho     = + (rho - rho_1)/dt * phi * dx \
                - drhodt * phi * dx \
                + w * rho_mid.dx(0) * phi * dx 
    
    # sigma residual : 
    theta     = 0.878
    sig_mid   = theta*sigma + (1 - theta)*sigma_1
    dsigdt    = bdot * g
    d_sigma   = + (sigma - sigma_1)/dt * psi * dx \
                - dsigdt * psi * dx \
                + w * sig_mid.dx(0) * psi * dx

    # r residual :
    theta   = 0.878
    r_mid   = theta*r + (1 - theta)*r_1
    drdt    = kg * exp( -Eg/(R*T) )
    d_r     = + (r - r_1)/dt * xi * dx \
              - drdt * xi * dx \
              + w * r_mid.dx(0) * xi * dx

    # velocity residual :
    theta   = 0.878
    w_mid   = theta*dw + (1 - theta)*w_1
    
    w_delta = + rho * w_mid.dx(0) * eta * dx \
              + drhodt * eta * dx
    
    ## Arthern equation of strain rate from 'Sorge's Law' :
    #w_delta = + rho**2 * w_mid.dx(0) * eta * dx \
    #          - bdot * rho.dx(0) * eta * dx

    surface  = model.surface
    rhoBc    = DirichletBC(Q3.sub(0), model.rho_surface,   model.surface)
    sigmaBc  = DirichletBC(Q3.sub(1), model.sigma_surface, model.surface)
    rBc      = DirichletBC(Q3.sub(2), model.r_surface,     model.surface)
    self.wBc = DirichletBC(Q,         model.w_surface,     model.surface)

    self.bcs = [rhoBc, sigmaBc, rBc]
    
    self.delta   = d_rho + d_sigma + d_r
    self.w_delta = w_delta
    self.J       = derivative(self.delta, U, dQ)
    self.rho     = rho
    self.sigma   = sigma
    self.r       = r
    self.U       = U
    self.U_1     = U_1
    self.drhodt  = drhodt
  
  def get_residual(self):
    """
    Returns the momentum residual.
    """
    return self.delta

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
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    params = {'newton_solver' : {'relaxation_parameter'    : 1.0,
                                 'maximum_iterations'      : 25,
                                 'error_on_nonconvergence' : False,
                                 'relative_tolerance'      : 1e-10,
                                 'absolute_tolerance'      : 1e-10}}
    m_params  = {'solver' : params}
    return m_params

  def solve_compaction_velocity(self, annotate=True):
    """
    """
    s    = "::: solving firn compaction velocity :::"
    print_text(s, self.color())

    w_delta = self.w_delta
    model   = self.model
    
    # linear solve :
    solve(lhs(w_delta) == rhs(w_delta), model.w, self.wBc, annotate=annotate)
    print_min_max(model.w, 'w')

  def solve(self, annotate=True):
    """
    """
    s    = "::: solving firn density, overburden stress, and grain radius :::"
    print_text(s, self.color())
    
    model   = self.model

    # newton's iterative method :
    solve(self.delta == 0, self.U, bcs=self.bcs, J=self.J, 
          solver_parameters=self.solve_params['solver'],
          annotate=annotate)
    rho, sigma, r = self.U.split()

    self.assrho.assign(model.rho,   rho)
    self.asssig.assign(model.sigma, sigma)
    self.assrss.assign(model.r,     r)

    rhop = model.rho.vector().array()

    # update kc term in drhodt :
    # if rho >  550, kc = kcHigh
    # if rho <= 550, kc = kcLow
    # with parameterizations given by ligtenberg et all 2011
    A                   = model.rhoi(0)/model.rhow(0) * 1e3 * model.adot
    rhoCoefNew          = np.ones(model.dof)
    rhoHigh             = np.where(rhop >  550)[0]
    rhoLow              = np.where(rhop <= 550)[0]
    rhoCoefNew[rhoHigh] = model.kcHh(0) * (2.366 - 0.293*ln(A))
    rhoCoefNew[rhoLow]  = model.kcLw(0) * (1.435 - 0.151*ln(A))
    model.assign_variable(model.rhoCoef, rhoCoefNew)
    
    print_min_max(model.rho,   'rho')
    print_min_max(model.sigma, 'sigma')
    print_min_max(model.r,     'r^2')
    
    s    = "::: solving firn densification rate :::"
    print_text(s, self.color())
    model.assign_variable(model.drhodt,  project(self.drhodt))
    print_min_max(model.drhodt, 'drho/dt')
  
    self.solve_compaction_velocity(annotate=annotate)



