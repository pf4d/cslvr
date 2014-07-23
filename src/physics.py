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

from pylab     import ndarray
from fenics    import *
from termcolor import colored, cprint
import numpy as np
import numpy.linalg as linalg


class VelocityStokes(object):
  r"""  
  This class solves the non-linear Blatter-Pattyn momentum balance, 
  given a possibly non-uniform temperature field.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
	                attributes such as velocties, age, and surface climate

  **Equations**
 
  +-------------------+---------------+---------------------------------------+
  |Equation Name      |Condition      | Formula                               |
  +===================+===============+=======================================+
  |Variational        |               |.. math::                              |
  |Principle          |               |   \mathcal{A}[\textbf{u}, P] =        |
  |For Power Law      |               |   \int\limits_{\Omega}\frac{2n}{n+1}  |
  |Rheology           |               |   \eta\left(\dot{\epsilon}^{2}\right) |
  |                   |None           |   \dot{\epsilon}^{2} + \rho\textbf{g} |
  |                   |               |   \cdot\textbf{u}-P\nabla\cdot        |
  |                   |               |   \textbf{u}                          |
  |                   |               |   \ d\Omega+\int\limits_{\Gamma_{B}}  |
  |                   |               |   \frac{\beta^{2}}{2}H^{r}\textbf{u}  |
  |                   |               |   \cdot                               |
  |                   |               |   \textbf{u}+P\textbf{u}\cdot         |
  |                   |               |   \textbf{n} d\Gamma                  |
  +-------------------+---------------+---------------------------------------+
  |Rate of            |None           |.. math::                              |
  |strain             |               |   \eta\left(\dot{\epsilon}^{2}\right) |
  |                   |               |   =                                   |
  |tensor             |               |   b(T)\left[\dot{\epsilon}^{2}\right] |
  |                   |               |   ^{\frac{1-n}{2n}}                   |
  +-------------------+---------------+---------------------------------------+
  |Temperature        |Viscosity mode +.. math::                              |
  |Dependent          |is isothermal  |   b(T) = A_0^{\frac{-1}{n}}           |
  |Rate Factor        +---------------+---------------------------------------+
  |                   |Viscosity mode |Model dependent                        |
  |                   |is linear      |                                       |
  |                   +---------------+---------------------------------------+
  |                   |Viscosity mode |                                       |
  |                   |is full        |.. math::                              |
  |                   |               |   b(T) = \left[Ea(T)e^{-\frac{Q(T)}   |
  |                   |               |   {RT^*}}                             |
  |                   |               |   \right]^{\frac{-1}{n}}              |
  +-------------------+---------------+---------------------------------------+
  
  **Terms**

  +------------+-------------------------+------------------------------------+
  |Equation    |Term                     | Description                        |
  +============+=========================+====================================+
  |Variational |.. math::                |*Viscous dissipation* including     |
  |Principle   |                         |terms for strain rate dependent ice |
  |For Power   |   \frac{2n}{n+1}        |viscosity and the strain rate       |
  | Law        |   \eta\left(\dot        |tensor, respectively                |
  |Rheology    |   {\epsilon}^{2}        |                                    |
  |            |   \right)\dot{          |                                    |
  |            |   \epsilon}^{2}         |                                    |
  |            +-------------------------+------------------------------------+
  |            |.. math::                |*Graviataional potential* energy    |
  |            |   \rho\textbf{g}\cdot   |calculated using the density,       |
  |            |   \textbf{u}            |graviational force, and ice velocity|
  |            +-------------------------+------------------------------------+
  |            |.. math::                |*Incompressibility constraint*      |
  |            |    P\nabla\cdot         |included terms for pressure and the |
  |            |    \textbf{u}\ d\Omega  |divergence of the ice velocity      |
  |            +-------------------------+------------------------------------+
  |            |.. math::                |*Frictional head dissipation*       |
  |            |   \frac{\beta^{2}}      |including terms for the basal       |
  |            |   {2}H^{r}\textbf{u}    |sliding coefficient, ice thickness, |
  |            |   \cdot\textbf{u}       |and the ice velocity dotted into    |
  |            |                         |itself                              |
  |            +-------------------------+------------------------------------+
  |            |.. math::                |*Impenetrability constraint*        |
  |            |   P\textbf{u}\cdot      |calculated using the pressure and   |
  |            |   \textbf{n}            |the ice velocity dotted into the    |
  |            |                         |outward normal vector               |
  +------------+-------------------------+------------------------------------+
  |Rate of     |.. math::                |Temperature dependent rate factor,  |
  |strain      |   b(T)\left[\dot        |square of the second invarient of   |
  |tensor      |   {\epsilon}^{2}        |the strain rate tensor              |
  |            |   \right]^              |                                    |
  |            |  {\frac{1-n}{2n}}       |                                    |
  +------------+-------------------------+------------------------------------+
  |Temperature |.. math::                |Enhancement factor                  |
  |Dependent   |   E                     |                                    |
  |Rate Factor +-------------------------+------------------------------------+
  |            |.. math::                |Temperature dependent parameters    |
  |            |   a(T)                  |                                    |
  |            |                         |                                    |
  |            |   Q(T)                  |                                    |
  |            +-------------------------+------------------------------------+
  |            |.. math::                |Rate constant                       |
  |            |   R                     |                                    |
  +            +-------------------------+------------------------------------+
  |            |.. math::                |Temperature corrected for melting   |
  |            |   T^*                   |point dependence                    |
  +------------+-------------------------+------------------------------------+
  
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    self.model    = model
    self.config   = config

    mesh          = model.mesh
    r             = config['velocity']['r']
    Q             = model.Q
    Q4            = model.Q4
    n             = model.n
    b             = model.b
    Tstar         = model.Tstar
    T             = model.T
    gamma         = model.gamma
    S             = model.S
    B             = model.B
    x             = model.x
    E             = model.E
    W             = model.W
    R             = model.R
    epsdot        = model.epsdot
    eps_reg       = model.eps_reg
    eta           = model.eta
    rho           = model.rho
    rho_w         = model.rho_w
    g             = model.g
    Vd            = model.Vd
    Pe            = model.Pe
    Sl            = model.Sl
    Pc            = model.Pc
    Nc            = model.Nc
    Pb            = model.Pb
    Lsq           = model.Lsq
    beta2         = model.beta2

    newton_params = config['velocity']['newton_params']
    A0            = config['velocity']['A0']
    
    # initialize bed friction coefficient :
    model.assign_variable(beta2, config['velocity']['beta2'])
   
    # initialize enhancement factor :
    model.assign_variable(E, config['velocity']['E'])

    # pressure boundary :
    class Depth(Expression):
      def eval(self, values, x):
        b         = model.B_ex(x[0], x[1], x[2])
        values[0] = abs(min(0, b))
    D = Depth(element=Q.ufl_element())
    N = FacetNormal(mesh)
    
    # Check if there are non-linear solver parameters defined.  If not, set 
    # them to dolfin's default.  The default is not likely to converge if 
    # thermomechanical coupling is used.
    if newton_params:
      self.newton_params = newton_params
    
    else:
      self.newton_params = NonlinearVariationalSolver.default_parameters()
    
    # Define a test function
    Phi                  = TestFunction(Q4)

    # Define a trial function
    dU                   = TrialFunction(Q4)
    model.U              = Function(Q4)
    U                    = model.U
 
    phi, psi, xsi, kappa = split(Phi)
    du,  dv,  dw,  dP    = split(dU)
    u,   v,   w,   P     = split(U)

    # set up surfaces to integrate :
    ds       = model.ds  
    dSrf     = ds(2)        # surface
    dGnd     = ds(3)        # grounded bed 
    dFlt     = ds(5)        # floating bed
    dFltS    = ds(6)        # marine terminating sides
    dBed     = dGnd + dFlt  # bed

    # Set the value of b, the temperature dependent ice hardness parameter,
		# using the most recently calculated temperature field, if expected.
    if   config['velocity']['viscosity_mode'] == 'isothermal':
      b = A0**(-1/n)
    
    elif config['velocity']['viscosity_mode'] == 'linear':
      b = config['velocity']['b_linear']
      n = 1.0
    
    else:
      # Define pressure corrected temperature
      Tstar = T + gamma * (S - x[2])
       
      # Define ice hardness parameteri
      a_T = conditional( lt(Tstar, 263.15), 1.1384496e-5, 5.45e10 )
      Q_T = conditional( lt(Tstar, 263.15), 6e4,13.9e4 )
      b   = ( E * (a_T * (1 + 181.25*W)) * exp( -Q_T / (R * Tstar) ) )**(-1/n)
    
    # Second invariant of the strain rate tensor squared
    term   = + 0.5*(   (u.dx(1) + v.dx(0))**2  \
                     + (u.dx(2) + w.dx(0))**2  \
                     + (v.dx(2) + w.dx(1))**2) \
             + u.dx(0)**2 + v.dx(1)**2 + w.dx(2)**2 
    epsdot = 0.5 * term + eps_reg
    eta    = b * epsdot**((1.0 - n) / (2*n))

    # 1) Viscous dissipation
    Vd     = (2*n)/(n+1) * b * epsdot**((n+1)/(2*n))

    # 2) Potential energy
    Pe     = rho * g * w

    # 3) Dissipation by sliding
    Sl     = 0.5 * beta2 * (S - B)**r * (u**2 + v**2 + w**2)

    # 4) Incompressibility constraint
    Pc     = -P * (u.dx(0) + v.dx(1) + w.dx(2)) 
    
    # 5) Impenetrability constraint
    Nc     = P * (u*B.dx(0) + v*B.dx(1) - w)

    # 6) pressure boundary
    Pb     = (rho*g*(S - B) + rho_w*g*B) / (S - B) * (u + v + w) 

    g      = Constant((0.0, 0.0, g))
    h      = CellSize(mesh)
    tau    = h**2 / (12 * b * rho**2)
    Lsq    = -tau * dot( (grad(P) + rho*g), (grad(P) + rho*g) )
    
    # Variational principle
    A      = (Vd + Pe + Pc + Lsq)*dx + Sl*dGnd + Nc*dBed + Pb*dFltS

    model.A      = A
    model.epsdot = epsdot
    model.eta    = eta
    model.Vd     = Vd
    model.Pe     = Pe
    model.Sl     = Sl
    model.Pc     = Pc
    model.Nc     = Nc
    model.Pb     = Pb
    model.Lsq    = Lsq
    model.u      = u
    model.v      = v
    model.w      = w
    model.P      = P

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.F = derivative(A, U, Phi)   

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.J = derivative(self.F, U, dU)

  def solve(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    # Note that for solving the full Stokes functional, the edges of the 
    # domain require some sort of boundary condition other than homogeneous 
    # Neumann, since the geometry of the domain dictates the flow, not an 
    # imposed driving stress.  Here we have two options, all zeros or the 
    # solution that is already stored in the model class.  For the latter of 
    # these two options, this would mean that if you wanted to have some 
    # arbitrary section of greenland as a domain, you could solve the first 
    # order equations, which happily operate with homogeneous Neumann, and 
    # impose these as Dirichlet boundary conditions for the Stokes equations.
    model  = self.model
    config = self.config
    Q4     = model.Q4
    Q      = model.Q

    self.bcs = []

    if config['velocity']['boundaries'] == 'homogeneous':
      self.bcs.append(DirichletBC(Q4.sub(0), 0.0, model.ff, 4))
      self.bcs.append(DirichletBC(Q4.sub(1), 0.0, model.ff, 4))
      self.bcs.append(DirichletBC(Q4.sub(2), 0.0, model.ff, 4))
      
    if config['velocity']['boundaries'] == 'solution':
      self.bcs.append(DirichletBC(Q4.sub(0), model.u, model.ff, 4))
      self.bcs.append(DirichletBC(Q4.sub(1), model.v, model.ff, 4))
      self.bcs.append(DirichletBC(Q4.sub(2), model.w, model.ff, 4))
      
    if config['velocity']['boundaries'] == 'user_defined':
      u_t = config['velocity']['u_lat_boundary']
      v_t = config['velocity']['v_lat_boundary']
      self.bcs.append(DirichletBC(Q4.sub(0), u_t, model.ff, 4))
      self.bcs.append(DirichletBC(Q4.sub(1), v_t, model.ff, 4))
      self.bcs.append(DirichletBC(Q4.sub(2), 0.0, model.ff, 4))
       
    # Solve the nonlinear equations via Newton's method
    if self.model.MPI_rank==0:
      s    = "::: solving full-stokes velocity :::"
      text = colored(s, 'cyan')
      print text
    solve(self.F == 0, model.U, bcs=self.bcs, J = self.J, 
          solver_parameters = self.newton_params)


class VelocityBP(object):
  r"""				
  This class solves the non-linear Blatter-Pattyn momentum balance, 
  given a possibly non-uniform temperature field.
  
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
	                attributes such as velocties, age, and surface climate
  
  This class uses a simplification of the full Stokes' functional by expressing
  vertical velocities in terms of horizontal ones through incompressibility
  and bed impenetrability constraints.
  	
  **Equations**
	
  +-------------------+---------------+---------------------------------------+
  |Equation           |Condition      | Formula                               |
  +===================+===============+=======================================+
  |Variational        |               |.. math::                              |
  |Principle          |               |   \mathcal{A}\left[\textbf{u}_{| |}   |
  |                   |               |   \right]=                            |
  |For Power          |               |   \int\limits_{\Omega}\frac{2n}{n+1}  |
  |Law                |               |   \eta\left(\dot{\epsilon}^2_1\right) |
  |Rheology           |None           |   \dot{\epsilon}^2_1 + \rho g         |
  |                   |               |   \textbf{u}_{| |}\cdot\nabla_{| |}S  |
  |                   |               |   \ d\Omega+\int\limits_{\Gamma_{B}}  |
  |                   |               |   \frac{\beta^{2}}{2}H^{r}\textbf     |
  |                   |               |   {u}_{| |}                           |
  |                   |               |   \cdot\textbf{u}_{| |}               |
  |                   |               |   \ d\Gamma                           |
  +-------------------+---------------+---------------------------------------+
  |Rate of            |None           |.. math::                              |
  |strain             |               |   \eta\left(\dot{\epsilon}^{2}\right)=|
  |tensor             |               |   b(T)\left[\dot{\epsilon}^{2}\right] |
  |                   |               |   ^{\frac{1-n}{2n}}                   |
  +-------------------+---------------+---------------------------------------+
  |Temperature        |Viscosity mode +.. math::                              |
  |Dependent          |is isothermal  |   b(T) = A_0^{\frac{-1}{n}}           |
  |Rate Factor        +---------------+---------------------------------------+
  |                   |Viscosity mode |Model dependent                        |
  |                   |is linear      |                                       |
  +                   +---------------+---------------------------------------+
  |                   |Viscosity mode |                                       |
  |                   |is full        |.. math::                              |
  |                   |               |   b(T) = \left[Ea(T)e^{-\frac{Q(T)    |
  |                   |               |   }{RT^*}}                            |
  |                   |               |   \right]^{\frac{-1}{n}}              |
  +-------------------+---------------+---------------------------------------+
  |Incompressibility  |.. math::      |.. math::                              |
  |                   |   w_b=\textbf |   w\left(u\right)=-\int\limits^{z}_{B}|
  |                   |   {u}_{| | b} |   \nabla_{| |}\textbf{u}_{| |}dz'     |
  |                   |   \cdot       |                                       |
  |                   |   \nabla_{| | |                                       |
  |                   |   }B          |                                       |
  +-------------------+---------------+---------------------------------------+
  
  **Terms**

  +-------------------+-------------------------------+-----------------------+
  |Equation Name      |Term                           | Description           |
  +===================+===============================+=======================+
  |Variational        |.. math::                      |*Viscous dissipation*  |
  |Principle          |                               |including              |
  |For Power Law      |                               |terms for strain rate  |
  |Rheology           |                               |dependent ice          |
  |                   |   \frac{2n}{n+1}\eta\left(    |viscosity and the      |
  |                   |   \dot{\epsilon}^2_1\right)   |strain rate            |
  |                   |   \dot{\epsilon}^2_1          |tensor, respectively   |
  |                   |                               |                       |
  |                   +-------------------------------+-----------------------+
  |                   |.. math::                      |*Graviataional         |
  |                   |   \rho g                      |potential* energy      |
  |                   |   \textbf{u}\cdot\nabla       |calculated using the   |
  |                   |   _{| |}S                     |density,               |
  |                   |                               |graviational force,    |
  |                   |                               |and horizontal         |
  |                   |                               |ice velocity dotted    |
  |                   |                               |into the               |
  |                   |                               |gradient of the        |
  |                   |                               |surface elevation of   |
  |                   |                               |the ice                |
  |                   +-------------------------------+-----------------------+
  |                   |.. math::                      |*Frictional head       |
  |                   |                               |dissipation*           |
  |                   |   \frac{\beta^{2}}{2}H^{r}    |including terms for    |
  |                   |   \textbf{u}_{| |}\cdot       |the basal              |
  |                   |   \textbf{u}_{| |}            |sliding coefficient,   |
  |                   |                               |ice thickness,         |
  |                   |                               |and the horizontal     |
  |                   |                               |ice velocity           |
  |                   |                               |dotted into itself     |
  +-------------------+-------------------------------+-----------------------+
  |Rate of            |.. math::                      |Temperature dependent  |
  |strain             |                               |rate factor,           |
  |tensor             |   b(T)\left[\dot{\epsilon}    |square of the second   |
  |                   |   ^{2}\right]                 |invarient of           |
  |                   |   ^{\frac{1-n}{2n}}           |the strain rate        |
  |                   |                               |tensor                 |
  |                   |                               |                       |
  +-------------------+-------------------------------+-----------------------+
  |Temperature        |.. math::                      |Enhancement factor     |
  |Dependent          |   E                           |                       |
  |Rate Factor        +-------------------------------+-----------------------+
  |                   |.. math::                      |Temperature            |
  |                   |                               |dependent parameters   |
  |                   |   a(T)                        |                       |
  |                   |   Q(T)                        |                       |
  |                   |                               |                       |
  |                   +-------------------------------+-----------------------+
  |                   |.. math::                      |Rate constant          |
  |                   |   R                           |                       |
  +                   +-------------------------------+-----------------------+
  |                   |.. math::                      |Temperature corrected  |
  |                   |                               |for melting            |
  |                   |   T^*                         |point dependence       |
  +-------------------+-------------------------------+-----------------------+
  """
  def __init__(self, model, config):
    """ 
    Here we set up the problem, and do all of the differentiation and
    memory allocation type stuff.
    """
    self.model    = model
    self.config   = config

    mesh          = model.mesh
    r             = config['velocity']['r']
    Q             = model.Q
    Q2            = model.Q2
    n             = model.n
    b             = model.b
    Tstar         = model.Tstar
    T             = model.T
    gamma         = model.gamma
    S             = model.S
    B             = model.B
    H             = S - B
    x             = model.x
    E             = model.E
    W             = model.W
    R             = model.R
    epsdot        = model.epsdot
    eps_reg       = model.eps_reg
    eta           = model.eta
    rho           = model.rho
    rho_w         = model.rho_w
    g             = model.g
    Vd            = model.Vd
    Pe            = model.Pe
    Sl            = model.Sl
    Pb            = model.Pb
    beta          = model.beta2

    # pressure boundary :
    class Depth(Expression):
      def eval(self, values, x):
        b         = model.B_ex(x[0], x[1], x[2])
        values[0] = abs(min(0, b))
    D = Depth(element=Q.ufl_element())
    N = FacetNormal(mesh)
    
    newton_params = config['velocity']['newton_params']
    A0            = config['velocity']['A0']

    # initialize the temperature depending on input type :
    if config['velocity']['use_T0']:
      model.assign_variable(T, config['velocity']['T0'])

    # initialize the bed friction coefficient :
    if config['velocity']['init_beta_from_U_ob']:
      U_ob   = config['velocity']['U_ob']
      U_mag  = sqrt(inner(U_ob, U_ob))
      S_mag  = sqrt(inner(grad(S), grad(S)))
      beta_0 = project(sqrt((rho*g*H*S_mag) / (H**r * U_mag + 0.1)), Q)
      beta_0_v               = beta_0.vector().array()
      beta_0_v[beta_0_v < 0] = DOLFIN_EPS
      model.assign_variable(beta, beta_0_v)
    else:
      model.assign_variable(beta, config['velocity']['beta2'])
   
    # initialize the enhancement factor :
    model.assign_variable(E, config['velocity']['E'])

    # Check if there are non-linear solver parameters defined.  If not, set 
    # them to dolfin's default.  The default is not likely to converge if 
    # thermomechanical coupling is used.
    if newton_params:
      self.newton_params = newton_params
    
    else:
      self.newton_params = NonlinearVariationalSolver.default_parameters()

    # Define a test function
    Phi      = TestFunction(Q2)

    # Define a trial function
    dU       = TrialFunction(Q2)
    model.U  = Function(Q2)
    U        = model.U 

    phi, psi = split(Phi)
    du,  dv  = split(dU)
    u,   v   = split(U)  # x,y velocity components

    # vertical velocity components :
    chi      = TestFunction(Q)
    dw       = TrialFunction(Q)

    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    ds       = model.ds  
    dSrf     = ds(2)         # surface
    dGnd     = ds(3)         # grounded bed
    dFlt     = ds(5)         # floating bed
    dFltS    = ds(6)         # marine terminating sides
    dBed     = dGnd + dFlt   # bed

    # Set the value of b, the temperature dependent ice hardness parameter,
    # using the most recently calculated temperature field, if expected.
    if   config['velocity']['viscosity_mode'] == 'isothermal':
      b     = A0**(-1/n)
      b_gnd = b
      b_shf = b
    
    elif config['velocity']['viscosity_mode'] == 'linear':
      b     = config['velocity']['b_linear']
      b_gnd = b
      b_shf = b
      n     = 1.0
    
    elif config['velocity']['viscosity_mode'] == 'shelf_control':
      b_shf   = config['velocity']['b_linear_shf']
      b_gnd   = config['velocity']['b_linear_gnd']
      b       = Function(Q)
      b.vector()[model.shf_dofs] = b_shf.vector()[model.shf_dofs]
      b.vector()[model.gnd_dofs] = b_gnd.vector()[model.gnd_dofs]
      n       = 1.0
    
    elif config['velocity']['viscosity_mode'] == 'full':
      # Define pressure corrected temperature
      Tstar = T + gamma * (S - x[2])
       
      # Define ice hardness parameterization :
      a_T   = conditional( lt(Tstar, 263.15), 1.1384496e-5, 5.45e10)
      Q_T   = conditional( lt(Tstar, 263.15), 6e4,13.9e4)
      b     = ( E * (a_T * (1 + 181.25*W)) * exp( -Q_T / (R * Tstar)) )**(-1/n)
      b_gnd = b
      b_shf = b
    
    else:
      print "Acceptable choices for 'viscosity_mode' are 'linear', " + \
            "'isothermal', or 'full'."

    # second invariant of the strain rate tensor squared :
    term     = + 0.5 * (u.dx(2)**2 + v.dx(2)**2 + (u.dx(1) + v.dx(0))**2) \
               +        u.dx(0)**2 + v.dx(1)**2 + (u.dx(0) + v.dx(1))**2
    epsdot   =   0.5 * term + eps_reg
    eta      =     b * epsdot**((1.0 - n) / (2*n))

    # 1) Viscous dissipation
    Vd_shf   = (2*n)/(n+1) * b_shf * epsdot**((n+1)/(2*n))
    Vd_gnd   = (2*n)/(n+1) * b_gnd * epsdot**((n+1)/(2*n))
    Vd       = (2*n)/(n+1) * b * epsdot**((n+1)/(2*n))

    # 2) Potential energy
    Pe       = rho * g * (u * S.dx(0) + v * S.dx(1))

    # 3) Dissipation by sliding
    Sl       = 0.5 * beta**2 * H**r * (u**2 + v**2)
    
    # 4) pressure boundary
    Pb       = (rho*g*H + rho_w*g*B) / H * (u + v) 

    # Variational principle
    A        = Vd_shf*dx_s + Vd_gnd*dx_g + Pe*dx + Sl*dGnd + Pb*dFltS

    # Calculate the first variation (the action) of the variational 
    # principle in the direction of the test function
    self.F   = derivative(A, U, Phi)

    # Calculate the first variation of the action (the Jacobian) in
    # the direction of a small perturbation in U
    self.J   = derivative(self.F, U, dU)
 
    self.w_R = (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx - \
               (u*B.dx(0) + v*B.dx(1) - dw)*chi*dBed
    
    # Set up linear solve for vertical velocity.
    self.aw = lhs(self.w_R)
    self.Lw = rhs(self.w_R)

    model.eta    = eta
    model.Vd_shf = Vd_shf
    model.Vd_gnd = Vd_gnd
    model.Pe     = Pe
    model.Sl     = Sl
    model.Pb     = Pb
    model.A      = A
    model.T      = T
    model.beta2  = beta
    model.E      = E
    model.u      = u
    model.v      = v

  def solve(self):
    """ 
    Perform the Newton solve of the first order equations 
    """
    model  = self.model
    config = self.config
    
    # List of boundary conditions
    self.bcs = []

    # Add any user defined boundary conditions    
    for i in range(len(model.boundary_values)) :
      # Get the value in the facet function for this boundary
      marker_val = model.boundary_values[i]
      # The u component of velocity on the boundary
      bound_u = model.boundary_u[i]
      # The v component of velocity on the boundary
      bound_v = model.boundary_v[i]
      
      # Add the Direchlet boundary condition
      self.bcs.append(DirichletBC(model.Q2.sub(0), bound_u, model.ff, marker_val))
      self.bcs.append(DirichletBC(model.Q2.sub(1), bound_v, model.ff, marker_val))
    
    # solve nonlinear system :
    if self.model.MPI_rank==0:
      s    = "::: solving BP horizontal velocity :::"
      text = colored(s, 'cyan')
      print text
    solve(self.F == 0, model.U, J = self.J,
          solver_parameters = self.newton_params, bcs = self.bcs)

    
    # solve for vertical velocity :
    if self.model.MPI_rank==0:
      s    = "::: solving BP vertical velocity :::"
      text = colored(s, 'cyan')
      print text
    solve(self.aw == self.Lw, model.w)
    

class Enthalpy(object):
  r""" 
  This class solves the internal energy balance (enthalpy) in steady state or 
  transient, and converts that solution to temperature and water content.

  Time stepping uses Crank-Nicholson, which is 2nd order accurate.
    
  :param model  : An instantiated 2D flowline ice :class:`~src.model.Model`
  :param config : Dictionary object containing information on physical 
                 	attributes such as velocties, age, and surface climate
	
  The enthalpy equation used in this class is a typical advection-diffusion 
  equation with a non-linear diffusivity

  :Enthalpy:
     .. math::
      \rho\left(\partial_t+\textbf{u}\cdot\nabla\right)H = 
      \rho\nabla\cdot\kappa\left(H\right)\nabla H + Q
		 
  +-------------------------+---------------------------------------------+
  |Term                     |Description                                  |
  +=========================+=============================================+
  |.. math::                |                                             |
  |   H                     |Enthalpy                                     |
  +-------------------------+---------------------------------------------+
  |.. math::                |                                             |
  |   \rho                  |Ice density                                  |
  +-------------------------+---------------------------------------------+
  |.. math::                |Strain heat generated by viscious dissipation|
  |   Q                     |given by the first term in the Stokes'       |
  |                         |functional                                   |
  +-------------------------+---------------------------------------------+
  |.. math::                |Ice velocity                                 |
  |   \textbf{u}            |                                             | 
  +-------------------------+---------------------------------------------+
  |.. math::                |Enthalpy dependent diffusivity               |
  |   \kappa                |                                             |
  |                         +--------------+------------------------------+
  |                         |if the ice is |.. math::                     |
  |                         |cold          |   \frac{k}{\rho C_p}         |
  |                         +--------------+------------------------------+
  |                         |if the ice is |.. math::                     |
  |                         |temperate     |   \frac{\nu}{\rho}           |	
  +-------------------------+--------------+------------------------------+
  |.. math::                |Thermal conductivity of cold ice             |
  |   k                     |                                             |
  +-------------------------+---------------------------------------------+
  |.. math::                |Heat capacity                                |
  |   C_p                   |                                             |
  +-------------------------+---------------------------------------------+
  |.. math::                |Diffusivity of enthalpy in temperate ice     |
  |   \nu                   |                                             |
  +-------------------------+---------------------------------------------+
  
  +-----------------------------------------------------------------------+	
  |Ice Definitions                                                        |
  +====================+==================================================+
  |Cold ice            |.. math::                                         |
  |                    |   \left(H-h_i\left(P\right)\right) < 0           |
  +--------------------+--------------------------------------------------+
  |Temperate ice       |.. math::                                         |
  |                    |   \left(H-h_i\left(P\right)\right) \geq 0        |
  +--------------------+--------------------------------------------------+

  +------------------------+----------------------------------------------+
  |Term                    |Definition                                    |
  +========================+==============================================+
  |.. math::               |Pressure melting point expressed in enthalpy  |
  |   h_i\left(P\right)=   |                                              |
  |   -L+C_w\left(273-     |                                              |
  |   \gamma P\right)      |                                              |
  +------------------------+----------------------------------------------+
  |.. math::               |Latent heat of fusion                         |
  |   L                    |                                              |
  +------------------------+----------------------------------------------+
  |.. math::               |Heat capacity of liquid water                 |
  |   C_w                  |                                              |
  +------------------------+----------------------------------------------+
  |.. math::               |Dependence of the melting point on pressure   |
  |   \gamma               |                                              |
  +------------------------+----------------------------------------------+
  |.. math::               |Pressure                                      |
  |   P                    |                                              |
  +------------------------+----------------------------------------------+
  
  **Stabilization**
  
  The enthalpy equation is hyperbolic and so the 
  standard centered Galerkin Finite Element method is non-optimal and 
  spurious oscillations can arise. In order to stabilize it, we apply 
  streamline upwind Petrov-Galerkin methods. 
  This consists of adding an additional diffusion term of the form
  
  :Term:
     .. math::
      \rho\nabla\cdot K\nabla H
      
  +--------------------------------+--------------------------------------+
  |Term                            |Description                           |
  +================================+======================================+
  |.. math::                       |Tensor valued diffusivity             |
  |   K_{ij} = \frac{\alpha h}{2}  |                                      |
  |   \frac{u_i u_j}{| |u| |}      |                                      |
  +--------------------------------+--------------------------------------+
  |.. math::                       |Taken to be equal to unity            |
  |   \alpha                       |                                      |
  +--------------------------------+--------------------------------------+
  |.. math::                       |Cell size metric                      |
  |   h                            |                                      |
  +--------------------------------+--------------------------------------+
  
  Alternatively, to weight the advective portion of the governing equation
  we can view this stabilization as using skewed finite element test 
  functions of the form
  
  :Equation:
     .. math::
      \hat{\phi} = \phi + \frac{\alpha h}{2}\frac{u_i u_j}{| |u| |}
      \cdot\nabla_{| |}\phi
  """
  def __init__(self, model, config):
    """ 
    Set up equation, memory allocation, etc. 
    """
    self.config = config
    self.model  = model

    T_surface   = config['enthalpy']['T_surface']
    q_geo       = config['enthalpy']['q_geo']
    r           = config['velocity']['r']

    mesh        = model.mesh
    Q           = model.Q
    Q2          = model.Q2
    H           = model.H
    H0          = model.H0
    n           = model.n
    b           = model.b
    Tstar       = model.Tstar
    T           = model.T
    T0          = model.T0
    Mb          = model.Mb
    h_i         = model.h_i
    L           = model.L
    C           = model.C
    C_w         = model.C_w
    T_w         = model.T_w
    gamma       = model.gamma
    S           = model.S
    B           = model.B
    x           = model.x
    E           = model.E
    W           = model.W
    R           = model.R
    epsdot      = model.epsdot
    eps_reg     = model.eps_reg
    eta         = model.eta
    rho         = model.rho
    g           = model.g
    beta2       = model.beta2
    u           = model.u
    v           = model.v
    w           = model.w
    cold        = model.cold
    kappa       = model.kappa
    k           = model.k
    Hhat        = model.Hhat
    uhat        = model.uhat
    vhat        = model.vhat
    what        = model.what
    mhat        = model.mhat
    ds          = model.ds
    dx          = model.dx
    dx_s        = dx(1)
    dx_g        = dx(0)
    dx          = dx(1) + dx(0) # entire internal
    
    # If we're not using the output of the surface climate model,
    #  set the surface temperature to the constant or array that 
    #  was passed in.
    if not config['enthalpy']['use_surface_climate']:
      model.assign_variable(model.T_surface, T_surface)

    # initialize basal heat term :
    model.assign_variable(model.q_geo, q_geo)
    
    q_geo     = model.q_geo
    T_surface = model.T_surface

    # Define test and trial functions       
    psi = TestFunction(Q)
    dH  = TrialFunction(Q)

    # Pressure melting point
    T0  = T_w - gamma * (S - x[2])

    # Pressure melting enthalpy
    h_i = -L + C_w * T0

    # For the following heat sources, note that they differ from the 
    # oft-published expressions, in that they are both multiplied by constants.
    # I think that this is the correct form, as they must be this way in order 
    # to conserve energy.  This also implies that heretofore, models have been 
    # overestimating frictional heat, and underestimating strain heat.

    # Frictional heating = tau_b*u = beta2*u*u
    q_friction = 0.5 * beta2 * (S - B)**r * (u**2 + v**2)

    # Strain heating = stress*strain
    Q_s = (2*n)/(n+1) * b * epsdot**((n+1)/(2*n))

    # Different diffusion coefficent values for temperate and cold ice.  This
    # nonlinearity enters as a part of the Picard iteration between velocity
    # and enthalpy
    cold.vector()[:] = 1.0

    # diffusion coefficient :
    kappa = cold * k/(rho*C)

    # configure the module to run in steady state :
    if config['mode'] == 'steady':
      try:
        U    = as_vector([u, v, w])
      except NameError:
        print "No velocity field found.  Defaulting to no velocity"
        U    = 0.0

      # necessary quantities for streamline upwinding :
      h      = 2 * CellSize(mesh)
      vnorm  = sqrt(dot(U, U) + 1e-1)

      # skewed test function :
      psihat = psi + h/(2*vnorm) * dot(U, grad(psi))

      # residual of model :
      self.F = + rho * dot(U, grad(dH)) * psihat * dx \
               + rho * kappa * dot(grad(psi), grad(dH)) * dx \
               - (q_geo + q_friction) * psihat * ds(3) \
               - Q_s * psihat * dx

      self.a = lhs(self.F)
      self.L = rhs(self.F)

    # configure the module to run in transient mode :
    elif config['mode'] == 'transient':
      dt = config['time_step']
    
      # Skewed test function.  Note that vertical velocity has 
      # the mesh velocity subtracted from it.
      U = as_vector([uhat, vhat, what - mhat])

      h      = 2 * CellSize(mesh)
      vnorm  = sqrt(dot(U,U) + 1e-1)
      psihat = psi + h/(2*vnorm) * dot(U, grad(psi))

      theta = 0.5
      # Crank Nicholson method
      Hmid = theta*dH + (1 - theta)*H0
      
      # implicit system (linearized) for enthalpy at time H_{n+1}
      self.F = + rho * (dH - H0) / dt * psi * dx \
               + rho * dot(U, grad(Hmid)) * psihat * dx \
               + rho * kappa * dot(grad(psi), grad(Hmid)) * dx \
               - (q_geo + q_friction) * psi * ds(3) \
               - Q_s * psi * dx

      self.a = lhs(self.F)
      self.L = rhs(self.F)

    kappa_melt = conditional( ge(H, h_i), 0, kappa)

    # Form representing the basal melt rate
    vec   = as_vector([B.dx(0), B.dx(1), -1])
    term  = q_geo - (rho * kappa_melt * dot(grad(H), vec))
    Mb    = (q_friction + term) / (L * rho)

    model.T_surface = T_surface
    model.q_geo     = q_geo
    model.T0        = T0
    model.h_i       = h_i
    model.cold      = cold
    model.kappa     = kappa
    self.Mb         = Mb        # need this to project after solving
     
  
  def solve(self, H0=None, Hhat=None, uhat=None, 
            vhat=None, what=None, mhat=None):
    r""" 
    Uses boundary conditions and the linear solver to solve for temperature
    and water content.
    
    :param H0     : Initial enthalpy
    :param Hhat   : Enthalpy expression
    :param uhat   : Horizontal velocity
    :param vhat   : Horizontal velocity perpendicular to :attr:`uhat`
    :param what   : Vertical velocity
    :param mhat   : Mesh velocity
  
    
    A Neumann boundary condition is imposed at the basal boundary.
    
    :Boundary Condition:
       .. math::
        \kappa\left(H\right)\nabla H\cdot\textbf{n} = q_g+q_f
        -M_b\rho L
        
    +----------------------------+-------------------------------------------+
    |Terms                       |Description                                |
    +============================+===========================================+
    |.. math::                   |Geothermal heat flux, assumed to be known  |
    |   q_g                      |                                           |
    +----------------------------+-------------------------------------------+
    |.. math::                   |Frictional heat generated by basal sliding |
    |   q_f                      |                                           |
    +----------------------------+-------------------------------------------+
    |.. math::                   |Basal melt rate                            |
    |   M_b                      |                                           |
    +----------------------------+-------------------------------------------+
    
    Since temperature is uniquely related to enthalpy, it can be extracted 
    using the following equations
  
    +-----------------------------------------------------------------------+
    |                                                                       |
    +=================+=================================+===================+
    |.. math::        |.. math::                        |If the ice is cold |
    |   T\left(H,P    |   C_{p}^{-1}\left(H-h_i\left(P  |                   |
    |   \right) =     |   \right)\right)+T_{m}(p)       |                   |
    |                 +---------------------------------+-------------------+
    |                 |.. math::                        |If the ice is      |
    |                 |   T_{m}                         |temperate          |
    +-----------------+---------------------------------+-------------------+
    
    Similarly, the water content can also be extracted using the following 
    equations
    
    +-----------------------------------------------------------------------+
    |                                                                       |
    +=================+=================================+===================+
    |.. math::        |.. math::                        |If the ice is cold |
    |   \omega\left(  |   0                             |                   |
    |   H,P\right)=   |                                 |                   |
    |                 +---------------------------------+-------------------+
    |                 |.. math::                        |If the ice is      |
    |                 |   \frac{H-h_i\left(P\right)}    |temperate          |
    |                 |   {L}                           |                   |
    +-----------------+---------------------------------+-------------------+
    
    +---------------------------+-------------------------------------------+
    |Term                       |Description                                |
    +===========================+===========================================+
    |.. math::                  |Temperature melting point expressed in     |
    |   T_{m}                   |enthalpy                                   |
    +---------------------------+-------------------------------------------+
    """
    model  = self.model
    config = self.config
    
    # Assign values for H0,u,w, and mesh velocity
    if H0 is not None:
      model.assign_variable(model.H0,   H0)
      model.assign_variable(model.Hhat, Hhat)
      model.assign_variable(model.uhat, uhat)
      model.assign_variable(model.vhat, vhat)
      model.assign_variable(model.what, what)
      model.assign_variable(model.mhat, mhat)
      
    lat_bc    = config['enthalpy']['lateral_boundaries']
    T_w       = model.T_w
    T0        = model.T0
    Q         = model.Q
    H         = model.H
    H0        = model.H0
    Hhat      = model.Hhat
    uhat      = model.uhat
    vhat      = model.vhat
    what      = model.what
    mhat      = model.mhat
    T_surface = model.T_surface
    H_surface = model.H_surface
    C         = model.C
    h_i       = model.h_i
    T         = model.T
    W         = model.W
    Mb        = self.Mb
    L         = model.L
    cold      = model.cold

    # Surface boundary condition
    H_surface = project( (T_surface - T0) * C + h_i )
    H_float   = project( (T_w - T0) * C + h_i )
    model.H_surface = H_surface
   
    # surface boundary condition : 
    self.bc_H = []
    self.bc_H.append( DirichletBC(Q, H_surface, model.ff, 2) )
    
    # apply T_w conditions of portion of ice in contact with water :
    if model.mask != None:
      self.bc_H.append( DirichletBC(Q, H_float, model.ff, 5) )
   
    # apply lateral boundaries if desired : 
    if config['enthalpy']['lateral_boundaries'] is not None:
      self.bc_H.append( DirichletBC(Q, lat_bc, model.ff, 4) )
      
    # solve the linear equation for enthalpy :
    if self.model.MPI_rank==0:
      s    = "::: solving enthalpy :::"
      text = colored(s, 'cyan')
      print text
    solve(self.a == self.L, model.H, self.bc_H, 
          solver_parameters = {"linear_solver": "lu"})

    # Convert enthalpy values to temperatures and water contents
    T0_n  = project(T0,  Q)
    h_i_n = project(h_i, Q)

    # Calculate temperature
    T_n  = project( ((H - h_i_n) / C + T0_n), Q)
    W_n  = project( ((H - h_i_n) / L),        Q)
    Mb_n = project( Mb,                       Q)

    # update temperature (Adjust for polythermal stuff) :
    Ta = T_n.vector().array()
    Ts = T0_n.vector().array()
    #cold.vector().set_local((Ts > Ta).astype('float'))
    Ta[Ta > Ts] = Ts[Ta > Ts]
    model.assign_variable(T, Ta)

    # update water content :
    WW = W_n.vector().array()
    WW[WW < 0]    = 0
    WW[WW > 0.01] = 0.01
    model.assign_variable(W, WW)

    # update melt-rate :
    model.assign_variable(model.Mb, Mb_n)


class FreeSurface(object):
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
    model.assign_variable(self.ahat, model.smb) 
    model.assign_variable(self.uhat, model.u) 
    model.assign_variable(self.vhat, model.v) 
    model.assign_variable(self.what, model.w) 

    m = assemble(self.mass_matrix,      keep_diagonal=True)
    r = assemble(self.stiffness_matrix, keep_diagonal=True)

    if self.model.MPI_rank==0:
      s    = "::: solving free-surface :::"
      text = colored(s, 'cyan')
      print text
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
      model.print_min_max(r, 'D')

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

class AdjointVelocityBP(object):
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
    self.model  = model
    self.config = config

    # Adjoint variable in trial function form
    Q         = model.Q
    Vd_shf    = model.Vd_shf
    Vd_gnd    = model.Vd_gnd
    Pe        = model.Pe
    Sl        = model.Sl
    Pb        = model.Pb
    Pc        = model.Pc
    Lsq       = model.Lsq
    Nc        = model.Nc
    U         = model.U
    u_o       = model.u_o
    v_o       = model.v_o
    adot      = model.adot
    ds        = model.ds
    S         = model.S
    
    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    dSrf     = ds(2)         # surface
    dGnd     = ds(3)         # grounded bed 
    dFlt     = ds(5)         # floating bed
    dFltS    = ds(6)         # marine terminating sides
    dBed     = dGnd + dFlt   # bed


    control = config['adjoint']['control_variable']
    alpha   = config['adjoint']['alpha']

    if config['velocity']['approximation'] == 'fo':
      Q_adj   = model.Q2
      A       = Vd_shf*dx_s + Vd_gnd*dx_g + Pe*dx + Sl*dGnd + Pb*dFltS
    elif config['velocity']['approximation'] == 'stokes':
      Q_adj   = model.Q4
      A       = (Vd + Pe + Pc + Lsq)*dx + Sl*dGnd + Nc*dGnd

    L         = TrialFunction(Q_adj)
    Phi       = TestFunction(Q_adj)
    model.Lam = Function(Q_adj)

    rho       = TestFunction(Q)

    # Derivative, with trial function L.  This is the BP equations in weak form
    # multiplied by L and integrated by parts
    F_adjoint = derivative(A, U, L)

    # form regularization term 'R' :
    N = FacetNormal(model.mesh)
    for a,c in zip(alpha,control):
      t = Constant(0.5)
      if isinstance(a, (float,int)):
        a = Constant(a)
      if config['adjoint']['regularization_type'] == 'TV':
        R = a * t * sqrt(   (c.dx(0)*N[2] - c.dx(1)*N[0])**2 \
                          + (c.dx(1)*N[2] - c.dx(2)*N[1])**2 + 1e-3) * dGnd
      elif config['adjoint']['regularization_type'] == 'Tikhonov':
        R = a * t * (   (c.dx(0)*N[2] - c.dx(1)*N[0])**2 \
                      + (c.dx(1)*N[2] - c.dx(2)*N[1])**2) * dGnd
      else:
        print   "Valid regularizations are 'TV' and 'Tikhonov';" + \
              + " defaulting to no regularization."
        R = Constant(0.0) * dGnd
    
    # Objective function.  This is a least squares on the surface plus a 
    # regularization term penalizing wiggles in beta2
    if config['adjoint']['objective_function'] == 'logarithmic':
      a      = Constant(0.5)
      self.I = a * ln( (sqrt(U[0]**2 + U[1]**2) + 1.0) / \
                       (sqrt( u_o**2 +  v_o**2) + 1.0))**2 * dSrf + R
    
    elif config['adjoint']['objective_function'] == 'kinematic':
      a      = Constant(0.5)
      self.I = a * (+ U[0]*S.dx(0) + U[1]*S.dx(1) \
                    - (U[2] + adot))**2 * dSrf + R

    elif config['adjoint']['objective_function'] == 'linear':
      a      = Constant(0.5)
      self.I = a * ((U[0] - u_o)**2 + (U[1] - v_o)**2) * dSrf + R
    
    elif config['adjoint']['objective_function'] == 'log_lin_hybrid':
      g1     = Constant(0.5 * config['adjoint']['gamma1'])
      g2     = Constant(0.5 * config['adjoint']['gamma2'])
      self.I = + g1 * ((U[0] - u_o)**2 + (U[1] - v_o)**2) * dSrf \
               + g2 * ln( (sqrt(U[0]**2 + U[1]**2) + 1.0) / \
                          (sqrt( u_o**2 +  v_o**2) + 1.0))**2 * dSrf \
               + R

    else:
      print   "adjoint objection function may be 'linear', 'logarithmic', or" \
            + " 'kinematic'."
      exit(1)
    
    # Objective function constrained to obey the forward model
    I_adjoint  = self.I + F_adjoint

    # Gradient of this with respect to u in the direction of a test 
    # function yields a bilinear residual which, when solved yields the 
    # value of the adjoint variable
    self.dI    = derivative(I_adjoint, U, Phi)

    # Instead of treating the Lagrange multiplier as a trial function, treat 
    # it as a function.
    F_gradient = derivative(A, U, model.Lam)

    # This is a scalar quantity when discretized, as it contains no test or 
    # trial functions
    I_gradient = self.I + F_gradient

    # Differentiation wrt to the control variable in the direction of a test 
    # function yields a vector.  Assembly of this vector yields dJ/dbeta2
    self.J = []
    for c in control:
      self.J.append(derivative(I_gradient, c, rho))

  def solve(self):
    """
    Solves the bilinear residual created by differenciation of the 
    variational principle in combination with an objective function.
    """
    A = assemble(lhs(self.dI))
    l = assemble(rhs(self.dI))

    if self.model.MPI_rank==0:
      s    = "::: solving adjoint BP velocity :::"
      text = colored(s, 'cyan')
      print text
    solve(A, self.model.Lam.vector(), l)
    

class SurfaceClimate(object):

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
    if self.model.MPI_rank==0:
      s    = "::: solving surface climate :::"
      text = colored(s, 'cyan')
      print text
    model  = self.model
    config = self.config

    T_ma  = config['surface_climate']['T_ma']
    T_w   = model.T_w
    S     = model.S.vector().array()
    lat   = model.lat.vector().array()
    
    # Apply the lapse rate to the surface boundary condition
    model.assign_variable(model.T_surface, T_ma(S, lat) + T_w)


class Age(object):
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
    self.model  = model
    self.config = config

    # Trial and test
    a   = TrialFunction(model.Q)
    phi = TestFunction(model.Q)

    # Steady state
    if config['mode'] == 'steady':
      # SUPG method :
      h      = CellSize(model.mesh)
      U      = as_vector([model.u, model.v, model.w])
      vnorm  = sqrt(dot(U,U) + 1e-10)
      phihat = phi + h/(2*vnorm) * dot(U,grad(phi))
      
      # Residual 
      R = dot(U,grad(a)) - 1.0

      # Weak form of residual
      self.F = R * phihat * dx

    else:
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
      h      = CellSize(model.mesh)
      U      = as_vector([uhat, vhat, what-mhat])
      vnorm  = sqrt(dot(U,U) + 1e-10)
      phihat = phi + h/(2*vnorm)*dot(U,grad(phi))

      # Midpoint value of age for Crank-Nicholson
      a_mid = 0.5*(a + self.ahat)
      
      # Weak form of time dependent residual
      self.F = + (a - a0)/dt * phi * dx \
               + dot(U, grad(a_mid)) * phihat * dx \
               - 1.0 * phihat * dx

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
   
    if config['age']['use_smb_for_ela']:
      self.bc_age = DirichletBC(model.Q, 0.0, model.ff_acc, 1)
    
    else:
      def above_ela(x,on_boundary):
        return x[2] > config['age']['ela'] and on_boundary
      self.bc_age = DirichletBC(model.Q, 0.0, above_ela)

    # Solve!
    if self.model.MPI_rank==0:
      s    = "::: solving age :::"
      text = colored(s, 'cyan')
      print text
    solve(lhs(self.F) == rhs(self.F), model.age, self.bc_age)


class VelocityBalance(object):
  
  def __init__(self, model, config):
    
    self.model  = model
    self.config = config
    
    kappa       = config['balance_velocity']['kappa']
    smb         = config['balance_velocity']['smb']
    g           = model.g
    rho         = model.rho

    flat_mesh   = model.flat_mesh
    Q_flat      = model.Q_flat
    B           = model.B.vector().get_local()
    S           = model.S.vector().get_local()
    dSdx        = model.dSdx
    dSdy        = model.dSdy
    Ub          = model.Ub

    phi         = TestFunction(Q_flat)
    dU          = TrialFunction(Q_flat)
                
    Nx          = TrialFunction(Q_flat)
    Ny          = TrialFunction(Q_flat)
    H_          = Function(Q_flat)
    S_          = Function(Q_flat)
    smb_        = project(smb, Q_flat)
    
    ds          = model.ds
   
    model.assign_variable(H_, S-B) 
    model.assign_variable(S_, S) 

    R_dSdx = + Nx * phi * ds(2) \
             - rho * g * H_ * S_.dx(0) * phi * ds(2) \
             + (l*H_)**2 * (phi.dx(0)*Nx.dx(0) + phi.dx(1)*Nx.dx(1)) * ds(2)
    R_dSdy = + Ny * phi * ds(2) \
             - rho * g * H_ * S_.dx(1) * phi*ds(2) \
             + (l*H_)**2 * (phi.dx(0)*Ny.dx(0) + phi.dx(1)*Ny.dx(1)) * ds(2)

    slope  = sqrt(dSdx**2 + dSdy**2) + 1e-5
    dS     = as_vector([-dSdx/slope, -dSdy/slope])
    
    def inside(x,on_boundary):
      return on_boundary
    
    # SUPG method :
    h      = CellSize(flat_mesh)
    U_eff  = sqrt(dot(dS*H_, dS*H_))
    tau    = h/(2.0 * U_eff)
    
    term1  = phi + tau*(Dx(H_*phi*dS[0], 0) + Dx(H_*phi*dS[1], 1))
    term2  = Dx(dU*dS[0]*H_, 0) + Dx(dU*dS[1]*H_, 1) - smb_
    dI     = term1 * term2 * ds(2)
    
    self.R_dSdx = R_dSdx
    self.R_dSdy = R_dSdy
    self.dI     = dI
    self.dS     = dS

  def solve(self):
    Ub   = self.model.Ub
    dSdx = self.model.dSdx
    dSdy = self.model.dSdy

    a_x  = assemble(lhs(self.R_dSdx))
    a_x.ident_zeros()
    L_x  = assemble(rhs(self.R_dSdx))

    a_y  = assemble(lhs(self.R_dSdy))
    a_y.ident_zeros()
    L_y  = assemble(rhs(self.R_dSdy))

    solve(a_x, dSdx.vector(), L_x)
    solve(a_y, dSdy.vector(), L_y)

    a_U  = assemble(lhs(self.dI))
    a_U.ident_zeros()
    L_U  = assemble(rhs(self.dI))

    solve(a_U, U.vector(), L_U)
    u_b = project(Ub * self.dS[0])
    v_b = project(Ub * self.dS[1])
    self.model.assign_variable(self.model.u_balance, u_b.vector())
    self.model.assign_variable(self.model.v_balance, v_b.vector())
    

class VelocityBalance_2(object):

  def __init__(self, mesh, H, S, adot, l,dhdt=0.0, Uobs=None,Uobs_mask=None,N_data = None,NO_DATA=-9999,alpha=[0.0,0.0,0.0,0.]):

    set_log_level(PROGRESS)
    
    Q = FunctionSpace(mesh, "CG", 1)
    
    # Physical constants
    rho = 911
    g = 9.81

    if Uobs:
      pass
    else:
      Uobs = Function(Q)

    # solution and trial functions :
    Ubmag = Function(Q)
    dUbmag = TrialFunction(Q)

    lamda = Function(Q)
    dlamda = TrialFunction(Q)
    
    # solve for dhdx,dhdy with appropriate smoothing :
    dSdx = Function(Q)
    dSdy = Function(Q)
    dSdx2 = Function(Q)
    dSdy2 = Function(Q)
    phi = TestFunction(Q)

    Nx = TrialFunction(Q)
    Ny = TrialFunction(Q)
    
    # smoothing radius :
    kappa = Function(Q)
    kappa.vector()[:] = l
    
    R_dSdx = + (Nx*phi - rho*g*H*S.dx(0) * phi \
             + (kappa*H)**2 * dot(grad(phi), grad(Nx))) * dx
    R_dSdy = + (Ny*phi - rho*g*H*S.dx(1) * phi \
             + (kappa*H)**2 * dot(grad(phi), grad(Ny))) * dx
    
    solve(lhs(R_dSdx) == rhs(R_dSdx), dSdx)
    solve(lhs(R_dSdy) == rhs(R_dSdy), dSdy)

    # Replace values of slope that are known
    # I don't think this works in parallel, but it works for now...
    # Note I did try conditionals here, to bad effect!
    # Perhaps a DG space would have been better.
    # To make parallel, try:
    # remove .array() and replace with .get_local() and .set_local()
    if N_data:
        dSdx.vector().array()[N_data[0].vector().array() != NO_DATA] =\
            N_data[0].vector().array()[N_data[0].vector().array() != NO_DATA]
        dSdy.vector().array()[N_data[1].vector().array() != NO_DATA] =\
            N_data[1].vector().array()[N_data[1].vector().array() != NO_DATA]

    # Smoothing the merged results, using the same approach as before
    kappa = Function(Q)
    kappa.vector()[:] = 2.5  # Hard coded for development change later
    
    R_dSdx = + (Nx*phi - dSdx * phi \
             + (kappa*H)**2 * dot(grad(phi), grad(Nx))) * dx
    R_dSdy = + (Ny*phi - dSdy * phi \
             + (kappa*H)**2 * dot(grad(phi), grad(Ny))) * dx
    
    solve(lhs(R_dSdx) == rhs(R_dSdx), dSdx2)
    solve(lhs(R_dSdy) == rhs(R_dSdy), dSdy2)

    slope = project(sqrt(dSdx2**2 + dSdy2**2) + 1e-10, Q)

    dS = as_vector([project(-dSdx2 / slope, Q),
                        project(-dSdy2 / slope, Q)])
   
    def inside(x,on_boundary):
      return on_boundary
       
    dbc = DirichletBC(Q, 0.0, inside)
    
    # test function :
    phi = TestFunction(Q)
    
    cellh = CellSize(mesh)
    U_eff = sqrt( dot(dS * H, dS * H) + 1e-10 )
    tau = cellh / (2 * U_eff)

    adot_0 = adot.copy()

    if Uobs_mask:
        dx_masked = Measure('dx')[Uobs_mask]
        self.I = ln(abs(Ubmag+1.)/abs(Uobs+1.))**2*dx_masked(1) + alpha[0]*dot(grad(Uobs),grad(Uobs))*dx + alpha[1]*dot(grad(adot-adot_0),grad(adot-adot_0))*dx + alpha[2]*dot(grad(H),grad(H))*dx+ alpha[3]*dot(grad(dS[1]),grad(dS[1]))*dx
        #self.I = (Ubmag - Uobs)**2*dx_masked(1) + alpha[0]*dot(grad(Uobs),grad(Uobs))*dx + alpha[1]*dot(grad(adot-adot_0),grad(adot-adot_0))*dx + alpha[2]*dot(grad(H),grad(H))*dx
    else:
        self.I = ln(abs(Ubmag+1.)/abs(Uobs+1.))**2*dx + alpha[0]*dot(grad(Uobs),grad(Uobs))*dx + alpha[1]*dot(grad(adot-adot_0),grad(adot - adot_0))*dx + alpha[2]*dot(grad(H),grad(H))*dx
    
    self.forward_model = (phi + tau*div(H*dS*phi)) * (div(dUbmag*dS*H) - adot + dhdt) * dx

    self.adjoint_model = derivative(self.I,Ubmag,phi) + ((dlamda + tau*div(dlamda*dS*H))*(div(phi*dS*H)) )*dx

    self.I += (lamda + tau*div(H*dS*lamda)) * (div(Ubmag*dS*H) - adot + dhdt) * dx

    # Switch to use AD for the gradients:
    self.g_Uobs = derivative(self.I,Uobs,phi)
    self.g_adot = derivative(self.I,adot,phi)
    self.g_H    = derivative(self.I,H,phi)
    self.g_N    = derivative(self.I,dS[1],phi)

    # Gradients computed by hand.
    #self.g_adot = -(lamda + tau*div(lamda*dS*H))*phi*dx + 2.*alpha[1]*dot(grad(adot),grad(phi))*dx
    #self.g_H = (lamda + tau*div(lamda*dS*H))*div(Ubmag*dS*phi)*dx + tau*div(lamda*dS*phi)*(div(Ubmag*dS*H) - adot + dhdt)*dx + 2.*alpha[2]*dot(grad(H),grad(phi))*dx


    self.H = H
    self.S = S
    self.dS = dS
    self.adot = adot
    self.R_dSdx = R_dSdx
    self.R_dSdy = R_dSdy
    self.dSdx = dSdx
    self.dSdy = dSdy
    self.Ubmag = Ubmag
    self.lamda = lamda
    self.dbc = dbc
    self.slope = slope
    self.residual = Ubmag*div(dS*H) - adot
    self.residual = project(self.residual, Q)
    self.Uobs = Uobs
    self.dx_masked = dx_masked
    self.Q = Q
    self.signs = np.sign(self.dS[0].vector().array().copy())
    self.update_velocity_directions()

  def update_velocity_directions(self):
      ny = self.dS[1].vector().array().copy()

      # These protect against NaNs in the sqrt below
      ny[ny>1]  =  1.
      ny[ny<-1] = -1.
      nx = self.signs * np.sqrt(1-ny**2)

      # Maybe set_local is more parallel safe
      self.dS[0].vector().set_local(nx)
      self.dS[1].vector().set_local(ny)
      self.dS[0].vector().apply('insert')
      self.dS[1].vector().apply('insert')


  def solve_forward(self):
    # solve linear problem :
    self.update_velocity_directions()
    solve(lhs(self.forward_model) == rhs(self.forward_model), self.Ubmag)
    self.Ubmag.vector()[self.Ubmag.vector().array()<0] = 0.0

  def solve_adjoint(self):
    self.update_velocity_directions()
    self.Uobs.vector()[self.Uobs.vector().array()<0] = 0.0
    solve(lhs(self.adjoint_model) == rhs(self.adjoint_model), self.lamda)
   
  def get_gradient(self):
    gU = assemble(self.g_Uobs)
    gH = assemble(self.g_H)
    gN = assemble(self.g_N)
    ga = assemble(self.g_adot)
    #return ((gU.array() / linalg.norm(gU.array()) , ga.array() / linalg.norm(ga.array()),\
    #         gH.array() / linalg.norm(gH.array()) , gN.array() / linalg.norm(gN.array())))

    return ((gU.array() , ga.array() ,\
             gH.array() , gN.array() ))


class StokesBalance(object):

  def __init__(self, model, config):
    """
    """
    self.model  = model
    self.config = config

    Q       = model.Q
    u       = model.u
    v       = model.v
    w       = model.w
    S       = model.S
    B       = model.B
    H       = S - B
    eta     = model.eta
    beta2   = model.beta2
    
    # get the values at the bed :
    beta2_e = model.beta2_e
    u_b_e   = model.u_b_e
    v_b_e   = model.v_b_e
    
    # vertically average :
    etabar  = model.etabar
    ubar    = model.ubar
    vbar    = model.vbar
    ubar_d  = model.ubar_d
    vbar_d  = model.vbar_d

    # create functions used to solve for velocity :
    V        = MixedFunctionSpace([Q,Q])
    dU       = TrialFunction(V)
    du, dv   = split(dU)
    Phi      = TestFunction(V)
    phi, psi = split(Phi)
    U_s      = Function(V)
    u_s, v_s = split(U_s)
    
    #===========================================================================
    # form the stokes equations in the normal direction (n) and tangential 
    # direction (t) in relation to the stress-tensor :
    U_n  = model.normalize_vector(as_vector([ubar,vbar]), Q)
    u_n  = U_n[0]
    v_n  = U_n[1]
    U_n  = as_vector([u_n,  v_n,  0])
    U_t  = as_vector([v_n, -u_n,  0])
    U    = as_vector([du,   dv,   0])
    Ubar = as_vector([ubar, vbar, 0])

    # directional derivatives :
    uhat     = dot(U, U_n)
    vhat     = dot(U, U_t)
    graduhat = grad(uhat)
    gradvhat = grad(vhat)
    dudn     = dot(graduhat, U_n)
    dvdn     = dot(gradvhat, U_n)
    dudt     = dot(graduhat, U_t)
    dvdt     = dot(gradvhat, U_t)
    
    # integration by parts directional derivative terms :
    gradphi = grad(phi)
    dphidn  = dot(gradphi, U_n)
    dphidt  = dot(gradphi, U_t)
    gradpsi = grad(psi)
    dpsidn  = dot(gradpsi, U_n)
    dpsidt  = dot(gradpsi, U_t)

    # driving stres :
    tau_d   = model.tau_d
    
    # calc basal drag : 
    #u_c     = ubar - u_b_e
    #v_c     = vbar - v_b_e
    u_c     = ubar - ubar_d
    v_c     = vbar - vbar_d
    tau_b_u = beta2_e * H * (du - u_c)
    tau_b_v = beta2_e * H * (dv - v_c)
    tau_b   = as_vector([tau_b_u, tau_b_v, 0])

    # dot product of stress with the direction along (n) and across (t) flow :
    tau_bn = phi * dot(tau_b, U_n) * dx
    tau_bt = psi * dot(tau_b, U_t) * dx
    tau_dn = phi * dot(tau_d, U_n) * dx
    tau_dt = psi * dot(tau_d, U_t) * dx

    # stokes equation weak form in normal dir. (n) and tangent dir. (t) :
    tau_nn = - dphidn * H * etabar * (4*dudn + 2*dvdt) * dx
    tau_nt = - dphidt * H * etabar * (  dudt +   dvdn) * dx
    tau_tn = - dpsidn * H * etabar * (  dudt +   dvdn) * dx
    tau_tt = - dpsidt * H * etabar * (4*dvdt + 2*dudn) * dx
  
    # form residual in mixed space :
    rn = tau_nn + tau_nt - tau_bn - tau_dn
    rt = tau_tn + tau_tt - tau_bt - tau_dt
    r  = rn + rt

    # make the variables available to solve :
    self.Q      = Q
    self.r      = r
    self.U_s    = U_s
    model.ubar  = u_s
    model.vbar  = v_s
    model.tau_b = tau_b
    
  def solve(self):
    """
    """
    model = self.model

    if self.model.MPI_rank==0:
      s    = "::: solving 'stokes-balance' for ubar, vbar :::"
      text = colored(s, 'cyan')
      print text
    solve(lhs(self.r) == rhs(self.r), self.U_s)

  def component_stress_stokes(self):  
    """
    """
    if self.model.MPI_rank==0:
      s    = "solving 'stokes-balance' for stress terms :::" 
      text = colored(s, 'cyan')
      print text
    model = self.model

    outpath = self.config['output_path']
    Q       = self.Q
    S       = model.S
    B       = model.B
    H       = S - B
    etabar  = model.etabar
    
    #===========================================================================
    # form the stokes equations in the normal direction (n) and tangential 
    # direction (t) in relation to the stress-tensor :
    u_s = project(model.ubar, Q)
    v_s = project(model.vbar, Q)
    
    U   = model.normalize_vector(as_vector([u_s, v_s]), Q)
    u_n = U[0]
    v_n = U[1]
    U   = as_vector([u_s, v_s, 0])
    U_n = as_vector([u_n, v_n, 0])
    U_t = as_vector([v_n,-u_n, 0])

    # directional derivatives :
    uhat     = dot(U, U_n)
    vhat     = dot(U, U_t)
    graduhat = grad(uhat)
    gradvhat = grad(vhat)
    dudn     = dot(graduhat, U_n)
    dvdn     = dot(gradvhat, U_n)
    dudt     = dot(graduhat, U_t)
    dvdt     = dot(gradvhat, U_t)

    # get driving stress and basal drag : 
    tau_d = model.tau_d
    tau_b = model.tau_b
    
    # trial and test functions for linear solve :
    phi   = TestFunction(Q)
    dtau  = TrialFunction(Q)
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # integration by parts directional derivative terms :
    gradphi = grad(phi)
    dphidn  = dot(gradphi, U_n)
    dphidt  = dot(gradphi, U_t)
    
    # stokes equation weak form in normal dir. (n) and tangent dir. (t) :
    tau_nn = - dphidn * H * etabar * (4*dudn + 2*dvdt) * dx
    tau_nt = - dphidt * H * etabar * (  dudt +   dvdn) * dx
    tau_tn = - dphidn * H * etabar * (  dudt +   dvdn) * dx
    tau_tt = - dphidt * H * etabar * (4*dvdt + 2*dudn) * dx
    
    # dot product of stress with the direction along (n) and across (t) flow :
    tau_bn = phi * dot(tau_b, U_n) * dx
    tau_dn = phi * dot(tau_d, U_n) * dx
    tau_bt = phi * dot(tau_b, U_t) * dx
    tau_dt = phi * dot(tau_d, U_t) * dx
    
    # the residuals :
    tau_totn = tau_nn + tau_tn - tau_bn - tau_dn
    tau_tott = tau_nt + tau_tt - tau_bt - tau_dt

    # assemble the vectors :
    tau_nn_v   = assemble(tau_nn)
    tau_nt_v   = assemble(tau_nt)
    tau_tn_v   = assemble(tau_tn)
    tau_tt_v   = assemble(tau_tt)
    tau_totn_v = assemble(tau_totn)
    tau_tott_v = assemble(tau_tott)
    
    # solution functions :
    tau_nn   = Function(Q)
    tau_nt   = Function(Q)
    tau_tn   = Function(Q)
    tau_tt   = Function(Q)
    tau_totn = Function(Q)
    tau_tott = Function(Q)
    
    # solve the linear system :
    solve(M, tau_nn.vector(),   tau_nn_v)
    solve(M, tau_nt.vector(),   tau_nt_v)
    solve(M, tau_tn.vector(),   tau_tn_v)
    solve(M, tau_tt.vector(),   tau_tt_v)
    solve(M, tau_totn.vector(), tau_totn_v)
    solve(M, tau_tott.vector(), tau_tott_v)

    # give the stress balance terms :
    tau_bn = project(dot(tau_b, U_n))
    tau_dn = project(dot(tau_d, U_n))

    # output the files to the specified directory :
    File(outpath + 'tau_dn.pvd')   << tau_dn
    File(outpath + 'tau_bn.pvd')   << tau_bn
    File(outpath + 'tau_nn.pvd')   << tau_nn
    File(outpath + 'tau_nt.pvd')   << tau_nt
    File(outpath + 'tau_totn.pvd') << tau_totn
    File(outpath + 'tau_tott.pvd') << tau_tott
    File(outpath + 'u_s.pvd')      << u_s
    File(outpath + 'v_s.pvd')      << v_s

    # attach the results to the model :
    model.tau_nn = tau_nn
    model.tau_nt = tau_nt
    model.tau_tn = tau_tn
    model.tau_tt = tau_tt
    model.tau_bn = tau_bn
    model.tau_dn = tau_dn
    model.tau_totn = tau_totn
    model.tau_tott = tau_tott
    model.u_s      = u_s
    model.v_s      = v_s


class StokesBalance3D(object):

  def __init__(self, model, config):
    """
    """
    self.model  = model
    self.config = config

    Q       = model.Q
    u       = model.u
    v       = model.v
    w       = model.w
    S       = model.S
    B       = model.B
    H       = S - B
    beta2   = model.beta2
    eta     = model.eta
    rho     = model.rho
    g       = model.g
    ds      = model.ds  
    dSrf    = ds(2)        # surface
    dGnd    = ds(3)        # grounded bed 
    dFlt    = ds(5)        # floating bed
    dFltS   = ds(6)        # marine terminating sides
    dBed    = dGnd + dFlt  # bed
    
    # create functions used to solve for velocity :
    V        = MixedFunctionSpace([Q,Q])
    dU       = TrialFunction(V)
    du, dv   = split(dU)
    Phi      = TestFunction(V)
    phi, psi = split(Phi)
    U_s      = Function(V)
    u_s, v_s = split(U_s)
    
    #===========================================================================
    # form the stokes equations in the normal direction (n) and tangential 
    # direction (t) in relation to the stress-tensor :
    U_n  = model.normalize_vector(as_vector([u,v]), Q)
    u_n  = U_n[0]
    v_n  = U_n[1]
    U_n  = as_vector([u_n,  v_n,  0])
    U_t  = as_vector([v_n, -u_n,  0])
    U    = as_vector([du,   dv,   0])

    # directional derivatives :
    uhat     = dot(U, U_n)
    vhat     = dot(U, U_t)
    graduhat = grad(uhat)
    gradvhat = grad(vhat)
    dudn     = dot(graduhat, U_n)
    dvdn     = dot(gradvhat, U_n)
    dudt     = dot(graduhat, U_t)
    dvdt     = dot(gradvhat, U_t)
    dSdn     = dot(grad(S),  U_n)
    dSdt     = dot(grad(S),  U_t)
    
    # integration by parts directional derivative terms :
    gradphi = grad(phi)
    dphidn  = dot(gradphi, U_n)
    dphidt  = dot(gradphi, U_t)
    gradpsi = grad(psi)
    dpsidn  = dot(gradpsi, U_n)
    dpsidt  = dot(gradpsi, U_t)

    # driving stres :
    tau_d  = rho * g * grad(S)
    tau_dn = phi * dot(tau_d, U_n) * dx
    tau_dt = psi * dot(tau_d, U_t) * dx
    
    # calc basal drag : 
    tau_bn = - beta2 * uhat * H * phi * dBed
    tau_bt = - beta2 * vhat * H * psi * dBed

    # stokes equation weak form in normal dir. (n) and tangent dir. (t) :
    tau_nn = - dphidn * eta * (4*dudn + 2*dvdt) * dx
    tau_nt = - dphidt * eta * (  dudt +   dvdn) * dx
    tau_tn = - dpsidn * eta * (  dudt +   dvdn) * dx
    tau_tt = - dpsidt * eta * (4*dvdt + 2*dudn) * dx
  
    # form residual in mixed space :
    rn = tau_nn + tau_nt - tau_bn - tau_dn
    rt = tau_tn + tau_tt - tau_bt - tau_dt
    r  = rn + rt

    # make the variables available to solve :
    self.Q     = Q
    self.r     = r
    self.U_s   = U_s
    model.ubar = u_s
    model.vbar = v_s
    
  def solve(self):
    """
    """
    if self.model.MPI_rank==0:
      s    = "::: solving 3D 'stokes-balance' for ubar, vbar :::"
      text = colored(s, 'cyan')
      print text
    solve(lhs(self.r) == rhs(self.r), self.U_s)

  def component_stress_stokes(self):  
    """
    """
    model = self.model
    
    if model.MPI_rank==0:
      s    = "solving 3D 'stokes-balance' for stress terms :::" 
      text = colored(s, 'cyan')
      print text

    outpath = self.config['output_path']
    Q       = self.Q
    beta2   = model.beta2
    eta     = model.eta
    S       = model.S
    B       = model.B
    H       = S - B
    rho     = model.rho
    g       = model.g
    ds      = model.ds  
    dSrf    = ds(2)        # surface
    dGnd    = ds(3)        # grounded bed 
    dFlt    = ds(5)        # floating bed
    dFltS   = ds(6)        # marine terminating sides
    dBed    = dGnd + dFlt  # bed
    
    #===========================================================================
    # form the stokes equations in the normal direction (n) and tangential 
    # direction (t) in relation to the stress-tensor :
    u_s = project(model.u, Q)
    v_s = project(model.v, Q)
    
    U   = model.normalize_vector(as_vector([u_s, v_s]), Q)
    u_n = U[0]
    v_n = U[1]
    U   = as_vector([u_s, v_s, 0])
    U_n = as_vector([u_n, v_n, 0])
    U_t = as_vector([v_n,-u_n, 0])

    # directional derivatives :
    uhat     = dot(U, U_n)
    vhat     = dot(U, U_t)
    graduhat = grad(uhat)
    gradvhat = grad(vhat)
    dudn     = dot(graduhat, U_n)
    dvdn     = dot(gradvhat, U_n)
    dudt     = dot(graduhat, U_t)
    dvdt     = dot(gradvhat, U_t)

    # trial and test functions for linear solve :
    phi   = TestFunction(Q)
    dtau  = TrialFunction(Q)
    
    # integration by parts directional derivative terms :
    gradphi = grad(phi)
    dphidn  = dot(gradphi, U_n)
    dphidt  = dot(gradphi, U_t)

    # driving stres :
    tau_d  = rho * g * grad(S)
    tau_dn = phi * dot(tau_d, U_n) * dx
    tau_dt = phi * dot(tau_d, U_t) * dx
    
    # calc basal drag : 
    tau_bn = beta2 * uhat * H * phi * dBed
    tau_bt = beta2 * vhat * H * phi * dBed
    
    # stokes equation weak form in normal dir. (n) and tangent dir. (t) :
    tau_nn = - dphidn * eta * (4*dudn + 2*dvdt) * dx
    tau_nt = - dphidt * eta * (  dudt +   dvdn) * dx
    tau_tn = - dphidn * eta * (  dudt +   dvdn) * dx
    tau_tt = - dphidt * eta * (4*dvdt + 2*dudn) * dx
    
    # the residuals :
    tau_totn = tau_nn + tau_tn - tau_bn - tau_dn
    tau_tott = tau_nt + tau_tt - tau_bt - tau_dt
    
    # mass matrix :
    M = assemble(phi*dtau*dx)

    # assemble the vectors :
    tau_dn_v   = assemble(tau_dn)
    tau_dt_v   = assemble(tau_dt)
    tau_bn_v   = assemble(tau_bn)
    tau_bt_v   = assemble(tau_bt)
    tau_nn_v   = assemble(tau_nn)
    tau_nt_v   = assemble(tau_nt)
    tau_tn_v   = assemble(tau_tn)
    tau_tt_v   = assemble(tau_tt)
    tau_totn_v = assemble(tau_totn)
    tau_tott_v = assemble(tau_tott)
    
    # solution functions :
    tau_dn   = Function(Q)
    tau_dt   = Function(Q)
    tau_bn   = Function(Q)
    tau_bt   = Function(Q)
    tau_nn   = Function(Q)
    tau_nt   = Function(Q)
    tau_tn   = Function(Q)
    tau_tt   = Function(Q)
    tau_totn = Function(Q)
    tau_tott = Function(Q)
    
    # solve the linear system :
    solve(M, tau_dn.vector(),   tau_dn_v)
    solve(M, tau_dt.vector(),   tau_dt_v)
    #solve(M, tau_bn.vector(),   tau_bn_v)
    #solve(M, tau_bt.vector(),   tau_bt_v)
    solve(M, tau_nn.vector(),   tau_nn_v)
    solve(M, tau_nt.vector(),   tau_nt_v)
    solve(M, tau_tn.vector(),   tau_tn_v)
    solve(M, tau_tt.vector(),   tau_tt_v)
    #solve(M, tau_totn.vector(), tau_totn_v)
    #solve(M, tau_tott.vector(), tau_tott_v)

    # output the files to the specified directory :
    File(outpath + 'tau_dn.pvd')   << tau_dn
    File(outpath + 'tau_bn.pvd')   << tau_bn
    File(outpath + 'tau_nn.pvd')   << tau_nn
    File(outpath + 'tau_nt.pvd')   << tau_nt
    File(outpath + 'tau_totn.pvd') << tau_totn
    File(outpath + 'tau_tott.pvd') << tau_tott
    File(outpath + 'u_s.pvd')      << u_s
    File(outpath + 'v_s.pvd')      << v_s

    # attach the results to the model :
    model.tau_dn   = tau_dn
    model.tau_dt   = tau_dt
    model.tau_bn   = tau_bn
    model.tau_bt   = tau_bt
    model.tau_nn   = tau_nn
    model.tau_nt   = tau_nt
    model.tau_tn   = tau_tn
    model.tau_tt   = tau_tt
    model.tau_totn = tau_totn
    model.tau_tott = tau_tott
    model.u_s      = u_s
    model.v_s      = v_s

