from varglas.physics_new import Physics 
from varglas.io          import print_text, print_min_max
from varglas.d2model     import D2Model
from varglas.d3model     import D3Model
from varglas.helper      import VerticalBasis, VerticalFDBasis, \
                                VerticalIntegrator
from fenics         import *
from dolfin_adjoint import *

class Mass(Physics):
  """
  Abstract class outlines the structure of a mass conservation.
  """

  def __new__(self, model, *args, **kwargs):
    """
    Creates and returns a new Mass object.
    """
    instance = Physics.__new__(self, model)
    return instance
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'white'
  
  def solve(self, annotate=True, params=None):
    """ 
    Solve the conservation of mass equation for a free-surface evolution.
    """
    raiseNotDefined()


class FreeSurface(Mass):
  """
  """  
  def __init_(self, model):
    """
    """
    s    = "::: INITIALIZING FREE-SURFACE PHYSICS :::"
    print_text(s, self.D3Model_color)
    
    if type(model) != D3Model:
      s = ">>> FreeSurface REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    # sigma coordinate :
    self.sigma = project((model.x[2] - model.B) / (model.S - model.B),
                         annotate=False)
    print_min_max(self.sigma, 'sigma')

    Q      = self.Q
    Q_flat = self.Q_flat

    phi    = TestFunction(Q_flat)
    dS     = TrialFunction(Q_flat)
    
    Shat   = Function(Q_flat) # surface elevation velocity 
    ahat   = Function(Q_flat) # accumulation velocity
    uhat   = Function(Q_flat) # horizontal velocity
    vhat   = Function(Q_flat) # horizontal velocity perp. to uhat
    what   = Function(Q_flat) # vertical velocity
    mhat   = Function(Q_flat) # mesh velocity
    dSdt   = Function(Q_flat) # surface height change
    M      = Function(Q_flat) # mass
    ds     = self.ds_flat
    dSurf  = model.dSrf
    dBase  = model.dBed
    
    self.static_boundary = DirichletBC(Q, 0.0, self.ff_flat, 4)
    h = CellSize(self.flat_mesh)

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

    self.newz                   = Function(self.Q)
    self.mass_matrix            = mass_matrix
    self.stiffness_matrix       = stiffness_matrix
    self.diffusion_matrix       = diffusion_matrix
    self.lumped_mass            = lumped_mass
    self.A_pro                  = A_pro
    self.Shat                   = Shat
    self.ahat                   = ahat
    self.uhat                   = uhat
    self.vhat                   = vhat
    self.what                   = what
    self.mhat                   = mhat
    self.dSdt                   = dSdt
    
  def solve(self):
    """
    """
    config = self.config
   
    self.assign_variable(self.Shat, self.S) 
    self.assign_variable(self.ahat, self.adot) 
    self.assign_variable(self.uhat, self.u) 
    self.assign_variable(self.vhat, self.v) 
    self.assign_variable(self.what, self.w) 

    m = assemble(self.mass_matrix,      keep_diagonal=True)
    r = assemble(self.stiffness_matrix, keep_diagonal=True)

    s    = "::: solving free-surface :::"
    print_text(s, self.D3Model_color)
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
      self.assign_variable(self.dSdt, m_l_inv * r.get_local())
    else:
      m.ident_zeros()
      solve(m, self.dSdt.vector(), r, annotate=False)

    A = assemble(lhs(self.A_pro))
    p = assemble(rhs(self.A_pro))
    q = Vector()  
    solve(A, q, p, annotate=False)
    self.assign_variable(self.dSdt, q)


class MassHybrid(Mass):
  """
  New 2D hybrid model.
  """
  def __init__(self, model, solve_params=None, isothermal=True):
    """
    """
    s = "::: INITIALIZING HYBRID MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D2Model:
      s = ">>> MassHybrid REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)
    
    if solve_params == None:
      self.solve_params = self.default_solve_params()
    else:
      self.solve_params = solve_params
    
    # CONSTANTS
    year   = model.spy
    rho    = model.rhoi
    g      = model.g
    n      = model.n(0)
    
    Q      = model.Q
    B      = model.B
    beta   = model.beta
    adot   = model.adot
    ubar_c = model.ubar_c 
    vbar_c = model.vbar_c
    H      = model.H
    H0     = model.H0
    U      = model.UHV
    T_     = model.T_
    deltax = model.deltax
    sigmas = model.sigmas
    h      = model.h
    dt     = model.time_step
    S      = B + H
    coef   = [lambda s:1.0, lambda s:1./4.*(5*s**4 - 1.0)]
    T      = VerticalFDBasis(T_, deltax, coef, sigmas)
    
    Bc    = 3.61e-13*year
    Bw    = 1.73e3*year #model.a0 ice hardness
    Qc    = 6e4
    Qw    = model.Q0 # ice act. energy
    Rc    = model.R  # gas constant
   
    # function spaces : 
    dH  = TrialFunction(Q)
    xsi = TestFunction(Q)

    if isothermal:
      s = "    - using isothermal rate-factor -"
      print_text(s, self.color())
      def A_v(T):
        return model.b**(-n)
    else:
      s = "    - using temperature-dependent rate-factor -"
      print_text(s, self.color())
      def A_v(T):
        return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))
    
    # SIA DIFFUSION COEFFICIENT INTEGRAL TERM.
    def sia_int(s):
      return A_v(T.eval(s)) * s**(n+1)
    
    vi = VerticalIntegrator(order=4)
    
    #D = 2 * (rho*g)**n * A/(n+2) * H**(n+2) \
    #      * dot(grad(S),grad(S))**((n-1)/2)
    D = + 2 * (rho*g)**n * H**(n+2) \
            * dot(grad(S),grad(S))**((n-1)/2) \
            * vi.intz(sia_int) \
        + rho * g * H**2 / beta
    
    ubar = U[0]
    vbar = U[1]
    
    ubar_si = -D/H*S.dx(0)
    vbar_si = -D/H*S.dx(1)
    
    self.ubar_proj = (ubar-ubar_si)*xsi*dx
    self.vbar_proj = (vbar-vbar_si)*xsi*dx

    # mass term :
    self.M  = dH*xsi*dx
    
    # residual :
    self.R_thick = + (H-H0) / dt * xsi * dx \
                   + D * dot(grad(S), grad(xsi)) * dx \
                   + (Dx(ubar_c*H,0) + Dx(vbar_c*H,1)) * xsi * dx \
                   - adot * xsi * dx

    # Jacobian :
    self.J_thick = derivative(self.R_thick, H, dH)

    self.bc = []#NOTE ? DirichletBC(Q, thklim, 'on_boundary') ? maybe ?

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def default_ffc_options(self):
    """ 
    Returns a set of default ffc options that yield good performance
    """
    ffc_options = {"optimize"               : True,
                   "eliminate_zeros"        : True,
                   "precompute_basis_const" : True,
                   "precompute_ip_const"    : True}
    return ffc_options
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'nonlinear_solver' : 'snes',
               'snes_solver'      : {'method'                  : 'vinewtonrsls',
                                     'linear_solver'           : 'mumps',
                                     'relative_tolerance'      : 1e-6,
                                     'absolute_tolerance'      : 1e-6,
                                     'maximum_iterations'      : 20,
                                     'error_on_nonconvergence' : False,
                                     'report'                  : True}}
    m_params  = {'solver'      : nparams,
                 'ffc_params'  : self.default_ffc_options()}
    return m_params

  def solve(self, annotate=True):
    """
    Solves for hybrid conservation of mass.
    """
    model  = self.model
    params = self.solve_params

    # find corrective velocities :
    s    = "::: solving for corrective velocities :::"
    print_text(s, self.color())

    solve(self.M == self.ubar_proj, model.ubar_c,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=params['ffc_params'],
          annotate=annotate)

    solve(self.M == self.vbar_proj, model.vbar_c,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=params['ffc_params'],
          annotate=annotate)

    print_min_max(model.ubar_c, 'ubar_c', cls=self)
    print_min_max(model.vbar_c, 'vbar_c', cls=self)

    # SOLVE MASS CONSERVATION bounded by (H_max, H_min) :
    meth   = params['solver']['snes_solver']['method']
    maxit  = params['solver']['snes_solver']['maximum_iterations']
    s      = "::: solving 'MassTransportHybrid' using method '%s' with %i " + \
             "max iterations :::"
    print_text(s % (meth, maxit), self.color())
   
    # define variational solver for the mass problem :
    p = NonlinearVariationalProblem(self.R_thick, model.H, J=self.J_thick,
          bcs=self.bc, form_compiler_parameters=params['ffc_params'])
    p.set_bounds(model.H_min, model.H_max)
    s = NonlinearVariationalSolver(p)
    s.parameters.update(params['solver'])
    s.solve(annotate=annotate)
    
    print_min_max(model.H, 'H', cls=self)
    
    # update previous time step's H :
    model.assign_variable(model.H0, model.H, cls=self)
    
    # update the surface :
    s    = "::: updating surface :::"
    print_text(s, self.color())
    B_v = model.B.vector().array()
    H_v = model.H.vector().array()
    S_v = B_v + H_v
    model.assign_variable(model.S, S_v, cls=self)


class FirnMass(Mass):

  def __init__(self, model):
    """
    """
    s = "::: INITIALIZING FIRN MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D1Model:
      s = ">>> FirnMass REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

  def solve(self):
    """
    If conserving the mass of the firn column, calculate height of each 
    interval :
    """
    model  = self.model

    zOld   = model.z
    lnew   = append(0, model.lini) * model.rhoin / model.rhop
    zSum   = model.B
    zNew   = zeros(model.n)
    for i in range(model.n):
      zNew[i]  = zSum + lnew[i]
      zSum    += lnew[i]
    model.z    = zNew
    model.l    = lnew[1:]
    model.mp   = -(zNew - zOld) / model.dt
    model.lnew = lnew
    
    model.assign_variable(model.m_1, model.m)
    model.assign_variable(model.m,   model.mp)
    model.mesh.coordinates()[:,0][model.index] = model.z # update the mesh coor
    model.mesh.bounding_box_tree().build(model.mesh)     # rebuild the mesh tree



