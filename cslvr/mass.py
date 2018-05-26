from dolfin            import *
from dolfin_adjoint    import *
from cslvr.physics     import Physics 
from cslvr.inputoutput import print_text, print_min_max
from cslvr.d2model     import D2Model
from cslvr.d3model     import D3Model
from cslvr.helper      import VerticalBasis, VerticalFDBasis, \
                              VerticalIntegrator




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
  def __init__(self, model,
               thklim              = 1.0,
               use_shock_capturing = False,
               lump_mass_matrix    = False):
    """
    """
    print_text("::: INITIALIZING FREE-SURFACE PHYSICS :::", self.color())
    
    if type(model) != D3Model:
      s = ">>> FreeSurface REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    self.model                 = model   
    self.thklim                = thklim 
    self.use_shock_capturing   = use_shock_capturing
    self.lump_mass_matrix      = lump_mass_matrix

    # TODO: set up flat mesh stuff in D3Model!
    Q       = model.Q
    Q_flat  = model.Q
    ds      = model.ds
    dSrf    = model.dSrf
    dBed    = model.dBed
    rhob    = model.rhob       # for calculating the ice-sheet mass
    flat_mesh = model.mesh
    
    S       = Function(Q_flat, name='mass.S')
    adot    = Function(Q_flat, name='mass.adot')
    u       = Function(Q_flat)
    v       = Function(Q_flat)
    w       = Function(Q_flat)
    mhat    = Function(Q_flat)
    h       = CellDiameter(flat_mesh)
    
    self.assx  = FunctionAssigner(u.function_space(), model.u.function_space())
    self.assy  = FunctionAssigner(v.function_space(), model.v.function_space())
    self.assz  = FunctionAssigner(w.function_space(), model.w.function_space())

    # get the dof map :
    if model.use_periodic:
      self.v2d      = vertex_to_dof_map(model.Q_non_periodic)
      self.mhat_non = Function(model.Q_non_periodic)
      self.assmhat  = FunctionAssigner(model.mhat.function_space(),
                                       model.Q_non_periodic)
    else:
      self.v2d      = vertex_to_dof_map(Q)

    # self.m when assembled gives the mass of the domain :
    self.M_tot  = rhob * dx

    # set up linear variational problem for unknown dSdt :
    phi         = TestFunction(Q_flat)
    psi         = TrialFunction(Q_flat)
    dSdt        = Function(Q_flat, name='mass.dSdt')

    # SUPG-modified trial function (artifical diffusion in direction of flow) :
    U           = as_vector([u, v, w])
    unorm       = sqrt(dot(U,U) + 1e-1)
    phihat      = phi + h / (2.0*unorm) * dot(U, grad(phi))

    # mass matrices :
    mass        = psi * phi * dSrf
    lumped_mass = phi * dSrf

    # stiffness matrix :
    stiffness   = + adot * phi * dSrf \
                  + w * phi * dSrf \
                  - dot(U, grad(S)) * phihat * dSrf
   
    # shock-capturing steady-state residual dependent artificial diffusion :
    S_mag       = dot(grad(S), grad(S)) + 1e-1
    resid       = dot(U, grad(S)) - w - adot
    C           = 1e1 * h / (2*unorm) * S_mag * resid**2
    diffusion   = C * dot(grad(phi), grad(S)) * dSrf
    
    # TODO: just solve over a 2D mesh, then this step is not required.
    # extrude the surface value of dSdt throughout the domain; i.e., 
    # d/dz (dSdt) = 0, dSdt |_{Gamma_S} = f where M.f = K : 
    self.A_pro  = dSdt*phi*dSrf - psi*phi*dBed - psi*phi.dx(2)*dx

    self.stiffness    = stiffness
    self.diffusion    = diffusion
    self.mass         = mass
    self.lumped_mass  = lumped_mass
    self.S            = S
    self.adot         = adot
    self.u            = u
    self.v            = v
    self.w            = w
    self.mhat         = mhat
    self.dSdt         = dSdt

  def update_mesh_and_surface(self, S):
    """
    This method will update the surface height ``self.model.S`` and vertices 
    of deformed mesh ``self.model.mesh`` while imposing thickness limit set 
    by ``self.thklim``.
    """
    print_text("    - updating mesh and surface -", self.color())
    model  = self.model
    mesh   = model.mesh
    sigma  = model.sigma   # sigma coordinate
    thklim = self.thklim   # thickness limit
    v2d    = self.v2d
    
    # impose the thickness limit :
    B                       = model.B.vector().get_local()
    thin                    = (S - B) < thklim
    S[thin]                 = B[thin] + thklim

    # update the mesh and surface :
    sigma                   = sigma.compute_vertex_values()
    mesh.coordinates()[:,2] = sigma*(S[v2d] - B[v2d]) + B[v2d] # update mesh
    model.assign_variable(model.S, S)                          # update surface

  def solve(self):
    """
    This method solves the free-surface equation, updating ``self.model.dSdt``.
    """
    print_text("::: solving free-surface relation :::", self.color())
    
    model  = self.model
  
    # assign the current values from the deformed mesh to the sigma-coord mesh :
    model.assign_variable(self.S,    model.S) 
    model.assign_variable(self.adot, model.adot) 
    self.assx.assign(self.u,        model.u)
    self.assy.assign(self.v,        model.v)
    self.assz.assign(self.w,        model.w)
    #model.assign_variable(self.u,    model.u) 
    #model.assign_variable(self.v,    model.v) 
    #model.assign_variable(self.w,    model.w) 

    # assemple the stiffness matrix :
    K = assemble(self.stiffness, keep_diagonal=True)

    # add artificial diffusion to stiff. matrix in regions of high S gradients :
    if self.use_shock_capturing:
      D = assemble(self.diffusion)
      K = K - D
      print_min_max(D, 'D')
      print_min_max(K, 'K')

    # calculate preliminary guess :
    if self.lump_mass_matrix:
      M               = assemble(self.lumped_mass)
      M_a             = M.get_local()
      M_a[M_a == 0.0] = 1.0
      self.assign_variable(self.dSdt, K.get_local() / M_a)
    else:
      M = assemble(self.mass, keep_diagonal=True)
      M.ident_zeros()
      solve(M, self.dSdt.vector(), K, annotate=False)
   
    # extrude the surface value of dSdt throughout the domain :
    A = assemble(lhs(self.A_pro))
    p = assemble(rhs(self.A_pro))
    q = Vector()
    solve(A, model.dSdt.vector(), p, annotate=False)
    print_min_max(model.dSdt, 'dSdt')




class MassHybrid(Mass):
  """
  New 2D hybrid model.

  Original author: Doug Brinkerhoff: https://dbrinkerhoff.org/
  """
  def __init__(self, model, momentum, 
               thklim       = 1.0,
               solve_params = None,
               isothermal   = True):
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
    U      = momentum.U
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

    # when assembled, this gives the mass of the domain :
    self.M_tot  = rho * H * dx

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
    self.m  = dH*xsi*dx
    
    # residual :
    self.R_thick = + (H-H0) / dt * xsi * dx \
                   + D * dot(grad(S), grad(xsi)) * dx \
                   + (Dx(ubar_c*H,0) + Dx(vbar_c*H,1)) * xsi * dx \
                   - adot * xsi * dx

    # Jacobian :
    self.J_thick = derivative(self.R_thick, H, dH)

    self.bc = []#NOTE ? DirichletBC(Q, thklim, 'on_boundary') ? maybe ?
    self.bc = [DirichletBC(Q, thklim, 'on_boundary')]
   
    # create solver for the problem : 
    problem = NonlinearVariationalProblem(self.R_thick, model.H, 
                J=self.J_thick, bcs=self.bc)
    problem.set_bounds(model.H_min, model.H_max)
    self.solver = NonlinearVariationalSolver(problem)
    self.solver.parameters.update(self.solve_params['solver'])

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def default_ffc_options(self):
    """ 
    Returns a set of default ffc options that yield good performance
    """
    #ffc_options = {"optimize"               : True,
    #               "eliminate_zeros"        : True,
    #               "precompute_basis_const" : True,
    #               "precompute_ip_const"    : True}
    ffc_options = None
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

    solve(self.m == self.ubar_proj, model.ubar_c,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=params['ffc_params'],
          annotate=annotate)

    solve(self.m == self.vbar_proj, model.vbar_c,
          solver_parameters={'linear_solver':'mumps'},
          form_compiler_parameters=params['ffc_params'],
          annotate=annotate)

    print_min_max(model.ubar_c, 'ubar_c')
    print_min_max(model.vbar_c, 'vbar_c')

    # SOLVE MASS CONSERVATION bounded by (H_max, H_min) :
    meth   = params['solver']['snes_solver']['method']
    maxit  = params['solver']['snes_solver']['maximum_iterations']
    s      = "::: solving 'MassTransportHybrid' using method '%s' with %i " + \
             "max iterations :::"
    print_text(s % (meth, maxit), self.color())
   
    # define variational solver for the mass problem :
    #p = NonlinearVariationalProblem(self.R_thick, model.H, J=self.J_thick,
    #      bcs=self.bc, form_compiler_parameters=params['ffc_params'])
    #p.set_bounds(model.H_min, model.H_max)
    #s = NonlinearVariationalSolver(p)
    #s.parameters.update(params['solver'])
    out = self.solver.solve(annotate=annotate)
    
    print_min_max(model.H, 'H')
    
    # update previous time step's H :
    model.assign_variable(model.H0, model.H)
    
    # update the surface :
    s    = "::: updating surface :::"
    print_text(s, self.color())
    B_v = model.B.vector().get_local()
    H_v = model.H.vector().get_local()
    S_v = B_v + H_v
    model.assign_variable(model.S, S_v)

    return out




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



