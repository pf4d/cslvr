from dolfin            import *
from dolfin_adjoint    import *
from cslvr.physics     import Physics 
from cslvr.inputoutput import print_text, print_min_max
from cslvr.d2model     import D2Model
from cslvr.d3model     import D3Model
from cslvr.helper      import VerticalBasis, VerticalFDBasis, \
                              VerticalIntegrator
import json




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
  r"""

  This class defines the physics and solution to the upper free-surface equation.
    .. math::
       \frac{\partial S}{\partial t} - u_z + \underline{u} \cdot \nabla S = \mathring{S}

  When the :func:`~mass.FreeSurface.solve` method is executed, something similar to the following output will be displayed:

  .. code-block:: none

     ::: solving free-surface relation :::
     || K_source ||_2    : 2.205e+12
     || K_advection ||_2 : 7.736e+08
     || K_stab_u ||_2    : 1.322e+10
     || K_stab_gs ||_2   : 1.699e+10
     dSdt <min, max> : <-8.790e+09, 1.436e+08>
     ::: extruding function downwards :::
     Process 0: Solving linear variational problem.
     extruded function <min, max> : <-1.127e+01, 5.443e-01>
     dSdt <min, max> : <-1.127e+01, 5.443e-01>

  Here, 
  
  * ``S`` is the surface height :math:`S` saved to ``model.S``
  * ``dSdt`` is the time rate of change :math:`\partial_t S` of the surface :math:`S`
  * ``K_source`` is the tensor corresponding to the upper-surface accumulation/ablation function :math:`\mathring{S}` located (currently) at ``model.adot``
  * ``K_advection`` is the tensor corresponding to the advective part of the free-surface equation :math:`- u_z + \underline{u} \cdot \nabla S`
  * ``K_stab_u`` is the tensor corresponding to the streamline/Petrov-Galerkin in stabilization term the direction of velocity located (currently) at ``model.U3``
  * ``K_stab_gs`` is the tensor corresponding to the streamline/Petrov-Galerkin stabilization term in the direction of surface gradient :math:`\nabla S`

  """ 
  def __init__(self, model,
               thklim              = 1.0,
               lump_mass_matrix    = False):
    """

    """
    print_text("::: INITIALIZING FREE-SURFACE PHYSICS :::", self.color())
    
    if type(model) != D2Model:
      s = ">>> FreeSurface REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    self.solve_params    = self.default_solve_params()
    s = "::: using default parameters :::"
    print_text(s, self.color())
    s = json.dumps(self.solve_params, sort_keys=True, indent=2)
    print_text(s, '230')

    self.model            = model
    self.thklim           = thklim
    self.lump_mass_matrix = lump_mass_matrix

    # get model var's so we don't have a bunch of ``model.`` crap in our math :
    Q       = model.Q
    rhob    = model.rhob
    S_0     = model.S
    adot    = model.adot
    u       = model.u
    v       = model.v
    w       = model.w
    mhat    = model.mhat
    h       = model.h
    dt      = model.time_step

    # specical dof mapping for periodic spaces :
    if model.use_periodic:
      self.mhat_non = Function(model.Q_non_periodic)
      self.assmhat  = FunctionAssigner(mhat.function_space(),
                                       model.Q_non_periodic)

    # when assembled, this gives the total mass of the domain :
    self.M_tot  = rhob * dx

    # set up linear variational problem for unknown dSdt :
    phi         = TestFunction(Q)
    dS          = TrialFunction(Q)
    S           = Function(Q, name='mass.S')

    # velocity vector :
    U           = as_vector([u, v, 0])

    # z-coordinate unit vector :
    k           = as_vector([0, 0, 1])

    # SUPG-modified trial function (artifical diffusion in direction of flow) :
    unorm       = sqrt(dot(U,U)) + DOLFIN_EPS
    tau         = h / (2 * unorm)
    phihat      = phi + tau * dot(U, grad(phi))

    # SUPG-modified shock-capturing trial function :
    gSnorm      = dot(grad(S), grad(S)) + DOLFIN_EPS
    tau_gs      = h / (2 * gSnorm)
    phihat_gs   = phi + tau_gs * dot(grad(S), grad(phi))

    # source coefficient :
    gS          = sqrt(1 + dot(grad(S), grad(S)))

    # mass matrix :
    self.mass   = S * phi * dx

    # right-hand side, i.e., \partial_t S = \delta :
    delta       = gS*adot - (dot(U, grad(S)) - w)

    # theta-scheme (nu = 1/2 == Crank-Nicolson) :
    nu          = 0.5
    S_mid       = nu*S + (1 - nu)*S_0
    
    # the linear differential operator for this problem (pure advection) :
    def Lu(u): return dot(U, grad(u))

    # partial time derivative :
    dSdt = (S - S_0) / dt

    # LHS of dSdt + Lu(S_mid) = f :
    f  = adot + w

    # bilinear form :
    self.delta_S = + (dSdt + Lu(S_mid) - f) * phi * dx \
                   + inner( Lu(phi), tau*(dSdt + Lu(S_mid) - f) ) * dx
    # Jacobian :
    self.mass_Jac = derivative(self.delta_S, S, dS)
    self.S        = S

    # stiffness matrix :
    self.source      = f       * phi           * dx
    self.advection   = Lu(S_0) * phi           * dx
    self.stab_u      = Lu(S_0) * tau * Lu(phi) * dx

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
    
    # impose the thickness limit and update the surface :
    B                       = model.B.vector().get_local()
    thin                    = (S - B) < thklim
    S[thin]                 = B[thin] + thklim
    model.assign_variable(model.S, S)

    # update the mesh :
    sigma                   = sigma.compute_vertex_values()
    S                       = model.S.compute_vertex_values()
    B                       = model.B.compute_vertex_values()
    mesh.coordinates()[:,2] = sigma*(S - B) + B
  
  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.solve_params

  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
                                  'preconditioner'           : 'none',
                                  'relative_tolerance'       : 1e-12,
                                  'relaxation_parameter'     : 1.0,
                                  'maximum_iterations'       : 20,
                                  'error_on_nonconvergence'  : False}}
    params  = {'solver'  : {'linear_solver'       : 'mumps',
                            'preconditioner'      : 'none'},
               'nparams' : nparams}
    return params
  
  def solve(self, annotate=False):
    """
    This method solves the free-surface equation for the upper-surface height :math:`S`, updating ``self.model.S``.

    Currently does not support dolfin-adjoint annotation.
    """
    print_text("::: solving free-surface relation :::", self.color())
    
    model  = self.model
  
    ## solve the linear system :
    #solve(lhs(self.delta_S) == rhs(self.delta_S), self.S, annotate=annotate)

    # solve the non-linear system :
    model.assign_variable(self.S, DOLFIN_EPS, annotate=annotate)
    solve(self.delta_S == 0, self.S, J=self.mass_Jac,
          annotate=annotate, solver_parameters=self.solve_params['nparams'])
    print_min_max(self.S, self.S.name())

    # the solution is only valid at the upper surface :    
    #S = model.vert_extrude(self.S, d='down')
    
    # update the model variable :
    model.assign_variable(model.S, self.S, annotate=annotate)

    # assemple the stiffness and mass matrices :
    K_source    = assemble(self.source)
    K_advection = assemble(self.advection)
    K_stab_u    = assemble(self.stab_u)

    # print tensor statistics :
    print_min_max( norm(K_source,    'l2'),  '|| K_source ||_2   ' )
    print_min_max( norm(K_advection, 'l2'),  '|| K_advection ||_2' )
    print_min_max( norm(K_stab_u,    'l2'),  '|| K_stab_u ||_2   ' )

  def old_solve(self, annotate=False):
    """
    This method solves the free-surface equation, updating ``self.model.dSdt``.

    Currently does not support dolfin-adjoint annotation.
    """
    print_text("::: solving free-surface relation :::", self.color())
    
    model  = self.model
  
    # assemple the stiffness and mass matrices :
    K_source    = assemble(self.source)
    K_advection = assemble(self.advection)
    K_stab_u    = assemble(self.stab_u)
    K_stab_gs   = assemble(self.stab_gs)

    # print tensor statistics :
    print_min_max( norm(K_source,    'l2'),  '|| K_source ||_2   ' )
    print_min_max( norm(K_advection, 'l2'),  '|| K_advection ||_2' )
    print_min_max( norm(K_stab_u,    'l2'),  '|| K_stab_u ||_2   ' )
    print_min_max( norm(K_stab_gs,   'l2'),  '|| K_stab_gs ||_2  ' )

    # form stiffness matrix :
    K = K_source - K_advection - K_stab_u# - K_stab_gs
    
    # form mass matrix :
    # NOTE: ident_zeros() ensures that interior nodes are 
    #       set to the identity, otherwise it is not invertable; due to the 
    #       fact that the interior nodes are not needed for the calculation 
    #       of dSdt which is only defined at the upper surface boundary 
    #       ``dSrf``.  This fact gives rise to the following:
    # TODO: Solve the free-surface equation over a 2D mesh which is extruded        #       back to the 3D mesh
    if self.lump_mass_matrix:
      M = assemble(action(self.mass, Constant(1)))  # vector of row sums
      M[M == 0] = 1.0                               # analogous to ident_zeros()
      model.assign_variable(model.dSdt, K.get_local() / M.get_local())
    else:
      M = assemble(self.mass, keep_diagonal=True)
      M.ident_zeros()
      solve(M, model.dSdt.vector(), K, annotate=annotate)

    # extrude the value throughout the interior and assign it to the model :
    dSdt = model.vert_extrude(model.dSdt, d='down')
    model.assign_variable(model.dSdt, dSdt)




class MassHybrid(Mass):
  """
  New 2D hybrid model.

  Original author: `Doug Brinkerhoff <https://dbrinkerhoff.org/>`_
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

    # thermal parameters :
    Bc     = model.a_T_l    # lower bound of flow-rate const.
    Bw     = model.a_T_u    # upper bound of flow-rate const.
    Qc     = model.Q_T_l    # lower bound of ice act. energy
    Qw     = model.Q_T_u    # upper bound of ice act. energy
    Rc     = model.R        # gas constant
   
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



