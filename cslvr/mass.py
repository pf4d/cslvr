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

	def get_solve_params(self):
		"""
		Returns the solve parameters.
		"""
		return self.solve_params

	def solve(self, annotate=True, params=None):
		"""
		Solve the conservation of mass equation for a free-surface evolution.
		"""
		raiseNotDefined()




class UpperFreeSurface(Mass):
	r"""

	This class defines the physics and solution to the upper free-surface equation.
	  .. math::
	     \frac{\partial S}{\partial t} - u_z + \underline{u} \cdot \nabla S = \Vert \underline{\hat{k}} - \nabla S \Vert \mathring{S}.

	When the :func:`~mass.FreeSurface.solve` method is executed, something similar to the following output will be displayed:

	.. code-block:: none

		::: solving free-surface relation :::
		mass.S <min, max> : <3.000e-16, 3.000e-16>
		Solving nonlinear variational problem.
		  Newton iteration 0: r (abs) = 2.750e+12 (tol = 1.000e-10) r (rel) = 1.000e+00 (tol = 1.000e-12)
		  Newton iteration 1: r (abs) = 9.595e-04 (tol = 1.000e-10) r (rel) = 3.489e-16 (tol = 1.000e-12)
		  Newton solver finished in 1 iterations and 1 linear solver iterations.
		mass.S <min, max> : <-1.140e+02, 6.065e+02>
		S <min, max> : <1.000e+00, 6.065e+02>

	Here,

	* ``S`` is the surface height :math:`S` saved to ``model.S``

	"""
	def __init__(self, model,
	             thklim              = 1.0,
	             lump_mass_matrix    = False):
		"""

		"""
		print_text("::: INITIALIZING UPPER-FREE-SURFACE PHYSICS :::", cls=self)

		if type(model) != D2Model:
			s = ">>> UpperFreeSurface REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		self.solve_params    = self.default_solve_params()
		s = "::: using default parameters :::"
		print_text(s, cls=self)
		s = json.dumps(self.solve_params, sort_keys=True, indent=2)
		print_text(s, '230')

		self.model            = model
		self.thklim           = thklim
		self.lump_mass_matrix = lump_mass_matrix

		# interpolate if the function spaces differ :
		Q       = model.Q
		S_0     = model.S
		S_ring  = model.S_ring
		u       = model.u_x
		v       = model.v
		w       = model.w
		h       = model.h
		dt      = model.time_step

		# set up non-linear variational problem for unknown S :
		phi         = TestFunction(Q)
		dS          = TrialFunction(Q)
		S_f         = Function(Q, name='mass.S')
		dSdt_f      = Function(Q, name='mass.dSdt')

		# velocity vector :
		U           = as_vector([u, v, 0])

		# z-coordinate unit vector :
		k           = as_vector([0, 0, 1])

		# SUPG-modified trial function (artifical diffusion in direction of flow) :
		unorm       = sqrt(dot(U,U) + DOLFIN_EPS)
		tau         = h / (2 * unorm)
		phihat      = phi + tau * dot(U, grad(phi))

		# SUPG-modified shock-capturing trial function :
		gSnorm      = sqrt(dot(grad(S_0), grad(S_0)) + DOLFIN_EPS)
		tau_gs    	= h / (2 * gSnorm)
		phihat_gs   = phi + tau_gs * dot(grad(S_0), grad(phi))

		# the linear differential operator for this problem (pure advection) :
		def Lu(u): return dot(U, grad(u))

		# dSdt mass matrix :
		if self.lump_mass_matrix:
			M         = assemble(action(dS * phi * dx, Constant(1)))  # row sums
			M[M == 0] = 1.0
		else:
			M         = assemble(dS * phi * dx)

		# theta-scheme (nu = 1/2 == Crank-Nicolson) :
		nu          = 0.5
		S_mid       = nu*S_f + (1 - nu)*S_0

		# source coefficient :
		gS_f        = sqrt(1 + dot(grad(S_mid), grad(S_mid)))
		gS          = sqrt(1 + dot(grad(S_0),   grad(S_0)))

		# partial time derivative :
		dSdt = (S_f - S_0) / dt

		# LHS of dSdt + Lu(S_mid) = f :
		f    = w + S_ring#gS*S_ring

		# variational residual :
		self.delta_S  = + (dSdt + Lu(S_mid) - f) * phi * dx \
		                + inner( Lu(phi), tau*(dSdt + Lu(S_mid) - f) ) * dx

		# dSdt variational form :
		self.delta_dS = + (f - Lu(S_0)) * phi * dx \
		                - inner( Lu(phi), tau*Lu(S_0) ) * dx

		# Jacobian :
		self.mass_Jac    = derivative(self.delta_S, S_f, dS)
		self.M           = M
		self.dSdt_f      = dSdt_f
		self.S_f         = S_f
		self.Lu          = Lu
		self.f           = f

	def get_unknown(self):
		"""
		Return the unknown Function.
		"""
		return self.S_f

	def solve(self, annotate=False):
		"""
		This method solves the free-surface equation for the upper-surface height :math:`S`, updating ``self.model.S``.

		Currently does not support dolfin-adjoint annotation.
		"""
		print_text("::: solving upper-free-surface relation :::", cls=self)

		model  = self.model
		thklim = self.thklim   # thickness limit

		## solve the linear system :
		#solve(lhs(self.delta_S) == rhs(self.delta_S), self.S, annotate=annotate)

		# solve the non-linear system :
		model.assign_variable(self.S_f, DOLFIN_EPS, annotate=annotate, cls=self)
		solve(self.delta_S == 0, self.get_unknown(), J=self.mass_Jac,
		      annotate=annotate, solver_parameters=self.solve_params['nparams'])

		# update the model surface :
		self.update_model_var(self.get_unknown(), annotate=annotate)

		# calculate the surface height time derivative :
		self.solve_dSdt(annotate=annotate)

	def solve_dSdt(self, annotate=False):
		r"""
		This method solves the free-surface equation, updating ``self.model.dSdt``.

		Currently does not support dolfin-adjoint annotation.
		"""
		print_text("::: solving free-surface relation for dSdt :::", cls=self)

		model  = self.model

		# assemple the stiffness matrix :
		K = assemble(self.delta_dS)

		# add artificial diffusion to stiff. matrix in regions of high S gradients :
		#if self.use_shock_capturing:
		#  D = assemble(self.diffusion)
		#  K = K - D
		#  print_min_max(D, 'D')
		#  print_min_max(K, 'K')

		# calculate preliminary guess :
		if self.lump_mass_matrix:
			M_a             = self.M.get_local()
			model.assign_variable(self.dSdt_f, K.get_local() / M_a, cls=self)
		else:
			solve(self.M, self.dSdt_f.vector(), K, annotate=annotate)

		model.assign_variable(model.dSdt, self.dSdt_f, annotate=annotate, cls=self)

	def update_model_var(self, u, annotate=False):
		"""
		Update the two horizontal components of velocity in ``self.model.u``
		to those given by ``u``.
		"""
		# impose the thickness limit and update the model's surface :
		S           = u.vector().get_local()
		B           = self.model.B.vector().get_local()
		thin        = (S - B) < self.thklim
		S[thin]     = B[thin] + self.thklim
		self.model.assign_variable(self.model.S, S, annotate=annotate, cls=self)




class LowerFreeSurface(Mass):
	r"""

	This class defines the physics and solution to the lower free-surface equation.
		.. math::
	     \frac{\partial B}{\partial t} - u_z + \underline{u} \cdot \nabla B = - \Vert \nabla B - \underline{\hat{k}} \Vert \mathring{B}.

	When the :func:`~mass.FreeSurface.solve` method is executed, something similar to the following output will be displayed:

	Here,

	* ``B`` is the surface height :math:`B` saved to ``model.B``
	* ``K_source`` is the tensor corresponding to the source term :math:`f = u_z - \Vert \nabla B - \underline{\hat{k}} \Vert \mathring{B}` with lower-surface accumulation/ablation function located (currently) at ``model.B_ring`` and vertical velocity :math:`u_z`.
	* ``K_advection`` is the tensor corresponding to the advective part of the free-surface equation :math:`\underline{u} \cdot \nabla B`
	* ``K_stab_u`` is the tensor corresponding to the streamline/Petrov-Galerkin in stabilization term the direction of velocity located (currently) at ``model.u``

	"""
	def __init__(self, model,
	             thklim              = 1.0,
	             lump_mass_matrix    = False):
		"""

		"""
		print_text("::: INITIALIZING LOWER-FREE-SURFACE PHYSICS :::", cls=self)

		if type(model) != D2Model:
			s = ">>> LowerFreeSurface REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		self.solve_params    = self.default_solve_params()
		s = "::: using default parameters :::"
		print_text(s, cls=self)
		s = json.dumps(self.solve_params, sort_keys=True, indent=2)
		print_text(s, '230')

		self.model            = model
		self.thklim           = thklim
		self.lump_mass_matrix = lump_mass_matrix

		# get model var's so we don't have a bunch of ``model.`` crap in our math :
		Q       = model.Q
		rhob    = model.rhob
		B_0     = model.B
		B_ring  = model.B_ring
		u       = model.u_x
		v       = model.v
		w       = model.w
		h       = model.h
		dt      = model.time_step

		# set up non-linear variational problem for unknown B :
		phi         = TestFunction(Q)
		dS          = TrialFunction(Q)
		B           = Function(Q, name='mass.B')

		# velocity vector :
		U           = as_vector([u, v, 0])

		# z-coordinate unit vector :
		k           = as_vector([0, 0, 1])

		# SUPG-modified trial function (artifical diffusion in direction of flow) :
		unorm       = sqrt(dot(U,U)) + DOLFIN_EPS
		tau         = h / (2 * unorm)
		phihat      = phi + tau * dot(U, grad(phi))

		# source coefficient :
		gB          = sqrt(1 + dot(grad(B), grad(B)))

		# theta-scheme (nu = 1/2 == Crank-Nicolson) :
		nu          = 0.5
		B_mid       = nu*B + (1 - nu)*B_0

		# the linear differential operator for this problem (pure advection) :
		def Lu(u): return dot(U, grad(u))

		# partial time derivative :
		dBdt = (B - B_0) / dt

		# LHS of dSdt + Lu(S_mid) = f :
		f  = w - gB*B_ring

		# bilinear form :
		self.delta_B = + (dBdt + Lu(B_mid) - f) * phi * dx \
		               + inner( Lu(phi), tau*(dBdt + Lu(B_mid) - f) ) * dx
		# Jacobian :
		self.mass_Jac = derivative(self.delta_B, B, dB)
		self.B        = B

		# stiffness matrix :
		self.source      = f       * phi           * dx
		self.advection   = Lu(B_0) * phi           * dx
		self.stab_u      = Lu(B_0) * tau * Lu(phi) * dx

	def solve(self, annotate=False):
		"""
		This method solves the lower free-surface equation for the lower-surface height :math:`B`, updating ``self.model.B``.

		Currently does not support dolfin-adjoint annotation.
		"""
		print_text("::: solving lower-free-surface relation :::", cls=self)

		model  = self.model
		thklim = self.thklim   # thickness limit

		## solve the linear system :
		#solve(lhs(self.delta_S) == rhs(self.delta_S), self.S, annotate=annotate)

		# solve the non-linear system :
		model.assign_variable(self.B, DOLFIN_EPS, annotate=annotate)
		solve(self.delta_B == 0, self.B, J=self.mass_Jac,
		      annotate=annotate, solver_parameters=self.solve_params['nparams'])
		model.assign_variable(model.B, self.B, annotate=annotate)

		# assemple the stiffness and mass matrices :
		K_source    = assemble(self.source)
		K_advection = assemble(self.advection)
		K_stab_u    = assemble(self.stab_u)

		# print tensor statistics :
		print_min_max( norm(K_source,    'l2'),  '|| K_source ||_2   ' )
		print_min_max( norm(K_advection, 'l2'),  '|| K_advection ||_2' )
		print_min_max( norm(K_stab_u,    'l2'),  '|| K_stab_u ||_2   ' )




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
		print_text(s, cls=self)

		if type(model) != D2Model:
			s = ">>> MassHybrid REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		if solve_params == None:
			self.solve_params = self.default_solve_params()
		else:
			self.solve_params = solve_params

		# CONSTANTS
		rho    = model.rho_i
		g      = model.g
		n      = model.n(0)

		Q      = model.Q
		B      = model.B
		beta   = model.beta
		S_ring = model.S_ring
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
			print_text(s, cls=self)
			def A_v(T):
				return model.b**(-n)
		else:
			s = "    - using temperature-dependent rate-factor -"
			print_text(s, cls=self)
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
		               - S_ring * xsi * dx

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
		print_text(s, cls=self)

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
		print_text(s % (meth, maxit), cls=self)

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
		print_text(s, cls=self)
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
		print_text(s, cls=self)

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
		lnew   = append(0, model.lini) * model.rho_in / model.rhop
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



