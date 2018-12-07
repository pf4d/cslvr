from cslvr.physics import Physics

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

	def __init__(self, model,
	             solve_params    = None,
	             transient       = False,
	             use_smb_for_ela = False,
	             ela             = None):
		"""
		Set up the equations
		"""
		s    = "::: INITIALIZING AGE PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != D3Model:
			s = ">>> Age REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		if solve_params == None:
			self.solve_params = self.default_solve_params()
		else:
			self.solve_params = solve_params

		# only need the cell size and velocity :
		h = model.h
		U = model.u

		# Trial and test
		a        = TrialFunction(model.Q)
		phi      = TestFunction(model.Q)

		# Steady state
		if not transient:
			s    = "    - using steady-state -"
			print_text(s, cls=self)

			# SUPG intrinsic time parameter :
			Unorm = sqrt(dot(U,U) + DOLFIN_EPS)
			tau   = h / (2 * Unorm)

			# the advective part of the operator :
			def L(u): return dot(U, grad(u))

			# streamlin-upwind/Petrov-Galerkin form :
			self.a = + dot(U,grad(a)) * phi * dx \
			         + innner(L(phi), tau*L(a)) * dx
			self.L = + Constant(1.0) * phi * dx \
			         + tau * L(phi) * dx

		# FIXME: 3D model does not currently support mesh-movement.
		else:
			s    = "    - using transient -"
			print_text(s, cls=self)

			# Time step
			dt     = model.time_step

			# SUPG intrinsic-time (note subtraction of mesh velocity) :
			U      = as_vector([model.u_x, model.v, model.w - model.mhat])
			Unorm  = sqrt(dot(U,U) + DOLFIN_EPS)
			tau    = h / (2 * Unorm)

			# midpoint value of age for Crank-Nicholson :
			# FIXME: what is ahat?
			a_mid = 0.5*(a + self.ahat)

			# SUPG intrinsic time parameter :
			Unorm = sqrt(dot(U,U) + DOLFIN_EPS)
			tau   = h / (2 * Unorm)

			# the differental operator is entirely advective :
			def L(u): return dot(U, grad(u))

			# streamlin-upwind/Petrov-Galerkin form :
			# FIXME: no a0 anymore
			self.a = + (a - a0)/dt * phi * dx \
			         + dot(U,grad(a_mid)) * phi * dx \
			         + innner(L(phi), tau*L(a_mid)) * dx
			self.L = + Constant(1.0) * phi * dx \
			         + tau * L(phi) * dx

		# form the boundary conditions :
		if use_smb_for_ela:
			s    = "    - using S_ring (SMB) boundary condition -"
			print_text(s, cls=self)
			self.bc_age = DirichletBC(model.Q, 0.0, model.ff_acc, 1)

		else:
			s    = "    - using ELA boundary condition -"
			print_text(s, cls=self)
			def above_ela(x,on_boundary):
				return x[2] > ela and on_boundary
			self.bc_age = DirichletBC(model.Q, 0.0, above_ela)

	def solve(self, annotate=False):
		"""
		Solve the system
		"""
		print_text("::: solving age :::", cls=self)

		# Solve!
		#solve(lhs(self.F) == rhs(self.F), model.age, self.bc_age)
		solve(self.a == self.L, self.age, self.bc_age, annotate=annotate)
		self.model.age.interpolate(self.age)
		print_min_max(self.model.age, 'age')


class FirnAge(Physics):

	def __init__(self, model, solve_params=None):
		"""
		"""
		s    = "::: INITIALIZING FIRN AGE PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != D1Model:
			s = ">>> FirnAge REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		if solve_params == None:
			self.solve_params = self.default_solve_params()
		else:
			self.solve_params = solve_params

		Q       = model.Q
		w       = model.w                         # velocity
		w_1     = model.w0                        # previous step's velocity
		m       = model.m                         # mesh velocity
		m_1     = model.m0                        # previous mesh velocity
		a       = model.age                       # age
		a_1     = model.age0                      # previous step's age
		dt      = model.time_step                 # timestep

		da      = TrialFunction(Q)
		xi      = TestFunction(Q)

		model.assign_variable(a,   1.0)
		model.assign_variable(a_1, 1.0)

		# age residual :
		# theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger,
		#               0.5=Crank-Nicolson, 0=Forward-Euler) :
		# uses Taylor-Galerkin upwinding :
		theta   = 0.5
		a_mid   = theta*a + (1-theta)*a_1
		f       = + (a - a_1)/dt * xi * dx \
		          - 1 * xi * dx \
		          + w * a_mid.dx(0) * xi * dx \
		          - 0.5 * (w - w_1) * a_mid.dx(0) * xi * dx \
		          + w**2 * dt/2 * inner(a_mid.dx(0), xi.dx(0)) * dx \
		          - w * w.dx(0) * dt/2 * a_mid.dx(0) * xi * dx

		J       = derivative(f, a, da)

		self.ageBc = DirichletBC(Q, 0.0, model.surface)

		self.f = f
		self.J = J

	def get_solve_params(self):
		"""
		Returns the solve parameters.
		"""
		return self.solve_params

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		params = {'solver' : {'relaxation_parameter'     : 1.0,
		                       'maximum_iterations'      : 25,
		                       'error_on_nonconvergence' : False,
		                       'relative_tolerance'      : 1e-10,
		                       'absolute_tolerance'      : 1e-10}}
		return params

	def solve(self):
		"""
		"""
		s    = "::: solving FirnAge :::"
		print_text(s, cls=self)

		model  = self.model

		# solve for age :
		solve(self.f == 0, self.a, self.ageBc, J=self.J,
		      solver_parameters=self.solve_params['solver'])
		model.age.interpolate(self.a)
		print_min_max(model.a, 'age')



