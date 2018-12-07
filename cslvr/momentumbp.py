from dolfin               import *
from dolfin_adjoint       import *
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d3model        import D3Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
from cslvr.helper         import raiseNotDefined
import sys




class MomentumBPBase(Momentum):
	"""
	"""

	def __init__(self, model, solve_params=None,
	             linear=False, use_lat_bcs=False,
	             use_pressure_bc=True):
		"""
		Initializes the class's variables to default values that are then set
		by the individually created model.

		Initilize the residuals and Jacobian for the momentum equations.
		"""
		s = "::: INITIALIZING BP VELOCITY BASE PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != D3Model:
			s = ">>> MomentumBPBase REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		#===========================================================================
		# define function spaces :

		# finite element used :
		self.Qe    = model.QM2e

		# function space is available for later use :
		self.Q     = model.QM2

		# function assigner goes from the U function solve to u vector
		# function used to save :
		self.assz  = FunctionAssigner(model.V.sub(2),  model.Q)
		self.assu  = FunctionAssigner([model.V.sub(0), model.V.sub(1)], self.Q)
		self.assp  = FunctionAssigner(model.Q, model.Q)

		# momenturm functions :
		self.set_unknown(Function(self.Q, name='u_h'))
		self.set_trial_function(TrialFunction(self.Q))
		self.set_test_function(TestFunction(self.Q))

		# horizontal velocity :
		u_x, u_y  = self.get_unknown()
		v_x, v_y  = self.get_trial_function()

		# vertical velocity :
		u_z      = TrialFunction(model.Q)
		v_z      = TestFunction(model.Q)
		self.u_z = Function(model.Q, name='u_z')

		#model.calc_normal_vector()
		#n_b       = model.n_b
		dGamma_b   = model.dGamma_b()
		B          = model.B
		B_ring     = model.B_ring
		n_b_mag    = sqrt(1 + dot(grad(B), grad(B)))
		k_hat      = as_vector([0,0,1])
		n_b        = (grad(B) - k_hat) / n_b_mag
		self.u_z_b = (- B_ring - u_x*n_b[0] - u_y*n_b[1]) / n_b[2]
		self.u_z_b_f = Function(model.Q, name='u_z_b_f')

		self.u_z_F = + (u_x.dx(0) + u_y.dx(1) + u_z.dx(2))*v_z*dx \
		#             + 1e17*(u_x*n_b[0] + u_y*n_b[1] + u_z*n_b[2])*v_z*dGamma_b

		# generate vertical velocity boundary conditions :
		self.u_z_bcs = []
		if model.N_GAMMA_B_GND != 0:
			self.u_z_bcs.append(DirichletBC(model.Q, self.u_z_b_f, model.ff, \
			                                model.GAMMA_B_GND))
		if model.N_GAMMA_B_FLT != 0:
			self.u_z_bcs.append(DirichletBC(model.Q, self.u_z_b_f, model.ff, \
			                                model.GAMMA_B_FLT))

		# viscous dissipation :
		u       = self.get_velocity()
		epsdot  = self.effective_strain_rate(u)
		if linear:
			s  = "    - using linear form of momentum using model.u in epsdot -"
			self.eta  = self.get_viscosity(model.u)
			self.Vd   = 2 * self.eta * epsdot
		else:
			s  = "    - using nonlinear form of momentum -"
			self.eta  = self.get_viscosity(u)
			self.Vd   = self.get_viscous_dissipation(u)
		print_text(s, cls=self)

		# add lateral boundary conditions :
		mom_bcs  = []
		if not model.use_periodic and model.mark_divide and use_lat_bcs:
			s = "    - using Dirichlet lateral boundary conditions -"
			print_text(s, cls=self)

			mom_bcs.append(DirichletBC(self.Q.sub(0), model.u_x_lat, model.ff, \
			                           model.GAMMA_L_DVD))
			mom_bcs.append(DirichletBC(self.Q.sub(1), model.u_y_lat, model.ff, \
			                           model.GAMMA_L_DVD))
			self.u_z_bcs.append(DirichletBC(model.Q, model.u_z_lat, model.ff, \
			                           model.GAMMA_L_DVD))
		self.set_boundary_conditions(mom_bcs)

		# finally, set up all the other variables and call the child class's
		# ``initialize`` method :
		super(MomentumBPBase, self).__init__(model           = model,
		                                     solve_params    = solve_params,
		                                     linear          = linear,
		                                     use_lat_bcs     = use_lat_bcs,
		                                     use_pressure_bc = use_pressure_bc)

	def strain_rate_tensor(self, u):
		"""
		return the 'Blatter-Pattyn' simplified strain-rate tensor of velocity
		vector ``u``.
		"""
		print_text("    - forming Blatter-Pattyn strain-rate tensor -", cls=self)
		u_x, u_y, u_z  = u
		epi    = 0.5 * (grad(u) + grad(u).T)
		epi02  = 0.5*u_x.dx(2)
		epi12  = 0.5*u_y.dx(2)
		epi22  = -u_x.dx(0) - u_y.dx(1)  # incompressibility
		return as_matrix([[epi[0,0],  epi[0,1],  epi02],
		                  [epi[1,0],  epi[1,1],  epi12],
		                  [epi02,     epi12,     epi22]])

	def quasi_strain_rate_tensor(self, u):
		"""
		return the Dukowicz 2011 quasi-strain tensor.
		"""
		u_x, u_y = u
		epi_ii   = 2*u_x.dx(0) + u_y.dx(1)
		epi_ij   = 0.5 * (u_x.dx(1) + u_y.dx(0))
		epi_ik   = 0.5 * u_x.dx(2)
		epi_jj   = 2*u_y.dx(1) + u_x.dx(0)
		epi_jk   = 0.5 * u_y.dx(2)
		return as_matrix([[epi_ii, epi_ij, epi_ik], \
		                  [epi_ij, epi_jj, epi_jk]])

	def quasi_stress_tensor(self, u, eta):
		"""
		return the Dukowicz 2011 quasi-stress tensor.
		"""
		return 2 * eta * self.quasi_strain_rate_tensor(u)

	def get_velocity(self):
		"""
		Return the velocity :math:`\underline{u} = [u_x\ u_y\ u_z]^{\intercal}`
		with horizontal compoents taken from the unknown function returned by
		:func:`~momentumbp.MomentumBPBase.get_unknown` and zero vertical component.
		"""
		u_x, u_y = self.get_unknown()
		return as_vector([u_x, u_y, 0.0])

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		if (   self.model.energy_dependent_rate_factor == False \
		    or (self.model.energy_dependent_rate_factor == True \
		       and np.unique(self.model.T.vector().get_local()).size == 1)) \
		   and self.model.use_periodic:
			relax = 1.0
		else:
			relax = 0.7

		if self.linear:
			params  = {'linear_solver'       : 'cg',
			           'preconditioner'      : 'hypre_amg'}

		else:
			params = {'newton_solver' :
			         {
			           'linear_solver'            : 'cg',
			           'preconditioner'           : 'hypre_amg',
			           'relative_tolerance'       : 1e-9,
			           'relaxation_parameter'     : relax,
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
		m_params  = {'solver'               : params,
		             'solve_vert_velocity'  : True,
		             'solve_pressure'       : True,
		             'vert_solve_method'    : 'mumps'}
		return m_params

	def solve_pressure(self, annotate=False):
		"""
		Solve for the BP pressure 'p'.
		"""
		model  = self.model

		# solve for vertical velocity :
		s  = "::: solving BP pressure :::"
		print_text(s, cls=self)

		rho_i     = model.rho_i
		g        = model.g
		S        = model.S
		z        = model.x[2]
		eta      = self.eta
		u_x, u_y = self.get_unknown()

		p_f   = rho_i*g*(S - z) - 2*eta*(u_x.dx(0) + u_y.dx(1))
		p     = project(p_f, model.Q, annotate=annotate)

		#p_v   = p.vector().get_local()
		#p_v[p_v < 0.0] = 0.0

		self.assp.assign(model.p, p, annotate=annotate)
		print_min_max(model.p, 'p', cls=self)

	def solve_vert_velocity(self, annotate=False):
		"""
		Solve for the vertical component of velocity :math:`u_z`.
		"""
		s    = "::: solving BP vertical velocity :::"
		print_text(s, cls=self)

		# first, derive the z component of velocity :
		# TODO: avoid projection by wrapping this within an ``Expression``.
		u_z_b = project(self.u_z_b, self.model.Q, annotate=annotate)
		self.model.assign_variable(self.u_z_b_f, u_z_b, cls=self, annotate=annotate)

		model    = self.model
		A        = assemble(lhs(self.u_z_F))
		b        = assemble(rhs(self.u_z_F))
		if len(self.u_z_bcs) > 0:
			for bc in self.u_z_bcs:
				bc.apply(A, b)
		solver = LUSolver(self.solve_params['vert_solve_method'])
		solver.solve(A, self.u_z.vector(), b, annotate=annotate)

		self.assz.assign(model.u.sub(2), self.u_z, annotate=annotate)
		print_min_max(self.u_z, 'w')

	def solve(self, annotate=False):
		"""
		Perform the Newton solve of the first order equations
		"""

		model  = self.model
		params = self.solve_params

		# zero out self.velocity for good convergence for any subsequent solves,
		# e.g. model.L_curve() :
		model.assign_variable(self.get_unknown(), DOLFIN_EPS, \
		                      annotate=annotate, cls=self)

		# solve as defined in ``physics.Physics.solve()`` :
		super(MomentumBPBase, self).solve(annotate)

		# now solve the other parts too :
		if params['solve_vert_velocity']: self.solve_vert_velocity(annotate)
		if params['solve_pressure']:      self.solve_pressure(annotate)

	def update_model_var(self, u, annotate=False):
		"""
		Update the two horizontal components of velocity in ``self.model.u_x``
		to those given by ``u``.
		"""
		self.assu.assign([self.model.u.sub(0), self.model.u.sub(1)], u, \
		                 annotate=annotate)
		print_min_max(self.model.u, 'model.u', cls=self)






class MomentumBP(MomentumBPBase):
	"""
	"""
	def initialize(self, model, solve_params=None,
	               linear=False, use_lat_bcs=False, use_pressure_bc=True):
		"""
		Initializes the class's variables to default values that are then set
		by the individually created model.

		Initilize the residuals and Jacobian for the momentum equations.
		"""
		s = "::: INITIALIZING BP VELOCITY PHYSICS :::"
		print_text(s, cls=self)

		mesh       = model.mesh
		S          = model.S
		z          = model.x[2]
		rhob       = model.rhob  #TODO: implement ``bulk`` density effects.
		rho_i       = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		n          = model.N
		D          = model.D
		B_ring     = model.B_ring

		dOmega     = model.dOmega()
		dOmega_g   = model.dOmega_g()
		dOmega_w   = model.dOmega_w()
		dGamma     = model.dGamma()
		dGamma_bg  = model.dGamma_bg()
		dGamma_bw  = model.dGamma_bw()
		dGamma_b   = model.dGamma_b()
		dGamma_sgu = model.dGamma_sgu()
		dGamma_swu = model.dGamma_swu()
		dGamma_su  = model.dGamma_su()
		dGamma_sg  = model.dGamma_sg()
		dGamma_sw  = model.dGamma_sw()
		dGamma_s   = model.dGamma_s()
		dGamma_ld  = model.dGamma_ld()
		dGamma_lto = model.dGamma_lto()
		dGamma_ltu = model.dGamma_ltu()
		dGamma_lt  = model.dGamma_lt()
		dGamma_l   = model.dGamma_l()
		dGamma_w   = model.dGamma_w()

		# new constants :
		p0     = 101325
		T0     = 288.15
		M      = 0.0289644
		c_i    = model.c_i
		p      = model.p
		R      = model.R

		#===========================================================================
		# define variational problem :

		# momenturm and adjoint :
		u_h    = self.get_unknown()
		du_h   = self.get_trial_function()
		v_h    = self.get_test_function()
		eta    = self.get_viscosity(linear)

		# components of horizontal velocity :
		u_x, u_y = u_h
		v_x, v_y = v_h

		# the third dimension has been integrated out :
		grad_S_h = as_vector([S.dx(0), S.dx(1)])
		n_h      = as_vector([n[0],    n[1]])

		# boundary integral terms :
		p_d    = - 2*eta*(u_x.dx(0) + u_y.dx(1))      # BP dynamic pressure
		p_c    = + rho_i*g*(S - z)                     # cryostatic pressure
		p_i    = + p_c# + p_d                          # ice pressure
		p_w    = + rhosw*g*D                          # water pressure
		p_a    = + p0 * (1 - g*z/(c_i*T0))**(c_i*M/R)  # air pressure
		p_e    = + p_i - p_w# - p_a                   # total exterior pressure

		# collect the quasi-first-order stress tensor :
		sigma  = self.quasi_stress_tensor(u_h, eta)

		# residual :
		resid = + inner(sigma, grad(v_h)) * dOmega \
		        + rho_i * g * dot(grad_S_h, v_h) * dOmega \
		        + beta * dot(u_h, v_h) * dGamma_b \
		        - p_e * dot(n_h, v_h) * dGamma_bw

		# apply water pressure if desired :
		if (not model.use_periodic and use_pressure_bc):
			s = "    - using water pressure lateral boundary condition -"
			print_text(s, cls=self)
			resid -= p_e * dot(n_h, v_h) * dGamma_lt

		# add lateral boundary conditions :
		# FIXME (maybe): need correct BP treatment here
		"""
		if model.N_GAMMA_L_DVD > 0:
			s = "    - using internal divide lateral stress natural boundary" + \
			    " conditions -"
			print_text(s, cls=self)
			eta_l      = self.get_viscosity(linear=True)
			sig_l      = self.quasi_stress_tensor(model.u, model.p, eta_l)
			self.resid += dot(sig_l, n_h) * dGamma_ld
		"""

		# if the model is linear, replace the ``Function`` with a ``TrialFunction``:
		if linear: resid = replace(resid, {u_h : du_h})

		# set this Physics instance's residual :
		self.set_residual(resid)






class MomentumDukowiczBP(MomentumBPBase):
	"""
	"""
	def initialize(self, model, solve_params=None,
		             linear=False, use_lat_bcs=False, use_pressure_bc=True):
		"""
		Initializes the class's variables to default values that are then set
		by the individually created model.

		Initilize the residuals and Jacobian for the momentum equations.
		"""
		s = "::: INITIALIZING DUKOWICZ BP VELOCITY PHYSICS :::"
		print_text(s, cls=self)

		mesh       = model.mesh
		S          = model.S
		z          = model.x[2]
		rho_i       = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		n          = model.N
		D          = model.D

		dOmega     = model.dOmega()
		dOmega_g   = model.dOmega_g()
		dOmega_w   = model.dOmega_w()
		dGamma_bg  = model.dGamma_bg()
		dGamma_bw  = model.dGamma_bw()
		dGamma_b   = model.dGamma_b()
		dGamma_sgu = model.dGamma_sgu()
		dGamma_swu = model.dGamma_swu()
		dGamma_su  = model.dGamma_su()
		dGamma_sg  = model.dGamma_sg()
		dGamma_sw  = model.dGamma_sw()
		dGamma_s   = model.dGamma_s()
		dGamma_ld  = model.dGamma_ld()
		dGamma_lto = model.dGamma_lto()
		dGamma_ltu = model.dGamma_ltu()
		dGamma_lt  = model.dGamma_lt()
		dGamma_l   = model.dGamma_l()

		# new constants :
		p0         = 101325
		T0         = 288.15
		M          = 0.0289644

		#===========================================================================
		# define variational problem :

		# momenturm and adjoint :
		u_h    = self.get_unknown()
		v_h    = self.get_test_function()
		du_h   = self.get_trial_function()
		Vd     = self.get_viscous_dissipation(linear)

		u_x, u_y = u_h
		u_z      = self.u_z
		u        = as_vector([u_x, u_y, u_z])

		# upper surface gradient and lower surface normal :
		grad_S_h = as_vector([S.dx(0), S.dx(1)])
		n_h      = as_vector([n[0],    n[1]])

		# potential energy :
		Pe     = - rho_i * g * dot(u_h, grad_S_h)

		# dissipation by sliding :
		Sl     = - 0.5 * beta * dot(u_h, u_h)

		# pressure boundary :
		f_w    = rho_i*g*(S - z) - rhosw*g*D
		Pb     = f_w * dot(u_h, n_h)

		# action :
		A      = (Vd - Pe)*dOmega - Sl*dGamma_b - Pb*dGamma_bw

		if (not model.use_periodic and use_pressure_bc):
			s = "    - using water pressure lateral boundary condition -"
			print_text(s, cls=self)
			A -= Pb*dGamma_lt

		# add lateral boundary conditions :
		# FIXME: need correct BP treatment here
		if use_lat_bcs:
			s = "    - using internal divide lateral stress natural boundary" + \
			    " conditions -"
			print_text(s, cls=self)
			eta_l      = self.get_viscosity(linear=True)
			sig_l      = self.quasi_stress_tensor(model.u, model.p, eta_l)
			A         -= dot(dot(sig_l, n), u) * dGamma_ld

		# the first variation of the action in the direction of a
		# test function; the extremum :
		resid = derivative(A, u_h, v_h)

		# if the model is linear, replace the ``Function`` with a ``TrialFunction``:
		if linear: resid = replace(resid, {u_h : du_h})

		# set this Physics instance's residual :
		self.set_residual(resid)



