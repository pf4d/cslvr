from dolfin                 import *
from dolfin_adjoint         import *
from copy                   import deepcopy
from cslvr.helper           import raiseNotDefined
from cslvr.inputoutput      import get_text, print_text, print_min_max
from cslvr.physics          import Physics
import numpy                    as np
import matplotlib.pyplot        as plt
import sys
import os
import json


class Momentum(Physics):
	"""
	Abstract class outlines the structure of a momentum calculation.
	"""

	def __new__(self, model, *args, **kwargs):
		"""
		Creates and returns a new momentum object.
		"""
		instance = Physics.__new__(self, model)
		return instance

	def __init__(self, model, solve_params=None,
	             linear=False, use_lat_bcs=False,
	             use_pressure_bc=True, **kwargs):
		"""
		"""
		s = "::: INITIALIZING MOMENTUM :::"
		print_text(s, cls=self)

		self.linear          = linear
		self.use_pressure_bc = use_pressure_bc
		self.use_lat_bcs     = use_lat_bcs

		# save the starting values, as other algorithms might change the
		# values to suit their requirements :
		if isinstance(solve_params, dict):
			pass
		elif solve_params == None:
			self.solve_params    = self.default_solve_params()
			s = "::: using default parameters :::"
			print_text(s, cls=self)
			s = json.dumps(self.solve_params, sort_keys=True, indent=2)
			print_text(s, '230')
		else:
			s = ">>> Momentum REQUIRES A 'dict' INSTANCE OF SOLVER " + \
			    "PARAMETERS, NOT %s <<<"
			print_text(s % type(solve_params) , 'red', 1)
			sys.exit(1)

		self.solve_params_s    = deepcopy(self.solve_params)
		self.linear_s          = linear
		self.use_lat_bcs_s     = use_lat_bcs
		self.use_pressure_bc_s = use_pressure_bc
		self.kwargs            = kwargs

		self.initialize(model           = model,
		                solve_params    = self.solve_params,
		                linear          = self.linear,
		                use_lat_bcs     = self.use_lat_bcs,
		                use_pressure_bc = self.use_pressure_bc,
		                **kwargs)

	def initialize(self, model, solve_params=None,
		             linear=False, use_lat_bcs=False,
		             use_pressure_bc=True, **kwargs):
		"""
		Here we set up the problem, and do all of the differentiation and
		memory allocation type stuff.  Note that any Momentum object *must*
		call this method.  See the existing child Momentum objects for reference.
		"""
		raiseNotDefined()

	def reset(self):
		"""
		reset the momentum to the original configuration.
		"""
		s = "::: RE-INITIALIZING MOMENTUM PHYSICS :::"
		print_text(s, cls=self)

		s = "::: restoring desired Newton solver parameters :::"
		print_text(s, cls=self)
		s = json.dumps(self.solve_params_s, sort_keys=True, indent=2)
		print_text(s, '230')

		self.initialize(model           = self.model,
		                solve_params    = self.solve_params_s,
		                linear          = self.linear_s,
		                use_lat_bcs     = self.use_lat_bcs_s,
		                use_pressure_bc = self.use_pressure_bc_s,
		                **self.kwargs)

	def linearize_viscosity(self, reset_orig_config=True):
		"""
		linearize the viscosity using the velocity stored in ``model.u``.
		"""
		s = "::: RE-INITIALIZING MOMENTUM PHYSICS WITH LINEAR VISCOSITY :::"
		print_text(s, cls=self)

		# deepcopy the parameters so that we can change them without changing
		# the original values we started with :
		mom_params = deepcopy(self.solve_params_s)

		# adjust the parameters for incomplete-adjoint :
		new_params = mom_params['solver']['newton_solver']

		# only affects non-full-stokes formulations :
		mom_params['solve_vert_velocity']     = False
		mom_params['solve_pressure']          = False

		# the linear momentum systems solve much faster :
		new_params['relaxation_parameter']    = 1.0
		new_params['maximum_iterations']      = 2
		new_params['error_on_nonconvergence'] = False

		s = "::: altering solver parameters for optimal convergence :::"
		print_text(s, cls=self)
		s = json.dumps(mom_params, sort_keys=True, indent=2)
		print_text(s, '230')

		# this is useful so that when you call reset(), the viscosity stays
		# linear :
		if reset_orig_config:
			s = "::: reseting the original config to use linear viscosity :::"
			print_text(s, cls=self)
			self.linear_s       = True
			self.solve_params_s = mom_params

		self.initialize(model           = self.model,
		                solve_params    = mom_params,
		                linear          = True,
		                use_lat_bcs     = self.use_lat_bcs_s,
		                use_pressure_bc = self.use_pressure_bc_s,
		                **self.kwargs)

	def color(self):
		"""
		return the default color for this class.
		"""
		return 'cyan'

	def get_velocity(self):
		"""
		Return the velocity :math:`\underline{u}`.
		"""
		raiseNotDefined()

	def strain_rate_tensor(self, u):
		"""
		return the strain-rate tensor of velocity vector ``u``.
		"""
		print_text("    - forming strain-rate tensor -", cls=self)
		return 0.5 * (grad(u) + grad(u).T)

	def get_viscosity(self, linear=False):
		r"""
		calculates and returns the viscosity :math:`\eta` using velocity
		vector ``u`` with components ``u_x``, ``u_y``, ``u_z`` given by

		.. math::

		  \begin{align}
			  \eta(\theta, \rankone{u}) &= \frac{1}{2}A(\theta)^{-\frac{1}{n}} (\dot{\varepsilon}_e(\rankone{u}) + \dot{\varepsilon}_0)^{\frac{1-n}{n}}.
		  \end{align}

		If parameter ``linear == True``, return instead viscosity as a function of
		the velocity :math:`\underline{u}` given by the container of
		``self.model.u``.

		"""
		if linear:
			print_text("    - forming linear viscosity -",     cls=self)
			u   = self.model.u
		else:
			print_text("    - forming non-linear viscosity -", cls=self)
			u = self.get_velocity()
		n       = self.model.n
		A       = self.model.A
		eps_reg = self.model.eps_reg
		epsdot  = self.effective_strain_rate(u)
		return 0.5 * A**(-1/n) * (epsdot + eps_reg)**((1-n)/(2*n))

	def get_viscous_dissipation(self, linear=False):
		r"""
		Returns the Dukowicz viscous dissipation term :math:`V`
		using velocity vector `with components ``u_x``, ``u_y``, ``u_z`` returned
		by :func:`~momentum.Momentum.viscosity` given by

		.. math::

		  \begin{align}
		    V\left( \dot{\varepsilon}_e^2 \right) = \frac{2n}{n+1} A^{-\frac{1}{n}} \left( \dot{\varepsilon}_e^2 \right)^{\frac{n+1}{2n}}.
		  \end{align}

		If parameter ``linear == True``, return instead

		.. math::

		  \begin{align}
		    V\left( \dot{\varepsilon}_e^2 \right) = 2 \eta(\underline{u}_{\ell}) \dot{\varepsilon}_e^2(\underline{u})
		  \end{align}

		where :math:`\underline{u}_{\ell}` is the velocity container of
		``self.model.u``.

		"""
		epsdot = self.effective_strain_rate(self.get_velocity())
		if linear:
			print_text("    - forming linear viscous dissipation -",     cls=self)
			eta = self.get_viscosity(linear)
			V   = 2 * eta * epsdot
		else:
			print_text("    - forming non-linear viscous dissipation -", cls=self)
			n        = self.model.n
			A        = self.model.A
			eps_reg  = self.model.eps_reg
			V        = (2*n)/(n+1) * A**(-1/n) * (epsdot + eps_reg)**((n+1)/(2*n))
		return V

	def deviatoric_stress_tensor(self, u, eta):
		"""
		return the deviatoric stress tensor.
		"""
		s   = "::: forming the deviatoric part of the Cauchy stress tensor :::"
		print_text(s, cls=self)

		return 2 * eta * self.strain_rate_tensor(u)

	def stress_tensor(self, u, p, eta):
		"""
		return the Cauchy stress tensor.
		"""
		s   = "::: forming the Cauchy stress tensor :::"
		print_text(s, cls=self)

		I     = Identity(self.model.dim)
		tau   = self.deviatoric_stress_tensor(u, eta)

		return tau - p*I

	def effective_strain_rate(self, u):
		"""
		return the BP effective strain rate squared from velocity vector ``u``.
		"""
		epi    = self.strain_rate_tensor(u)
		epsdot = 0.5 * tr(dot(epi, epi))
		return epsdot

	def effective_stress(self, U, eta):
		"""
		return the effective stress squared.
		"""
		tau    = self.deviatoric_stress_tensor(U, eta)
		taudot = 0.5 * tr(dot(tau, tau))
		return taudot

	def add_compensatory_forcing_terms(self, verification):
		"""
		"""
		print_text("::: adding compensatory forcing terms:::", cls=self)

		# initialize the appropriate compensatary forcing terms :
		verification.init_r3_stress_balance(momentum=self)

		# get the interior compensatory forcing terms :
		f_int  = verification.get_compensatory_interior_rhs()

		# get the exterior compensatory forcing terms :
		f_ext_s = verification.get_compensatory_upper_surface_exterior_rhs()
		f_ext_b = verification.get_compensatory_lower_surface_exterior_rhs()
		sigma   = verification.get_stress_tensor()

		# get interior and exterior measures :
		dOmega   = self.model.dOmega()
		dGamma   = self.model.dGamma()
		dGamma_s = self.model.dGamma_s()
		dGamma_b = self.model.dGamma_b()
		dGamma_l = self.model.dGamma_l()
		n        = self.model.N

		# get the test function :
		Phi   = self.get_test_function()

		# create velocity test function vector :
		v     = as_vector([Phi[0], Phi[1], Phi[2]])

		# get the modified momentum residual :
		resid = self.get_residual() \
		        - dot(f_int,         v) * dOmega \
		        - dot(f_ext_b,       v) * dGamma_b \
		        - dot(f_ext_s,       v) * dGamma_s \
		        - dot(dot(sigma, n), v) * dGamma_l \
		        #- dot(dot(sigma, n), v) * dGamma \

		# replace the momentum residual and Jacobian :
		self.set_residual(resid)

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance.
		"""
		if (   self.model.energy_dependent_rate_factor == False \
		    or (self.model.energy_dependent_rate_factor == True \
		       and np.unique(self.model.T.vector().get_local()).size == 1)) \
		   and self.model.use_periodic:
			relax = 1.0
		else:
			relax = 0.7

		if self.linear:
			params  = {'linear_solver'       : 'mumps',
			           'preconditioner'      : 'none'}

		else:
			params = {'newton_solver' :
			         {
			           'linear_solver'            : 'mumps',
			           'relative_tolerance'       : 1e-9,
			           'relaxation_parameter'     : relax,
			           'maximum_iterations'       : 25,
			           'error_on_nonconvergence'  : False,
			         }}
		m_params  = {'solver'      : params}
		return m_params

	def solve_pressure(self, annotate=False):
		"""
		Solve for the hydrostatic pressure 'p'.
		"""
		self.model.solve_hydrostatic_pressure(annotate)

	def Lagrangian(self):
		"""
		Returns the Lagrangian of the momentum equations.
		"""
		s  = "::: forming Lagrangian :::"
		print_text(s, cls=self)

		R   = self.get_residual()
		Phi = self.get_test_function()
		trial_function  = self.get_trial_function()

		# this is the adjoint of the momentum residual, the Lagrangian :
		return self.J + replace(R, {Phi : trial_function})

	def dLdc(self, L, c):
		r"""
		Returns the derivative of the Lagrangian consisting of adjoint-computed
		self.Lam values w.r.t. the control variable ``c``, i.e.,

		.. math::

		   \frac{\mathrm{d} L}{\mathrm{d} c} = \frac{\mathrm{d}}{\mathrm{d} c} \mathscr{L} \left( \lambda \right),

		where :math:`\mathscr{L}` is the Lagrangian computed by :func:`~momentum.Momentum.Lagrangian` and :math:`\lambda` is the adjoint variable.

		"""
		s  = "::: forming dLdc :::"
		print_text(s, cls=self)

		trial_function  = self.get_trial_function()
		Lam = self.get_Lam()

		# we need to evaluate the Lagrangian with the values of Lam computed from
		# self.dI in order to get the derivative of the Lagrangian w.r.t. the
		# control variables.  Hence we need a new Lagrangian with the trial
		# functions replaced with the computed Lam values.
		L_lam  = replace(L, {trial_function : Lam})

		# the Lagrangian with unknowns replaced with computed Lam :
		H_lam  = self.J + L_lam

		# the derivative of the Hamiltonian w.r.t. the control variables in the
		# direction of a test function :
		return derivative(H_lam, c, TestFunction(self.model.Q))

	def solve_adjoint_momentum(self, H):
		"""
		Solves for the adjoint variables ``self.Lam`` from the Hamiltonian <H>.
		"""
		U   = self.get_unknown()
		Phi = self.get_test_function()
		Lam = self.get_Lam()

		# we desire the derivative of the Lagrangian w.r.t. the model state U
		# in the direction of the test function Phi to vanish :
		dI = derivative(H, U, Phi)

		s  = "::: solving adjoint momentum :::"
		print_text(s, cls=self)

		aw = assemble(lhs(dI))
		Lw = assemble(rhs(dI))

		a_solver = KrylovSolver('cg', 'hypre_amg')
		a_solver.solve(aw, Lam.vector(), Lw, annotate=False)

		print_min_max(Lam, 'Lam')



