from __future__ import division
from __future__ import print_function
from builtins import range
from dolfin                 import *
from dolfin_adjoint         import *
from cslvr.inputoutput      import get_text, print_text, print_min_max
from cslvr.d3model          import D3Model
from cslvr.d2model          import D2Model
from cslvr.d1model          import D1Model
from cslvr.physics          import Physics
from cslvr.helper           import VerticalBasis, VerticalFDBasis, \
                                   raiseNotDefined
from copy                   import deepcopy
import numpy                    as np
import matplotlib.pyplot        as plt
import sys
import os
import json






class Energy(Physics):
	"""
	Abstract class outlines the structure of an energy conservation.
	"""

	def __new__(self, model, *args, **kwargs):
		"""
		Creates and returns a new Energy object.
		"""
		instance = Physics.__new__(self, model)
		return instance

	# TODO: `energy_flux_mode` and `stabilization_method` are specific to the
	#       D3Model energy solvers.
	def __init__(self, model, momentum,
	             solve_params         = None,
	             transient            = False,
	             use_lat_bc           = False,
	             energy_flux_mode     = 'B_ring',
	             stabilization_method = 'GLS'):
		"""
		"""
		s    = "::: INITIALIZING ENERGY :::"
		print_text(s, cls=self)
		# save the starting values, as other algorithms might change the
		# values to suit their requirements :
		if isinstance(solve_params, dict):
			pass
		elif solve_params == None:
			solve_params    = self.default_solve_params()
			s = "::: using default parameters :::"
			print_text(s, cls=self)
			s = json.dumps(solve_params, sort_keys=True, indent=2)
			print_text(s, '230')
		else:
			s = ">>> Energy REQUIRES A 'dict' INSTANCE OF SOLVER " + \
			    "PARAMETERS, NOT %s <<<"
			print_text(s % type(solve_params) , 'red', 1)
			sys.exit(1)

		self.momentum_s             = momentum
		self.solve_params_s         = deepcopy(solve_params)
		self.transient_s            = transient
		self.use_lat_bc_s           = use_lat_bc
		self.energy_flux_mode_s     = energy_flux_mode
		self.stabilization_method_s = stabilization_method

		self.T_ini     = self.model.T.copy(True)
		self.W_ini     = self.model.W.copy(True)

		self.initialize(model, momentum, solve_params, transient,
		                use_lat_bc, energy_flux_mode, stabilization_method)

	def initialize(self, model, momentum, solve_params=None, transient=False,
	               use_lat_bc=False, energy_flux_mode='B_ring',
	               stabilization_method='GLS', reset=False):
		"""
		Here we set up the problem, and do all of the differentiation and
		memory allocation type stuff.  Note that any Energy object *must*
		call this method.  See the existing child Energy objects for reference.
		"""
		raiseNotDefined()

	def make_transient(self, time_step):
		"""
		set the energy system to transient form.
		"""
		s = "::: RE-INITIALIZING ENERGY PHYSICS WITH TRANSIENT FORM :::"
		print_text(s, cls=self)

		self.model.init_time_step(time_step)

		self.initialize(model                = self.model,
		                momentum             = self.momentum_s,
		                solve_params         = self.solve_params_s,
		                transient            = True,
		                use_lat_bc           = self.use_lat_bc_s,
		                energy_flux_mode     = self.energy_flux_mode_s,
		                stabilization_method = self.stabilization_method_s,
		                reset                = True)

	def make_steady_state(self):
		"""
		set the energy system to steady-state form.
		"""
		s = "::: RE-INITIALIZING ENERGY PHYSICS WITH STEADY-STATE FORM :::"
		print_text(s, cls=self)

		self.initialize(model                = self.model,
		                momentum             = self.momentum_s,
		                solve_params         = self.solve_params_s,
		                transient            = False,
		                use_lat_bc           = self.use_lat_bc_s,
		                energy_flux_mode     = self.energy_flux_mode_s,
		                stabilization_method = self.stabilization_method_s,
		                reset                = True)

	def set_basal_flux_mode(self, mode):
		"""
		reset the energy system to use zero energy basal flux.
		"""
		s = "::: RE-INITIALIZING ENERGY PHYSICS NEUMANN BASAL BC TO " + \
		    "\'%s\' :::" % mode
		print_text(s, cls=self)

		self.initialize(model                = self.model,
		                momentum             = self.momentum_s,
		                solve_params         = self.solve_params_s,
		                transient            = self.transient_s,
		                use_lat_bc           = self.use_lat_bc_s,
		                energy_flux_mode     = mode,
		                stabilization_method = self.stabilization_method_s,
		                reset                = True)

	def reset(self):
		"""
		reset the energy system to the original configuration.
		"""
		s = "::: RE-INITIALIZING ENERGY PHYSICS :::"
		print_text(s, cls=self)

		self.model.init_T(self.T_ini)
		self.model.init_W(self.W_ini)

		self.initialize(model                = self.model,
		                momentum             = self.momentum_s,
		                solve_params         = self.solve_params_s,
		                transient            = self.transient_s,
		                use_lat_bc           = self.use_lat_bc_s,
		                energy_flux_mode     = self.energy_flux_mode_s,
		                stabilization_method = self.stabilization_method_s,
		                reset                = True)

	def color(self):
		"""
		return the default color for this class.
		"""
		return '213'

	def get_ice_thermal_conductivity(self):
		"""
		Returns the thermal conductivity for ice.
		"""
		return self.model.spy * 9.828 * exp(-0.0057*self.model.T)

	def get_ice_heat_capacity(self):
		"""
		"""
		return 146.3 + 7.253*self.model.T

	def get_bulk_thermal_conductivity(self):
		"""
		"""
		k_i    = self.get_ice_thermal_conductivity()
		k_w    = self.model.spy * self.model.k_w
		W      = self.model.W
		return (1-W)*k_i + W*k_w

	def get_bulk_heat_capacity(self):
		"""
		"""
		c_i    = self.get_ice_heat_capacity()
		c_w    = self.model.c_w
		W      = self.model.W
		return (1-W)*c_i + W*c_w

	def get_bulk_density(self):
		"""
		"""
		W      = self.model.W
		rho_i  = self.model.rho_i
		rho_w  = self.model.rho_w
		return (1-W)*rho_i + W*rho_w

	def get_enthalpy_gradient_conductivity(self):
		"""
		"""
		# coefficient for non-advective water flux (enthalpy-gradient) :
		k_c  = conditional( gt(self.model.W, 0.0), self.model.k_0, 1 )
		k    = self.get_bulk_thermal_conductivity()
		return k_c * k

	def get_enthalpy_gradient_diffusivity(self):
		"""
		"""
		c     = self.get_bulk_heat_capacity()
		rho   = self.get_bulk_density()
		kappa = self.get_enthalpy_gradient_conductivity()
		return kappa / (rho*c)

	def get_grid_peclet_number(self):
		r"""
		Returns the grid P\'{e}clet number.
		"""
		rho    = self.get_bulk_density()
		c      = self.get_bulk_heat_capacity()
		kappa  = self.get_enthalpy_gradient_conductivity()
		u      = self.momentum.get_velocity()
		h      = self.model.h
		ut     = rho*u - grad(kappa/c)
		u_norm = sqrt(dot(ut, ut) + DOLFIN_EPS)
		return u_norm*h / (2*kappa/c)

	def get_temperature_flux_vector(self):
		"""
		Returns the temperature flux vector.
		"""
		T = self.model.T
		k = self.get_bulk_thermal_conductivity()
		return k * grad(T)

	def get_temperature_melting_flux_vector(self):
		"""
		Returns the temperature-melting flux vector.
		"""
		Tm = self.model.T_melt
		k  = self.get_bulk_thermal_conductivity()
		return k * grad(Tm)

	def get_basal_melting_rate(self):
		"""
		Returns the basal melting rate.
		"""
		q_geo      = self.model.q_geo    # geothermal heat
		q_fric     = self.model.q_fric   # friction heat
		L_f        = self.model.L_f        # latent heat of freezing
		rho_b      = self.model.rhob     # bulk density
		n_b        = self.model.n_b      # outward unit normal on B
		T          = self.model.T        # temperature
		k          = self.get_ice_thermal_conductivity()
		q          = k * grad(T)         # heat flux
		return (q_geo + q_fric - dot(q, n_b)) / (L_f * rho_b)

	def get_internal_friction_heat(self):
		"""
		Retuns the internal friction heat; the strain heating.
		"""
		# collect the velocity vector to be of the same dimension os the model :
		u      = self.momentum.get_velocity()
		epsdot = self.momentum.effective_strain_rate(u) + self.model.eps_reg
		eta    = self.momentum.get_viscosity(u)
		return 4 * eta * epsdot

	def get_external_friction_heat(self):
		r"""
		Retuns the external friction heat over the lower surface given by

		.. math::

		  \begin{align}
		    q_{\mathrm{fric}} = \beta \underline{u}_{\Vert} \cdot \underline{u}_{\Vert}
		  \end{align}

		with tangential component of velocity
		:math:`\underline{u}_{\Vert} = \underline{u} - (\underline{u} \cdot \hat{\underline{n}} ) \hat{\underline{n}}`.
		"""
		u       = self.momentum.get_velocity()     # velocity
		B_ring  = self.model.B_ring                # lower suface-mass balance
		beta    = self.model.beta                  # friction coefficient
		n_b     = self.model.n_b                   # outward unit normal on B
		u_t     = u - dot(u,n_b) * n_b             # tangential component of u
		return beta * dot(u_t, u_t)

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		params  = {'solver'              : 'mumps',
		           'use_surface_climate' : False}
		return params

	def solve_surface_climate(self):
		"""
		Calculates PDD, surface temperature given current model geometry and
		saves to model.T_surface.
		"""
		s    = "::: solving surface climate :::"
		print_text(s, cls=self)
		model = self.model

		T_w   = model.T_w(0)
		S     = model.S.vector().get_local()
		lat   = model.lat.vector().get_local()
		lon   = model.lon.vector().get_local()

		# greenland :
		Tn    = 41.83 - 6.309e-3*S - 0.7189*lat - 0.0672*lon + T_w

		## antarctica :
		#Tn    = 34.46 - 0.00914*S - 0.27974*lat

		# Apply the lapse rate to the surface boundary condition
		model.init_T_surface(Tn)

	def adjust_S_ring(self):
		"""
		"""
		s    = "::: solving surface accumulation/ablation :::"
		print_text(s, cls=self)
		model = self.model

		T_w     = model.T_w(0)
		T       = model.T_surface.vector().get_local()

		S_ring  = 2.5 * 2**((T-T_w)/10)

		if model.N_OMEGA_FLT > 0:
			shf_dofs = np.where(model.mask.vector().get_local() == 0.0)[0]
			S_ring[model.shf_dofs] = -100

		model.init_S_ring(S_ring)

	def form_cost_ftn(self, kind='abs'):
		"""
		Forms and returns a cost functional for use with adjoint.
		Saves to self.J.
		"""
		s   = "::: forming water-optimization cost functional :::"
		print_text(s, cls=self)

		model     = self.model
		theta     = self.get_unknown()
		thetam    = model.theta
		dGamma_bg = model.dGamma_bg()
		theta_c   = model.theta_melt + model.Wc*model.L_f

		if kind == 'TV':
			self.J   = sqrt((theta  - theta_c)**2 + 1e-15) * dGamma_bg
			self.Jp  = sqrt((thetam - theta_c)**2 + 1e-15) * dGamma_bg
			s   = "    - using TV cost functional :::"
		elif kind == 'L2':
			self.J   = 0.5 * (theta  - theta_c)**2 * dGamma_bg
			self.Jp  = 0.5 * (thetam - theta_c)**2 * dGamma_bg
			s   = "    - using L2 cost functional :::"
		elif kind == 'abs':
			self.J   = abs(theta  - theta_c) * dGamma_bg
			self.Jp  = abs(thetam - theta_c) * dGamma_bg
			s   = "    - using absolute value objective functional :::"
		else:
			s = ">>> ADJOINT OBJECTIVE FUNCTIONAL MAY BE 'TV', 'L2' " + \
			    "or 'abs', NOT '%s' <<<" % kind
			print_text(s, 'red', 1)
			sys.exit(1)
		print_text(s, cls=self)

	def calc_misfit(self):
		"""
		Calculates the misfit,
		"""
		s   = "::: calculating misfit L-infty norm ||theta - theta_c|| :::"
		print_text(s, cls=self)

		model   = self.model

		# set up functions for surface (s) and current objective (o) :
		theta_s = Function(model.Q)
		theta_o = Function(model.Q)

		# calculate L_inf norm :
		theta_v   = model.theta.vector().get_local()
		theta_m_v = model.theta_melt.vector().get_local()
		Wc_v      = model.Wc.vector().get_local()
		theta_c_v = theta_m_v + Wc_v * model.L_f(0)
		theta_o.vector().set_local(np.abs(theta_v - theta_c_v))
		theta_o.vector().apply('insert')

		# apply difference over only grounded surface :
		bc_theta  = DirichletBC(model.Q, theta_o, model.ff, model.GAMMA_B_GND)
		bc_theta.apply(theta_s.vector())

		# calculate L_inf vector norm :
		D        = MPI.max(mpi_comm_world(), theta_s.vector().max())

		s    = "||theta - theta_c|| : %.3E" % D
		print_text(s, '208', 1)
		return D

	def calc_functionals(self):
		"""
		Used to facilitate printing the objective function in adjoint solves.
		"""
		try:
			R = assemble(self.Rp, annotate=False)
		except AttributeError:
			R = 0.0
		J = assemble(self.Jp, annotate=False)
		print_min_max(R, 'R')
		print_min_max(J, 'J')
		return (R, J)

	def calc_obj(self):
		"""
		Used to facilitate printing the objective function in adjoint solves.
		"""
		J = assemble(self.Jp, annotate=False)
		print_min_max(J, 'J')
		return J

	def partition_energy(self, annotate=False):
		"""
		solve for the water content model.W and temperature model.T.
		"""
		# TODO: the operation below breaks dolfin-adjoint annotation.
		# temperature solved with quadradic formula, using expression for c :
		s = "::: calculating temperature :::"
		print_text(s, cls=self)

		model    = self.model
		T_w      = model.T_w(0)

		# temperature is a quadradic function of energy :
		theta_v  = model.theta.vector().get_local()
		T_n_v    = (-146.3 + np.sqrt(146.3**2 + 2*7.253*theta_v)) / 7.253
		T_v      = T_n_v.copy()
		Tp_v     = T_n_v.copy()

		# create pressure-adjusted temperature for rate-factor :
		Tp_v[Tp_v > T_w] = T_w
		model.init_Tp(Tp_v)

		# correct for the pressure-melting point :
		T_melt_v     = model.T_melt.vector().get_local()
		theta_melt_v = model.theta_melt.vector().get_local()
		warm         = theta_v >= theta_melt_v
		cold         = theta_v <  theta_melt_v
		T_v[warm]    = T_melt_v[warm]
		model.init_T(T_v)

		# water content solved diagnostically :
		s = "::: calculating water content :::"
		print_text(s, cls=self)
		W_v  = (theta_v - theta_melt_v) / model.L_f(0)

		# update water content :
		W_v[W_v < 0.0]  = 0.0    # no water where frozen, please.
		W_v[W_v > 1.0]  = 1.0    # no hot water, please.
		model.assign_variable(model.W0,  model.W)
		model.init_W(W_v)

	def optimize_water_flux(self, max_iter, bounds=(-1e8, 0), method='ipopt',
	                        adj_save_vars=None, adj_callback=None):
		"""
		determine the correct basal-mass balance saved (currently) to
		``model.B_ring``.
		"""
		s    = '::: optimizing for water-flux in %i maximum iterations :::'
		print_text(s % max_iter, cls=self)

		model = self.model

		# reset entire dolfin-adjoint state :
		adj_reset()

		# starting time :
		t0   = time()

		# need this for the derivative callback :
		global counter
		counter = 0

		# functional lists to be populated :
		global Rs, Js, Ds
		Rs = []
		Js = []
		Ds = []

		# now solve the control optimization problem :
		s    = "::: starting adjoint-control optimization with method '%s' :::"
		print_text(s % method, cls=self)

		def eval_cb(I, B_ring):
			s    = '::: adjoint objective eval post callback function :::'
			print_text(s, cls=self)
			print_min_max(I,  'I')
			print_min_max(B_ring, 'B_ring')

		# objective gradient callback function :
		def deriv_cb(I, dI, B_ring):
			global counter, Rs, Js
			if method == 'ipopt':
				s0    = '>>> '
				s1    = 'iteration %i (max %i) complete'
				s2    = ' <<<'
				text0 = get_text(s0, 'red', 1)
				text1 = get_text(s1 % (counter, max_iter), 'red')
				text2 = get_text(s2, 'red', 1)
				if MPI.rank(mpi_comm_world())==0:
					print(text0 + text1 + text2)
				counter += 1
			s    = '::: adjoint obj. gradient post callback function :::'
			print_text(s, cls=self)
			print_min_max(dI,    'dI/B_ring')

			# update the DA current velocity to the model for evaluation
			# purposes only; the model.assign_variable function is
			# annotated for purposes of linking physics models to the adjoint
			# process :
			theta_opt = DolfinAdjointVariable(model.theta).tape_value()
			model.init_theta(theta_opt)

			# print functional values :
			model.B_ring.assign(B_ring, annotate=False)
			ftnls = self.calc_functionals()
			D     = self.calc_misfit()

			# functional lists to be populated :
			Rs.append(ftnls[0])
			Js.append(ftnls[1])
			Ds.append(D)

			# call that callback, if you want :
			if adj_callback is not None:
				adj_callback(I, dI, B_ring)

		# solve the momentum equations with annotation enabled :
		s    = '::: solving forward problem for dolfin-adjoint annotatation :::'
		print_text(s, cls=self)
		self.solve(annotate=True)

		# get the cost, regularization, and objective functionals :
		I = self.J
		try:
			I += self.R
		except AttributeError:
			print_text('    - not using regularization -', cls=self)

		# define the control variable :
		m = Control(model.B_ring, value=model.B_ring)

		# state the minimization problem :
		F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
		                      derivative_cb_post=deriv_cb)

		# optimize with scipy's fmin_l_bfgs_b :
		if method == 'l_bfgs_b':
			out = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=bounds,
			               options={"disp"    : True,
			                        "maxiter" : max_iter,
			                        "gtol"    : 1e-5})
			B_ring_opt = out[0]

		# or optimize with IPOpt (preferred) :
		elif method == 'ipopt':
			try:
				import pyipopt
			except ImportError:
				info_red("""You do not have IPOPT and/or pyipopt installed.
				            When compiling IPOPT, make sure to link against HSL,
				            as it is a necessity for practical problems.""")
				raise
			problem = MinimizationProblem(F, bounds=bounds)
			parameters = {"tol"                : 1e-8,
			              "acceptable_tol"     : 1e-6,
			              "maximum_iterations" : max_iter,
			              "print_level"        : 5,
			              "ma97_order"         : "metis",
			              "ma86_order"         : "metis",
			              "linear_solver"      : "ma57"}
			solver = IPOPTSolver(problem, parameters=parameters)
			B_ring_opt  = solver.solve()

		# let's see it :
		print_min_max(B_ring_opt, 'B_ring_opt')

		# extrude the flux up and make the optimal control variable available :
		B_ring_ext = model.vert_extrude(B_ring_opt, d='up')
		model.init_B_ring(B_ring_ext)
		#Control(model.B_ring).update(B_ring_ext)  # FIXME: does this work?

		# save state to unique hdf5 file :
		if isinstance(adj_save_vars, list):
			s    = '::: saving variables in list arg adj_save_vars :::'
			print_text(s, cls=self)
			out_file = model.out_dir + 'w_opt.h5'
			foutput  = HDF5File(mpi_comm_world(), out_file, 'w')
			for var in adj_save_vars:
				model.save_hdf5(var, f=foutput)
			foutput.close()

		# calculate total time to compute
		tf = time()
		s  = tf - t0
		m  = s / 60.0
		h  = m / 60.0
		s  = s % 60
		m  = m % 60
		text = "time to optimize for water flux: %02d:%02d:%02d" % (h,m,s)
		print_text(text, 'red', 1)

		# save all the objective functional values :
		d    = model.out_dir + 'objective_ftnls_history/'
		s    = '::: saving objective functionals to %s :::'
		print_text(s % d, cls=self)
		if model.MPI_rank==0:
			if not os.path.exists(d):
				os.makedirs(d)
			np.savetxt(d + 'time.txt', np.array([tf - t0]))
			np.savetxt(d + 'Rs.txt',   np.array(Rs))
			np.savetxt(d + 'Js.txt',   np.array(Js))
			np.savetxt(d + 'Ds.txt',   np.array(Ds))

			fig = plt.figure()
			ax  = fig.add_subplot(111)
			#ax.set_yscale('log')
			ax.set_ylabel(r'$\mathscr{J}\left(\theta\right)$')
			ax.set_xlabel(r'iteration')
			ax.plot(np.array(Js), 'r-', lw=2.0)
			plt.grid()
			plt.savefig(d + 'J.png', dpi=100)
			plt.close(fig)

			try:
				R = self.R
				fig = plt.figure()
				ax  = fig.add_subplot(111)
				ax.set_yscale('log')
				ax.set_ylabel(r'$\mathscr{R}\left(\alpha\right)$')
				ax.set_xlabel(r'iteration')
				ax.plot(np.array(Rs), 'r-', lw=2.0)
				plt.grid()
				plt.savefig(d + 'R.png', dpi=100)
				plt.close(fig)
			except AttributeError:
				pass

			fig = plt.figure()
			ax  = fig.add_subplot(111)
			#ax.set_yscale('log')
			ax.set_ylabel(r'$\mathscr{D}\left(\theta\right)$')
			ax.set_xlabel(r'iteration')
			ax.plot(np.array(Ds), 'r-', lw=2.0)
			plt.grid()
			plt.savefig(d + 'D.png', dpi=100)
			plt.close(fig)

	def calc_bulk_density(self):
		"""
		Calculate the bulk density stored in ``model.rhob``.
		"""
		# calculate bulk density :
		s = "::: calculating bulk density :::"
		print_text(s, cls=self)
		model       = self.model
		rhob        = project(self.rho, annotate=False)
		model.assign_variable(model.rhob, rhob)






class Enthalpy(Energy):
	"""
	"""
	def initialize(self, model, momentum,
	               solve_params         = None,
	               transient            = False,
	               use_lat_bc           = False,
	               energy_flux_mode     = 'B_ring',
	               stabilization_method = 'GLS',
	               reset                = False):
		"""
		Set up energy equation residual.
		"""
		self.transient = transient

		s    = "::: INITIALIZING ENTHALPY PHYSICS :::"
		print_text(s, cls=self)

		# save the solver parameters and momentum instance :
		self.solve_params = solve_params
		self.momentum     = momentum
		self.linear       = True

		# save the state of basal boundary flux :
		self.energy_flux_mode = energy_flux_mode

		# create a facet function for temperate zone :
		self.ff      = MeshFunction('size_t', model.mesh, 2, 0)

		mesh          = model.mesh
		T             = model.T
		alpha         = model.alpha
		rho_w         = model.rho_w
		L_f           = model.L_f
		W             = model.W
		T_m           = model.T_melt
		B_ring        = model.B_ring
		T_surface     = model.T_surface
		theta_surface = model.theta_surface
		theta_float   = model.theta_float
		theta_app     = model.theta_app
		theta_0       = model.theta
		q_geo         = model.q_geo
		h             = model.h
		dt            = model.time_step
		dOmega        = model.dOmega()
		dGamma_bg     = model.dGamma_bg()

		self.Q = model.Q

		self.ass_theta = FunctionAssigner(model.Q, self.Q)

		self.set_unknown(Function(self.Q, name='energy.theta'))
		self.set_trial_function(TrialFunction(self.Q))
		self.set_test_function(TestFunction(self.Q))

		# define test and trial functions :
		psi    = self.get_test_function()
		dtheta = self.get_trial_function()
		theta  = self.get_unknown()

		# internal friction (strain heat) :
		Q      = self.get_internal_friction_heat()

		# velocity :
		u      = momentum.get_velocity()

		# bulk properties :
		c      = self.get_bulk_heat_capacity()
		k      = self.get_bulk_thermal_conductivity()
		rho    = self.get_bulk_density()

		# discontinuous with water, J/(a*m*K) :
		kappa  = self.get_enthalpy_gradient_conductivity()

		# bulk enthalpy-gradient diffusivity
		Xi     =  self.get_enthalpy_gradient_diffusivity()

		# frictional heating :
		q_fric = self.get_external_friction_heat()

		# basal heat-flux natural boundary condition :
		q_tm = self.get_temperature_melting_flux_vector()
		n    = model.N
		g_w  = dot(q_tm, n) + rho_w*L_f*B_ring
		g_n  = q_geo + q_fric
		if energy_flux_mode == 'zero_energy' or energy_flux_mode == 'B_ring':
			s = "    - using B_ring-energy flux boundary condition -"
			print_text(s, cls=self)
			g_b  = g_n - alpha*g_w
		elif energy_flux_mode == 'temperate_zone_mark':
			s = "    - using temperate-zone mark energy flux boundary condition -"
			print_text(s, cls=self)
			g_b  = g_n
		else:
			s = ">>> PARAMETER 'energy_flux_mode' MAY BE 'zero_energy', " + \
			    "'B_ring', or 'temperate_zone_mark', NOT '%s' <<<"
			print_text(s % energy_flux_mode , 'red', 1)
			sys.exit(1)

		# configure the module to run in steady state :
		if not transient:
			print_text("    - using steady-state formulation -", cls=self)
			nu = 1.0
		else:
			print_text("    - using transient formulation -", cls=self)
			nu = 0.5

		# form time-interpolated unknown :
		theta_mid = nu*dtheta + (1 - nu)*theta_0

		# quasi-velocity (see Cummings et al., 2016)
		ut      = rho*u - grad(kappa/c)
		ut_norm = sqrt(dot(ut, ut) + DOLFIN_EPS)

		# the Peclet number :
		Pe      = self.get_grid_peclet_number()

		# for linear elements :
		if model.order == 1:
		 xi    = 1/tanh(Pe) - 1/Pe

		# for quadradic elements :
		elif model.order == 2:
			xi_1  = 0.5*(1/tanh(Pe) - 2/Pe)
			xi    =     ((3 + 3*Pe*xi_1)*tanh(Pe) - (3*Pe + Pe**2*xi_1)) \
				       /  ((2 - 3*xi_1*tanh(Pe))*Pe**2)

		# intrinsic time parameter :
		tau     = h*xi / (2 * ut_norm)
		psihat  = psi + tau * dot(ut, grad(psi))

		# the linear differential operator for this problem :
		def Lu(theta):
			return + rho * dot(u, grad(theta)) \
			       - kappa/c * div(grad(theta)) \
			       - dot(grad(kappa/c), grad(theta))

		# the advective part of the operator :
		def L_adv(theta):
			return dot(ut, grad(theta))

		# the adjoint of the operator :
		def L_star(theta):
			return - dot(u, grad(theta)) \
			       - Xi * div(grad(theta)) \
			       + 1/rho * dot(grad(kappa/c), grad(theta))

		# use streamline-upwind/Petrov-Galerkin stabilization :
		if stabilization_method == 'SUPG':
			s      = "    - using streamline-upwind/Petrov-Galerkin stabilization -"
			LL     = lambda x: + L_adv(x)
		# use Galerkin/least-squares stabilization :
		elif stabilization_method == 'GLS':
			s      = "    - using Galerkin/least-squares stabilization -"
			LL     = lambda x: + Lu(x)
		# use subgrid-scale-model stabilization :
		elif stabilization_method == 'SSM':
			s      = "    - using subgrid-scale-model stabilization -"
			LL     = lambda x: - L_star(x)
		print_text(s, cls=self)

		# form the residual :
		resid = + rho * dot(u, grad(theta_mid)) * psi * dOmega \
			      + kappa/c * inner(grad(psi), grad(theta_mid)) * dOmega \
			      - dot(grad(kappa/c), grad(theta_mid)) * psi * dOmega \
			      - g_b * psi * dGamma_bg \
			      - Q * psi * dOmega \
			      + inner(LL(psi), tau*(Lu(theta_mid) - Q)) * dOmega \

		# add the time derivative term if transient :
		if transient:
			resid += rho * (dtheta - theta_0) / dt * psi * dOmega

		# set this Physics instance's residual, left-, and right-hand sides :
		self.set_residual(resid)

		# surface boundary condition :
		theta_bcs = []
		if model.N_GAMMA_S_GND > 0:
			theta_bcs.append( DirichletBC(self.Q, theta_surface,
			                              model.ff, model.GAMMA_S_GND) )
		if model.N_GAMMA_U_GND > 0:
			theta_bcs.append( DirichletBC(self.Q, theta_surface,
			                              model.ff, model.GAMMA_U_GND) )
		if model.N_GAMMA_S_FLT > 0:
			theta_bcs.append( DirichletBC(self.Q, theta_surface,
			                              model.ff, model.GAMMA_S_FLT) )
		if model.N_GAMMA_U_FLT > 0:
			theta_bcs.append( DirichletBC(self.Q, theta_surface,
			                              model.ff, model.GAMMA_U_FLT) )

		# apply T_melt conditions of portion of ice in contact with water :
		if model.N_GAMMA_B_FLT > 0:
			theta_bcs.append( DirichletBC(self.Q, theta_float,
			                               model.ff, model.GAMMA_B_FLT) )
		if model.N_GAMMA_L_UDR > 0:
			theta_bcs.append( DirichletBC(self.Q, theta_float,
			                               model.ff, model.GAMMA_L_UDR) )

		# apply lateral ``divide'' boundaries if desired :
		if use_lat_bc:
			s = "    - using divide-lateral boundary conditions -"
			print_text(s, cls=self)
			if model.N_GAMMA_L_DVD > 0:
				theta_bcs.append( DirichletBC(self.Q, model.theta_app,
				                              model.ff, model.GAMMA_L_DVD) )

		# update this Physics instance's list of boundary conditions :
		self.set_boundary_conditions(theta_bcs)

		# initialize the boundary conditions and thermal properties, if
		# we have not done so already :
		if not reset:
			# calculate energy and temperature melting point :
			self.calc_T_melt(annotate=False)

			T_v        = T.vector().get_local()
			W_v        = W.vector().get_local()
			T_s_v      = T_surface.vector().get_local()
			T_m_v      = T_m.vector().get_local()
			Tp_v       = T_v.copy()
			theta_s_v  = 146.3*T_s_v + 7.253/2.0*T_s_v**2
			theta_f_v  = 146.3*(T_m_v - 1.0) + 7.253/2.0*(T_m_v - 1.0)**2
			theta_i_v  = 146.3*T_v + 7.253/2.0*T_v**2 + W_v * L_f(0)

			# Surface boundary condition :
			s = "::: calculating energy boundary conditions :::"
			print_text(s, cls=self)

			# initialize the boundary conditions :
			model.init_theta_surface(theta_s_v)
			model.init_theta_app(theta_s_v)
			model.init_theta_float(theta_f_v)

			# initialize energy from W and T :
			model.init_theta(theta_i_v)

	def calc_Pe(self, avg=False, annotate=False):
		r"""
		calculates the grid P\'{e}clet number to self.model.Pe.

		if avg=True, calculate the vertical average.
		"""
		s    = "::: calculating Peclet number :::"
		print_text(s, cls=self)

		Pe          = self.get_grid_peclet_number()
		if avg:  Pe = self.model.calc_vert_average(Pe, annotatate=annotate)
		else:    Pe = project(Pe, solver_type='iterative', annotate=annotate)
		self.model.assign_variable(self.model.Pe, Pe, annotate=annotate)

	def calc_vert_avg_W(self):
		"""
		calculates the vertical averge water content W, saved to model.Wbar.
		"""
		s   = "::: calculating vertical average internal water content :::"
		print_text(s, cls=self)

		Wbar = self.model.calc_vert_average(self.model.W)
		self.model.init_Wbar(Wbar)

	def calc_vert_avg_strain_heat(self):
		"""
		calculates integrated strain-heating, saved to model.Qbar.
		"""
		s   = "::: calculating vertical average strain heat :::"
		print_text(s, cls=self)

		# calculate downward vertical integral :
		Q    = self.get_internal_friction_heat()
		Qbar = self.model.calc_vert_average(Q)
		self.model.init_Qbar(Qbar)

	def calc_temperate_thickness(self):
		"""
		calculates the temperate zone thickness, saved to model.alpha_int.
		"""
		s   = "::: calculating temperate zone thickness :::"
		print_text(s, cls=self)

		model     = self.model
		alpha_int = model.vert_integrate(model.alpha, d='down')
		alpha_int = model.vert_extrude(alpha_int, d='up')
		model.init_alpha_int(alpha_int)

	def calc_temp_rat(self):
		"""
		calculates the ratio of the temperate zone, saved to model.temp_rat.
		"""
		s   = "::: calculating ratio of column that is temperate :::"
		print_text(s, cls=self)

		model   = self.model

		self.calc_temperate_thickness()

		# TODO: the operation below breaks dolfin-adjoint annotation.
		S_v         = model.S.vector().get_local()
		B_v         = model.B.vector().get_local()
		alpha_int_v = model.alpha_int.vector().get_local()
		H_v         = S_v - B_v + DOLFIN_EPS
		temp_rat_v  = alpha_int_v / H_v
		temp_rat_v[temp_rat_v < 0.0] = 0.0
		temp_rat_v[temp_rat_v > 1.0] = 1.0
		model.init_temp_rat(alpha_int_v / H_v)

	def calc_T_melt(self, annotate=False):
		"""
		Calculates temperature melting point model.T_melt and energy melting point
		model.theta_melt.

		"""
		s    = "::: calculating pressure-melting temperature :::"
		print_text(s, cls=self)

		model = self.model

		gamma = model.gamma
		T_w   = model.T_w
		p     = model.p

		# TODO: the operation below breaks dolfin-adjoint annotation.
		p_v   = p.vector().get_local()
		Tm    = T_w(0) - gamma(0)*p_v
		tht_m = 146.3*Tm + 7.253/2.0*Tm**2

		model.assign_variable(model.T_melt,     Tm,    annotate=annotate)
		model.assign_variable(model.theta_melt, tht_m, annotate=annotate)

	def get_solve_params(self):
		"""
		Returns the solve parameters.
		"""
		return self.solve_params

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		nparams = {'newton_solver' : {'linear_solver'            : 'gmres',
		                              'preconditioner'           : 'hypre_amg',
		                              'relative_tolerance'       : 1e-13,
		                              'relaxation_parameter'     : 1.0,
		                              'maximum_iterations'       : 20,
		                              'error_on_nonconvergence'  : False}}
		params  = {'solver' : {'linear_solver'       : 'mumps',
		                       'preconditioner'      : 'none'},
		           'nparams'             : nparams,
		           'use_surface_climate' : False}
		return params

	def mark_temperate_zone(self):
		"""
		mark basal regions with overlying temperate layer to model.alpha.
		"""
		s = "::: marking basal regions with an overlying temperate layer :::"
		print_text(s, cls=self)

		# TODO: the operation below breaks dolfin-adjoint annotation.
		W_v              = self.model.W.vector().get_local()
		alpha_v          = self.model.alpha.vector().get_local()
		alpha_v[:]       = 0
		alpha_v[W_v > 0] = 1
		self.model.init_alpha(alpha_v)

	def calc_basal_temperature_flux(self, annotate=False):
		"""
		Solve for the basal temperature flux stored in model.gradT_B.
		"""
		# calculate melt-rate :
		s = "::: solving basal temperature flux k \\nabla T \\cdot n :::"
		print_text(s, cls=self)

		n_b     = self.model.n_b
		q       = self.get_temperature_flux_vector()
		q_dot_n = project(dot(q, n_b), self.model.Q, annotate=annotate)
		self.model.assign_variable(self.model.gradT_B, q_dot_n, annotate=annotate)
		print_min_max(self.model.gradT_B, 'gradT_B')

	def calc_basal_temperature_melting_flux(self, annotate=False):
		"""
		Solve for the basal temperature melting flux stored in model.gradTm_B.
		"""
		# calculate melt-rate :
		s = "::: solving basal temperature flux k \\nabla T_m \\cdot n :::"
		print_text(s, cls=self)

		n_b     = self.model.n_b
		q       = self.get_temperature_melting_flux_vector()
		q_dot_n = project(dot(q, n_b), self.model.Q, annotate=annotate)
		self.model.assign_variable(self.model.gradTm_B, q_dot_n, annotate=annotate)
		print_min_max(self.model.gradTm_B, 'gradTm_B')

	def calc_basal_melting_rate(self, annotate=False):
		"""
		Solve for the basal melt rate stored in model.Mb.
		"""
		# calculate melt-rate :
		s = "::: solving basal-melt-rate :::"
		print_text(s, cls=self)

		M_b = project(self.get_basal_melting_rate(), self.model.Q, \
		              annotate=annotate)
		self.model.assign_variable(self.model.Mb, M_b, annotate=annotate)

	def calc_q_fric(self):
		r"""
		Solve for the friction heat term stored in ``model.q_fric``.
		"""
		# calculate melt-rate :
		s = "::: solving basal friction heat :::"
		print_text(s, cls=self)

		q_fric = project(self.get_external_friction_heat(), self.model.Q, \
		                 annotate=annotate)
		self.model.assign_variable(self.model.q_fric, q_fric, annotate=annotate)

	def derive_temperate_zone(self, annotate=False):
		"""
		Solve the steady-state energy equation, saving enthalpy to model.theta,
		temperature to model.T, and water content to model.W such that the
		regions with overlying temperate ice are properly marked by model.alpha.
		"""
		model = self.model

		# solve the energy equation :
		s    = "::: solving for temperate zone locations :::"
		print_text(s, cls=self)

		# ensure that the boundary-marking process is done in steady state :
		transient = False
		if self.transient:
			self.make_steady_state()
			transient = True

		# put the physics in temperate zone marking mode :
		if self.energy_flux_mode != 'temperate_zone_mark':
			zef  = True
			mode = self.energy_flux_mode
			self.set_basal_flux_mode('temperate_zone_mark')

		# solve the linear system :
		solve(self.get_lhs() == self.get_rhs(), self.get_unknown(),
		      self.get_boundary_conditions(),
		      solver_parameters = self.solve_params['solver'], annotate=annotate)

		# calculate water content :
		# TODO: the operation below breaks dolfin-adjoint annotation.
		theta_v         = self.get_unknown().vector().get_local()
		theta_melt_v    = model.theta_melt.vector().get_local()
		W_v             = (theta_v - theta_melt_v) / model.L_f(0)
		W_v[W_v < 0.0]  = 0.0    # no water where frozen, please.

		# mark appropriately basal regions with an overlying temperate layer :
		# TODO: the operation below breaks dolfin-adjoint annotation.
		alpha_v          = model.alpha.vector().get_local()
		alpha_v[:]       = 0
		alpha_v[W_v > 0] = 1
		model.init_alpha(alpha_v)

		# reset to previous energy flux mode, if necessary :
		if zef:
			self.set_basal_flux_mode(mode)

		# convert back to transient if necessary :
		if transient:
			energy.make_transient(time_step = model.time_step)

	def update_thermal_parameters(self, annotate=False):
		"""
		fixed-point iterations to make all linearized thermal parameters consistent.
		"""
		# TODO: the operation below breaks dolfin-adjoint annotation.
		model = self.model

		# solve the energy equation :
		s    = "::: updating thermal parameters :::"
		print_text(s, cls=self)

		# ensure that we have steady state :
		transient = False
		if self.transient:
			self.make_steady_state()
			transient = True

		# previous theta for norm calculation
		U_prev  = self.get_unknown().copy(True)

		# iteration counter :
		counter = 1

		# maximum number of iterations :
		max_iter = 1000

		# L_2 erro norm between iterations :
		abs_error = np.inf
		rel_error = np.inf

		# tolerances for stopping criteria :
		atol = 1e-7
		rtol = 1e-8

		# perform a fixed-point iteration until the L_2 norm of error
		# is less than tolerance :
		while abs_error > atol and rel_error > rtol and counter <= max_iter:

			# solve the linear system :
			solve(self.get_lhs() == self.get_rhs(), self.get_unknown(),
			      self.get_boundary_conditions(),
			      solver_parameters = self.solve_params['solver'], annotate=annotate)

			# calculate L_2 norms :
			abs_error_n  = norm(U_prev.vector() - self.get_unknown().vector(), 'l2')
			tht_nrm      = norm(self.get_unknown().vector(), 'l2')

			# save convergence history :
			if counter == 1:
				rel_error  = abs_error_n
			else:
				rel_error = abs(abs_error - abs_error_n)

			# print info to screen :
			if model.MPI_rank == 0:
				s0    = '>>> '
				s1    = 'thermal parameter update iteration %i (max %i) done: ' \
				        % (counter, max_iter)
				s2    = 'r (abs) = %.2e ' % abs_error
				s3    = '(tol %.2e), '    % atol
				s4    = 'r (rel) = %.2e ' % rel_error
				s5    = '(tol %.2e)'      % rtol
				s6    = ' <<<'
				text0 = get_text(s0, 'red', 1)
				text1 = get_text(s1, 'red')
				text2 = get_text(s2, 'red', 1)
				text3 = get_text(s3, 'red')
				text4 = get_text(s4, 'red', 1)
				text5 = get_text(s5, 'red')
				text6 = get_text(s6, 'red', 1)
				print(text0 + text1 + text2 + text3 + text4 + text5 + text6)

			# update error stuff and increment iteration counter :
			abs_error    = abs_error_n
			U_prev       = self.get_unknown().copy(True)
			counter     += 1

			# update the model variable :
			self.update_model_var(self.get_unknown(), annotate=annotate)

			# update the temperature and water content for other physics :
			self.partition_energy(annotate=annotate)

		# convert back to transient if necessary :
		if transient:
			energy.make_transient(time_step = model.time_step)

	def solve(self, annotate=False):
		"""
		Solve the energy equations, saving energy to ``model.theta``, temperature
		to ``model.T``, and water content to ``model.W``.
		"""
		model = self.model

		# update the surface climate if desired :
		if self.solve_params['use_surface_climate']:  self.solve_surface_climate()

		# solve as defined in ``physics.Physics.solve()`` :
		super(Energy, self).solve(annotate)

		# update the temperature and water content for other physics :
		self.partition_energy(annotate=False)

	def update_model_var(self, u, annotate=False):
		"""
		Update the energy function ``self.model.theta`` to those given by ``u``.
		"""
		self.ass_theta.assign(self.model.theta, u, annotate=annotate)
		print_min_max(self.model.theta, 'theta')






class EnergyHybrid(Energy):
	"""
	New 2D hybrid model.

	Original author: Doug Brinkerhoff: https://dbrinkerhoff.org/
	"""
	# TODO: `energy_flux_mode` and `stabilization_method` makes no sense here.
	def initialize(self, model, momentum,
	               solve_params         = None,
	               transient            = False,
	               use_lat_bc           = False,
	               energy_flux_mode     = 'B_ring',
	               stabilization_method = 'GLS'):
		"""
		Set up energy equation residual.
		"""
		s    = "::: INITIALIZING HYBRID ENERGY PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != D2Model:
			s = ">>> EnergyHybrid REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		# save the solver parameters :
		self.solve_params = solve_params

		self.transient = transient

		# CONSTANTS
		year    = model.spy
		g       = model.g
		n       = model.n

		k       = model.k_i
		rho     = model.rho_i
		Cp      = model.c_i
		kappa   = year*k/(rho*Cp)

		q_geo   = model.q_geo
		S       = model.S
		B       = model.B
		beta    = model.beta
		T_s     = model.T_surface
		T_w     = model.T_w
		H       = model.H
		H0      = model.H0
		T_      = model.T_
		T0_     = model.T0_
		deltax  = model.deltax
		sigmas  = model.sigmas
		eps_reg = model.eps_reg
		h       = model.h
		dt      = model.time_step
		N_T     = model.N_T

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

		U    = momentum.U
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
			w_eff = u(s)*dsdx(s) + v(s)*dsdy(s) + w(s)*dsdz(s)

			if transient:
				w_eff += 1.0/H*(1.0 - s)*(H - H0)/dt

			# STRAIN HEAT
			#Phi_strain = (2*n)/(n+1)*2*eta_v(s)*epsilon_dot(s)
			Phi_strain = 4*eta_v(s)*epsilon_dot(s)

			# STABILIZATION SCHEME
			#Umag   = sqrt(u(s)**2 + v(s)**2 + 1e-3)
			#tau    = h/(2*Umag)
			#Psihat = Psi[i] + tau*(u(s)*Psi[i].dx(0) + v(s)*Psi[i].dx(1))
			Unorm  = sqrt(u(s)**2 + v(s)**2 + DOLFIN_EPS)
			Pe     = Unorm*h/(2*kappa)
			tau    = 1/tanh(Pe) - 1/Pe
			Psihat = Psi[i] + h*tau/(2*Unorm) * (+ u(s)*Psi[i].dx(0) \
			                                     + v(s)*Psi[i].dx(1) )

			# SURFACE BOUNDARY
			if i==0:
				R_T += Psi[i]*(T(i) - T_s)*dx
			# BASAL BOUNDARY
			elif i==(N_T-1):
				R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
				R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx
				R_T += -w_eff*q_geo/(rho*Cp*kappa*dsdz(s))*Psi[i]*dx
				f    = (q_geo + beta*(u(s)**2 + v(s)**2))/(rho*Cp*kappa*dsdz(s))
				R_T += -2.*kappa*dsdz(s)**2*(+ (T(N_T-2) - T(N_T-1)) / deltax**2 \
				                             - f/deltax)*Psi[i]*dx
			# INTERIOR
			else:
				R_T += -kappa*dsdz(s)**2.*T.d2s(i)*Psi[i]*dx
				R_T += w_eff*T.ds(i)*Psi[i]*dx
				R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
				R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx

			if transient:
				dTdt = (T(i) - T0(i))/dt
				R_T += dTdt*Psi[i]*dx

		# PRETEND THIS IS LINEAR (A GOOD APPROXIMATION IN THE TRANSIENT CASE)
		self.R_T = replace(R_T, {T_:dT})

		# pressure melting point calculation, do not annotate for initial calc :
		self.Tm  = as_vector([T_w - sigma*gamma*rho*g*H for sigma in sigmas])
		self.calc_T_melt(annotate=False)

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
		ffc_options = {"optimize"               : True}
		return ffc_options

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		m_params  = {'solver'      : {'linear_solver': 'mumps'},
		             'ffc_params'  : self.default_ffc_options()}
		return m_params

	def solve(self, annotate=False):
		"""
		Solves for hybrid energy.
		"""
		s    = "::: solving 'EnergyHybrid' for temperature :::"
		print_text(s, cls=self)

		model  = self.model

		# SOLVE TEMPERATURE
		solve(lhs(self.R_T) == rhs(self.R_T), model.T_,
		      solver_parameters=self.solve_params['solver'],
		      form_compiler_parameters=self.solve_params['ffc_params'],
		      annotate=annotate)
		print_min_max(model.T_, 'T_')

		if self.transient:
			model.T0_.assign(model.T_)

		#  correct for pressure melting point :
		T_v                 = model.T_.vector().get_local()
		T_melt_v            = model.Tm.vector().get_local()
		T_v[T_v > T_melt_v] = T_melt_v[T_v > T_melt_v]
		model.assign_variable(model.T_, T_v)

		out_T = model.T_.split(True)            # deepcopy avoids projections

		model.assign_variable(model.Ts, out_T[0])
		model.assign_variable(model.Tb, out_T[-1])

		# update the melting temperature too :
		self.calc_T_melt(annotate=annotate)

	def calc_T_melt(self, annotate=False):
		"""
		Calculates pressure-melting point in model.T_melt.
		"""
		s    = "::: calculating pressure-melting temperature :::"
		print_text(s, cls=self)

		model   = self.model

		T_melt  = project(self.Tm, solver_type='iterative', annotate=annotate)

		Tb_m    = T_melt.split(True)[-1]  # deepcopy avoids projections
		model.assign_variable(model.T_melt, Tb_m)
		model.assign_variable(model.Tm,     T_melt)






class EnergyFirn(Energy):
	"""
	"""
	# TODO: energy flux mode makes no sense here.
	def initialize(self, model, momentum,
	               solve_params     = None,
	               transient        = False,
	               use_lat_bc       = False,
	               energy_flux_mode = 'B_ring',
	               reset            = False):
		"""
		"""
		s    = "::: INITIALIZING FIRN ENERGY PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != D1Model:
			s = ">>> FirnEnergy REQUIRES A 'D1Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		# save the solver parameters :
		self.solve_params = solve_params

		mesh    = model.mesh
		Q       = model.Q

		spy     = model.spy
		theta   = model.theta                     # enthalpy
		theta0  = model.theta0                    # previous enthalpy
		T       = model.T                         # temperature
		rhof    = model.rho                       # density of firn
		sigma   = model.sigma                     # overburden stress
		r       = model.r                         # grain size
		w       = model.w                         # velocity
		m       = model.m                         # mesh velocity
		dt      = model.time_step                 # timestep
		rho_i   = model.rho_i                      # density of ice
		rho_w   = model.rho_w                      # density of water
		c_i     = model.c_i                        # heat capacity of ice
		c_w     = model.c_w                        # heat capacity of water
		k_w     = model.k_w                        # thermal conductivity of water
		T       = model.T
		T_w     = model.T_w
		L_f     = model.L_f
		thetasp = model.thetasp
		p       = model.p
		etaw    = model.etaw
		rho_w   = model.rho_w
		#w       = w - m
		z       = model.x[0]
		g       = model.g
		S       = model.S
		h       = model.h
		#W       = model.W
		dx      = model.dx

		xi      = TestFunction(Q)
		dtheta  = TrialFunction(Q)

		# thermal conductivity parameter :
		#k_i  = model.k_i*(rho / rho_i)**2
		k_i  = 9.828 * exp(-0.0057*T)

		# water content :
		Wm    = conditional(lt(theta, c_i*T_w), 0.0, (theta - c_i*T_w)/L_f)

		# bulk properties :
		kb   = k_w * Wm   + (1-Wm)*k_i
		cb   = c_w * Wm   + (1-Wm)*c_i
		rhob = rho_w * Wm + (1-Wm)*rhof

		# initialize energy :
		T_v = T.vector().get_local()
		model.assign_variable(theta,  c_i(0)*T_v)
		model.assign_variable(theta0, c_i(0)*T_v)

		# boundary condition on the surface :
		self.thetaBc = DirichletBC(Q, model.theta_surface,  model.surface)

		# Darcy flux :
		k     = 0.077 * r**2 * exp(-7.8*rhob/rho_w) # intrinsic permeability
		phi   = 1 - rhob/rho_i                      # porosity
		Wmi   = 0.0057 / (1 - phi) + 0.017         # irriducible water content
		Se    = (Wm - Wmi) / (1 - Wmi)             # effective saturation
		K     = k * rho_w * g / etaw                # saturated hydraulic cond.
		krw   = Se**3.0                            # relative permeability
		psi_m = p / (rho_w * g)                     # matric potential head
		psi_g = z                                  # gravitational potential head
		psi   = psi_m + psi_g                      # total water potential head
		u     = - K * krw * psi.dx(0)              # darcy water velocity

		# skewed test function in areas with high velocity :
		Pe     = (u+w)*h/(2*kb/(rhob*cb))
		tau    = 1/tanh(Pe) - 1/Pe
		xihat  = xi + h*tau/2 * xi.dx(0)

		# enthalpy residual :
		eta       = 1.0
		theta_mid = eta*theta + (1 - eta)*theta0
		delta     = + kb/(rhob*cb) * inner(theta_mid.dx(0), xi.dx(0)) * dx \
		            + (theta - theta0)/dt * xi * dx \
		            + (w + u) * theta_mid.dx(0) * xihat * dx \
		            - sigma * w.dx(0) / rhob * xi * dx

		# equation to be minimzed :
		self.J     = derivative(delta, theta, dtheta)   # jacobian

		self.delta = delta
		self.u     = u
		self.Wm    = Wm
		self.Wmi   = Wmi

	def get_solve_params(self):
		"""
		Returns the solve parameters.
		"""
		return self.solve_params

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		params = {'newton_solver' : {'relaxation_parameter'     : 1.0,
		                             'maximum_iterations'      : 25,
		                             'error_on_nonconvergence' : False,
		                             'relative_tolerance'      : 1e-10,
		                             'absolute_tolerance'      : 1e-10}}
		m_params  = {'solver' : params}
		return m_params

	def solve(self, annotate=False):
		"""
		"""
		s    = "::: solving FirnEnergy :::"
		print_text(s, cls=self)

		model = self.model

		# newton's iterative method :
		solve(self.delta == 0, model.theta, self.thetaBc, J=self.J,
		      solver_parameters=self.solve_params['solver'],
		      annotate=annotate)

		model.assign_variable(model.W0,  model.W)
		model.assign_variable(model.W,   project(self.Wm, annotate=False))

		T_w     = model.T_w(0)
		rho_w   = model.rho_w(0)
		rho_i   = model.rho_i(0)
		g       = model.g(0)
		c_i     = model.c_i(0)
		thetasp = c_i * T_w
		L_f     = model.L_f(0)

		# update coefficients used by enthalpy :
		thetap     = model.theta.vector().get_local()
		thetahigh  = np.where(thetap > thetasp)[0]
		thetalow   = np.where(thetap < thetasp)[0]

		# calculate T :
		Tp             = thetap / c_i
		Tp[thetahigh]  = T_w
		model.assign_variable(model.T, Tp)

		# calculate dW :
		Wp   = model.W.vector().get_local()
		Wp0  = model.W0.vector().get_local()
		dW   = Wp - Wp0                 # water content change
		model.assign_variable(model.dW, dW)

		# adjust the snow density if water is refrozen :
		rho_v         = model.rho.vector().get_local()
		freeze        = dW < 0
		melt          = dW > 0
		rho_v[freeze] = rho_v[freeze] - dW[freeze] * model.rho_i(0)
		model.assign_variable(model.rho, rho_v)

		## calculate W :
		#model.assign_variable(model.W0, model.W)
		#Wp             = model.W.vector().get_local()
		#Wp[thetahigh]  = (thetap[thetahigh] - c_i*T_w) / L_f
		#Wp[thetalow]   = 0.0
		#Wp0            = model.W0.vector().get_local()
		#dW             = Wp - Wp0                 # water content change
		#model.assign_variable(model.W,  Wp)
		#model.assign_variable(model.dW, dW)

		print_min_max(model.T,     'T')
		print_min_max(model.theta, 'theta')
		print_min_max(model.W,     'W')

		p     = model.vert_integrate(rho_w * g * model.W)
		phi   = 1 - model.rho/rho_i                         # porosity
		Wmi   = 0.0057 / (1 - phi) + 0.017                 # irr. water content
		model.assign_variable(model.p,   p)
		model.assign_variable(model.u,   project(self.u, annotate=False))
		model.assign_variable(model.Smi, project(Wmi, annotate=False))
		print_min_max(model.p, 'p')
		print_min_max(model.u, 'u')



