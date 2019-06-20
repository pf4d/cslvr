from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
from dolfin            import *
from dolfin_adjoint    import *
from cslvr.helper      import raiseNotDefined
from cslvr.inputoutput import get_text, print_text, print_min_max
from time              import time
import numpy               as np
import matplotlib.pyplot   as plt
import os


class Physics(object):
	"""
	This abstract class outlines the structure of a physics calculation.

	:param model: the model instance for this physics problem
	:type model:  :class:`~model.Model`
	"""

	def __new__(self, model, *args, **kwargs):
		"""
		Creates and returns a new Physics object.
		"""
		instance = object.__new__(self)
		instance.model = model
		return instance

	def color(self):
		"""
		return the default color for this class.

		:rtype: string
		"""
		return 'white'

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance

		:rtype: dict
		"""
		params  = {'solver' : 'mumps'}
		return params

	def get_solve_params(self):
		"""
		Returns the solve parameters.
		"""
		return self.solve_params

	def get_residual(self):
		"""
		Returns the residual of this instance ``self.resid``.
		"""
		return self.resid

	def set_residual(self, resid):
		"""
		Replaces the residual ``self.resid`` with parameter ``resid``.
		If the physics are non-linear as set my ``self.linear``, this will also
		re-calculate the Jacobian for use with the Newton solver.
		"""
		self.resid = resid

		# if linear, separate the left- and right-hand sides :
		if self.linear:
			self.lhs = lhs(resid)
			self.rhs = rhs(resid)

		# re-define the momentum Jacobian :
		else:
			self.Jac = derivative(self.get_residual(),
			                      self.get_unknown(),
			                      self.get_trial_function())

	def get_jacobian(self):
		"""
		Returns the Jacobian of this problem set by called
		:func:`~physics.Physics.set_residual`.
		"""
		err_msg = "this problem is linear; there is no Jacobian."
		assert not self.linear, err_msg
		return self.Jac

	def set_boundary_conditions(self, bc_list):
		"""
		Set the Dirichlet boundary conditions of this problem to those of the
		list ``bc_list``.
		"""
		self.bcs = bc_list

	def get_boundary_conditions(self):
		"""
		Returns the list of Dirichlet boundary conditions for this problem.
		"""
		return self.bcs

	def get_lhs(self):
		"""
		Returns the left-hand side of this problem.
		"""
		err_msg = "this problem is non-linear; there is no left-hand side."
		assert self.linear, err_msg
		return self.lhs

	def get_rhs(self):
		"""
		Returns the right-hand side of this problem.
		"""
		err_msg = "this problem is non-linear; there is no right-hand side."
		assert self.linear, err_msg
		return self.rhs

	def get_unknown(self):
		"""
		Return the unkonwn :dolfin:`~function.Function` set by ``self.u``.
		"""
		return self.u

	def set_unknown(self, u):
		"""
		Sets the unkonwn :dolfin:`~function.Function` ``self.u`` to
		parameter ``u``.
		"""
		self.u = u

	def get_trial_function(self):
		"""
		Return the :dolfin:`~function.TrialFunction` for the unknown
		``self.u`` set by ``self.trial_function``.
		"""
		return self.trial_function

	def set_trial_function(self, trial_function):
		"""
		Sets the :dolfin:`~function.TrialFunction` for the unknown
		``self.u`` given by ``self.trial_function`` to parameter ``trial_function``.
		"""
		self.trial_function = trial_function

	def get_test_function(self):
		"""
		Return the :dolfin:`~function.TestFunction` for the unknown
		``self.u`` set by ``self.test_function``.
		"""
		return self.test_function

	def set_test_function(self, test_function):
		"""
		Sets the :dolfin:`~function.TestFunction` for the unknown
		``self.u`` given by ``self.test_function`` to parameter ``test_function``.
		"""
		self.test_function = test_function

	def get_Lam(self):
		"""
		Return the adjoint function for ``self.U``.
		"""
		raiseNotDefined()

	def get_velocity(self):
		r"""
		Return the velocity :math:`\underline{u}`.
		"""
		raiseNotDefined()

	def form_reg_ftn(self, c, integral, kind='TV', verbose=True):
		r"""
		Formulates a regularization functional for use
		with optimization of the control parameter :math:`c` given by ``c``
		over the integral ``integral``.

		The choices for ``kind`` are :

		1. ``tikhonov`` -- Tikhonov regularization

		.. math::

		  \mathscr{R}(c) = \frac{1}{2} \int_{\Gamma} \nabla c \cdot \nabla c\ \mathrm{d}\Gamma

		2. ``tv`` -- total variation regularization

		.. math::

		  \mathscr{R}(c) = \int_{\Gamma} \left( \nabla c \cdot \nabla c + c_0 \right)^{\frac{1}{2}}\ \mathrm{d}\Gamma,

		3. ``square`` -- squared regularization

		.. math::

		  \mathscr{R}(c) = \frac{1}{2} \int_{\Gamma} c^2\ \mathrm{d}\Gamma,

		4. ``abs`` -- absolute regularization

		.. math::

		  \mathscr{R}(c) = \int_{\Gamma} |c|\ \mathrm{d}\Gamma,

		:param c:         the control variable
		:param integral:  measure over which to integrate
		                  (see :func:`~model.calculate_boundaries`)
		:param kind:      kind of regularization to use
		:type c:          :class:`dolfin.Function`
		:type integral:   :class:`~helper.Boundary`
		:type kind:       string
		"""
		if verbose:
			s  = "::: forming '%s' regularization functional for variable" + \
			     " '%s' integrated over the %s :::"
			print_text(s % (kind, c.name(), integral.description), cls=self)

		dR = integral()

		# the various possible regularization types :
		kinds = ['tv', 'tikhonov', 'square', 'abs']

		# form regularization term 'R' :
		if kind not in kinds:
			s    =   ">>> VALID REGULARIZATIONS ARE 'tv', 'tikhonov', 'square'," + \
			         " or 'abs' <<<"
			print_text(s, 'red', 1)
			sys.exit(1)

		elif kind == 'tv':
			R = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR

		elif kind == 'tikhonov':
			R = 0.5 * inner(grad(c), grad(c)) * dR

		elif kind == 'square':
			R = 0.5 * c**2 * dR

		elif kind == 'abs':
			R = abs(c) * dR

		return R

	def form_cost_ftn(self, u, u_ob, integral, kind='log', verbose=True):
		r"""
		Forms and returns a cost functional for use with adjoint calculations such
		as :func:`~physics.Physics.optimize`.

		The choices for ``kind`` are :

		1. ``log`` -- logarithmic cost

		.. math::

		  \mathscr{J}(u) = \frac{1}{2} \int_{\Gamma} \ln \left( \frac{ \Vert u \Vert_2 }{ \Vert u_{\mathrm{ob}} \Vert_2 + \epsilon } \right)^2 \mathrm{d}\Gamma

		2. ``l2`` -- :math:`L_2`-norm cost

		.. math::

		  \mathscr{J}(u) = \frac{1}{2} \int_{\Gamma} \Vert u - u_{\mathrm{ob}} \Vert_2^2 \mathrm{d}\Gamma

		3. ``ratio`` -- ratio cost

		.. math::

		  \mathscr{J}(u) = \frac{1}{2} \int_{\Gamma} \left( 1 - \left( \frac{ \Vert u \Vert_2 }{ \Vert u_{\mathrm{ob}} \Vert_2 + \epsilon } \right) \right)^2 \mathrm{d}\Gamma

		4. ``l1`` -- absolute value :math:`L_1` norm cost

		.. math::

		  \mathscr{J}(u) = \int_{\Gamma} \left| u - u_{\mathrm{ob}} \right| \mathrm{d}\Gamma

		The parameter :math:`\epsilon` is on the order of machine precision.

		:param u:        the state variable
		:param u_ob:     observation of the state variable ``u``
		:param integral: measure over which to integrate
		                 (see :func:`~model.calculate_boundaries`)
		:param kind:     kind of cost function to use
		:param verbose:  inform what the function is doing.
		:type u:         :class:`dolfin_adjoint.Function`
		:type u_ob:      :class:`dolfin.Function`
		:type integral:  :class:`~helper.Boundary`
		:type kind:      string
		:type verbose:   boolean
		"""
		if verbose:
			s  = "::: forming '%s' objective functional integrated over %s :::"
			print_text(s % (kind, integral.description), cls=self)

		dJ = integral()

		if type(u) is not list and type(u_ob) is not list:
			u    = [u]
			u_ob = [u_ob]
		# for a UFL expression for the norm and difference :
		U     = 0
		U_ob  = 0
		U_err = 0
		U_abs = 0
		for Ui, U_obi in zip(u, u_ob):
			U     += Ui**2
			U_ob  += U_obi**2
			U_err += (Ui - U_obi)**2
			U_abs += abs(Ui - U_obi)
		U    = sqrt(U)
		U_ob = sqrt(U_ob)

		if kind == 'log':
			J  = 0.5 * ln( U / (U_ob + DOLFIN_EPS) )**2 * dJ

		elif kind == 'l2':
			J  = 0.5 * U_err * dJ

		elif kind == 'ratio':
			J  = 0.5 * (1 -  U / (U_ob + DOLFIN_EPS))**2 * dJ

		elif kind == 'l1':
			J  = U_abs * dJ

		else:
			s = ">>> ADJOINT OBJECTIVE FUNCTIONAL MAY BE 'l2', " + \
			    "'log', 'ratio', OR 'l1', NOT '%s' <<<" % kind
			print_text(s, 'red', 1)
			sys.exit(1)
		return J

	def calc_functionals(self, u, u_ob, control, J_measure, R_measure):
		"""
		Used to facilitate printing the objective function in adjoint solves.
		"""
		s   = "::: calculating functionals :::"
		print_text(s, cls=self)

		color = '208'

		# ensure we can iterate :
		if type(control) != list:    control = [control]
		if type(R_measure) != list:  R_measure = [R_measure]

		# form a UFL expression for the norm and difference :
		J_log = self.form_cost_ftn(u, u_ob, J_measure, kind='log',   verbose=False)
		J_l1  = self.form_cost_ftn(u, u_ob, J_measure, kind='l1',    verbose=False)
		J_l2  = self.form_cost_ftn(u, u_ob, J_measure, kind='l2',    verbose=False)
		J_rat = self.form_cost_ftn(u, u_ob, J_measure, kind='ratio', verbose=False)

		# assemble the tensors :
		J_log_a = assemble(J_log, annotate=False)
		J_l1_a  = assemble(J_l1,  annotate=False)
		J_l2_a  = assemble(J_l2,  annotate=False)
		J_rat_a = assemble(J_rat, annotate=False)

		# print their values :
		print_min_max(J_log_a, 'J_log : %s \t' % u_ob[0].name(), color=color)
		print_min_max(J_l1_a,  'J_l1  : %s \t' % u_ob[0].name(), color=color)
		print_min_max(J_l2_a,  'J_l2  : %s \t' % u_ob[0].name(), color=color)
		print_min_max(J_rat_a, 'J_rat : %s \t' % u_ob[0].name(), color=color)

		# form a list of the cost functionals to return :
		J_a = [J_log_a, J_l2_a,  J_rat_a, J_l1_a]

		# iterate over each of the regularization functionals for each control :
		R_a = []
		for c, R_m in zip(control, R_measure):

			# get the UFL form :
			R_tv    = self.form_reg_ftn(c, R_m, kind='tv',       verbose=False)
			R_tik   = self.form_reg_ftn(c, R_m, kind='tikhonov', verbose=False)
			R_sq    = self.form_reg_ftn(c, R_m, kind='square',   verbose=False)
			R_abs   = self.form_reg_ftn(c, R_m, kind='abs',      verbose=False)

			# assemble the tensors :
			R_tv_a  = assemble(R_tv,  annotate=False)
			R_tik_a = assemble(R_tik, annotate=False)
			R_sq_a  = assemble(R_sq,  annotate=False)
			R_abs_a = assemble(R_abs, annotate=False)

			# print their values :
			print_min_max(R_tv_a,  'R_tv  : %s \t' % c.name(), color=color)
			print_min_max(R_tik_a, 'R_tik : %s \t' % c.name(), color=color)
			print_min_max(R_sq_a,  'R_sq  : %s \t' % c.name(), color=color)
			print_min_max(R_abs_a, 'R_abs : %s \t' % c.name(), color=color)

			# append the values to the list to return :
			R_a.append([R_tv_a, R_tik_a, R_sq_a, R_abs_a])

		return [J_a, R_a]

	def plot_ftnls(self, t0, tf, J, R, control):
		"""
		Save all the objective functional values and produce a rudimentary plot.
		"""
		model = self.model
		d     = model.out_dir + 'objective_ftnls_history/'

		s    = '::: saving objective functionals to %s :::'
		print_text(s % d, cls=self)

		# for images :
		J_lab = r'$\mathscr{J}_{\mathrm{%s}}$'
		R_lab = r'$\mathscr{R}_{\mathrm{%s}}\left( %s \right)$'

		# for subscripts of file names :
		J_sub = ['log', 'l2',  'rat', 'l1']
		R_sub = ['tv',  'tik', 'sq',  'abs']

		if model.MPI_rank==0:
			if not os.path.exists(d):
				os.makedirs(d)
			for i,sub in enumerate(J_sub):
				np.savetxt(d + 'J_%s.txt' % sub, J[:,i])
				fig = plt.figure()
				ax  = fig.add_subplot(111)
				# if the difference is greater than one order of magnitude :
				if max(J[0,i], J[-1,i]) / min(J[0,i], J[-1,i]) > 10:
					ax.set_yscale('log')
				ax.set_ylabel(J_lab % sub)
				ax.set_xlabel(r'iteration')
				ax.plot(J[:,i], 'r-', lw=2.0)
				plt.grid()
				plt.savefig(d + 'J_%s.png' % sub, dpi=100)
				plt.close(fig)
			for j, c in enumerate(control):
				for i, sub in enumerate(R_sub):
					np.savetxt(d + 'R_%s_%s.txt' % (sub, c.name()), R[:,j][:,i])
					fig = plt.figure()
					ax  = fig.add_subplot(111)
					# if the difference is greater than one order of magnitude :
					if max(J[0,i], J[-1,i]) / min(J[0,i], J[-1,i]) > 10:
						ax.set_yscale('log')
					ax.set_ylabel(R_lab % (sub, c.name()))
					ax.set_xlabel(r'iteration')
					ax.plot(R[:,j][:,i], 'r-', lw=2.0)
					plt.grid()
					plt.savefig(d + 'R_%s_%s.png' % (sub, c.name()), dpi=100)
					plt.close(fig)

			# save the total time to compute :
			np.savetxt(d + 'time.txt',  np.array([tf - t0]))

	def optimize(self, u, u_ob, I, control, J_measure, R_measure, bounds,
	             method            = 'ipopt',
	             max_iter          = 100,
	             adj_save_vars     = None,
	             adj_callback      = None,
	             post_adj_callback = None):
		"""
		"""
		s    = "::: solving optimal control to minimize ||u - u_ob|| with " + \
		       "control parameter %s :::"
		if type(control) != list:
			control = [control]
		tx = 's '
		for i in control:
			tx += "'" + i.name() + "'"
			if i != control[-1]: tx += ' and '
		print_text(s % tx, cls=self)

		model = self.model

		# reset entire dolfin-adjoint state :
		adj_reset()

		# starting time :
		t0   = time()

		# need this for the derivative callback :
		global counter
		counter = 0

		# functional lists to be populated :
		global J_a, R_a
		J_a, R_a = [],[]

		# container for the current optimal value for evaluation :
		global u_opt
		u_opt = DolfinAdjointVariable(u)

		# solve the physics with annotation enabled :
		s    = '::: solving forward problem :::'
		print_text(s, cls=self)
		self.solve(annotate=True)

		# now solve the control optimization problem :
		s    = "::: starting adjoint-control optimization with method '%s' :::"
		print_text(s % method, cls=self)

		# objective function callback function :
		def eval_cb_post(I, c):
			s    = '::: adjoint objective post-eval callback function :::'
			print_text(s, cls=self)
			print_min_max(I,    'I', cls=self)
			for ci in c:
				print_min_max(ci,    'control: ' + ci.name(), cls=self)

		# objective gradient callback function :
		def derivative_cb_post(I, dI, c):
			global counter, J_a, R_a, u_opt

			# update model velocity to that of the current velocity^* ;
			u_opt = DolfinAdjointVariable(u).tape_value()
			self.update_model_var(u_opt, annotate=False)

			# make the control available too :
			for i in range(len(control)):
				control[i].assign(c[i], annotate=False)

			# this method is called by IPOPT only once per nonlinear iteration.
			# SciPy will call this many times, and we only want to save the
			# functionals once per iteration.	See ``scipy_callback()`` below :
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

				# print functional values :
				ftnls = self.calc_functionals(u_opt,u_ob,control,J_measure,R_measure)

				# add to the list of functional evaluations :
				J_a.append(ftnls[0])
				R_a.append(ftnls[1])

			s    = '::: adjoint objective gradient post-eval callback function :::'
			print_text(s, cls=self)
			for (dIi,ci) in zip(dI,c):
				print_min_max(dIi,    'dI/control: ' + ci.name(), cls=self)
				#self.model.save_xdmf(dIi, 'dI_control_' + ci.name())
				#self.model.save_xdmf(ci, 'control_' + ci.name())

			# call that callback, if you want :
			if adj_callback is not None:
				adj_callback(I, dI, c)

		# define the control parameter :
		m = []
		for i in control:
			m.append(Control(i, value=i))

		# create the reduced functional to minimize :
		F = ReducedFunctional(Functional(I), m,
		                      eval_cb_post        = eval_cb_post,
		                      derivative_cb_post  = derivative_cb_post)

		# optimize with scipy's fmin_l_bfgs_b :
		if method == 'l_bfgs_b':

			# this callback is called only once per iteration :
			def scipy_callback(c):
				global counter, J_a, R_a, u_opt
				s0    = '>>> '
				s1    = 'iteration %i (max %i) complete'
				s2    = ' <<<'
				text0 = get_text(s0, 'red', 1)
				text1 = get_text(s1 % (counter, max_iter), 'red')
				text2 = get_text(s2, 'red', 1)
				if MPI.rank(mpi_comm_world())==0:
					print(text0 + text1 + text2)
				counter += 1

				# NOTE: the variables ``u_opt`` and ``control`` were updated by the
				#       ``derivative_cb_post`` method.
				# print functional values :
				ftnls = self.calc_functionals(u_opt,u_ob,control,J_measure,R_measure)

				# add to the list of functional evaluations :
				J_a.append(ftnls[0])
				R_a.append(ftnls[1])

			# TODO: provide access to the solver parameters and set up default values.
			out = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=bounds,
			               callback = scipy_callback,
			               options  ={"disp"    : True,
			                          "maxiter" : max_iter,
			                          "gtol"    : 1e-5})
			b_opt = out

			# convert to something that we can zip (see below) !
			if type(b_opt) is not list: b_opt = [b_opt]

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
			              #"ma97_order"         : "metis",
			              "linear_solver"      : "ma97"}
			solver = IPOPTSolver(problem, parameters=parameters)
			b_opt  = solver.solve()


		# make the optimal control parameter available :
		for c,b in zip(control, b_opt):
			model.assign_variable(c, b)

		# call the post-adjoint callback function if set :
		if post_adj_callback is not None:
			s    = '::: calling optimize_ubar() post-adjoined callback function :::'
			print_text(s, cls=self)
			post_adj_callback()

		# save state to unique hdf5 file :
		if isinstance(adj_save_vars, list):
			s    = '::: saving variables in list arg adj_save_vars :::'
			print_text(s, cls=self)
			out_file = model.out_dir + 'u_opt.h5'
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
		text = "time to optimize ||u - u_ob||: %02d:%02d:%02d" % (h,m,s)
		print_text(text, 'red', 1)

		# convert lists to arrays :
		J_a = np.array(J_a)
		R_a = np.array(R_a)

		# save a rudimentary plot of functionals
		self.plot_ftnls(t0, tf, J_a, R_a, control)

	def solve(self, annotate=False):
		"""
		Solves the physics calculation.
		"""
		model  = self.model
		params = self.solve_params

    # starting time (for total time to compute)
		t0 = time()

		# solve linear system :
		if self.linear:
			print_text("::: solving linear equations :::", cls=self)
			solve(self.get_lhs() == self.get_rhs(), \
			      self.get_unknown(), \
			      bcs               = self.get_boundary_conditions(), \
			      annotate          = annotate, \
			      solver_parameters = params['solver'])

		# or solve non-linear system :
		else:
			rtol   = params['solver']['newton_solver']['relative_tolerance']
			maxit  = params['solver']['newton_solver']['maximum_iterations']
			alpha  = params['solver']['newton_solver']['relaxation_parameter']
			s      = "::: solving nonlinear system with %i max iterations " + \
			         "and step size = %.1f :::"
			print_text(s % (maxit, alpha), cls=self)

			# compute solution :
			solve(self.get_residual() == 0, \
			      self.get_unknown(), \
			      J                 = self.get_jacobian(), \
			      bcs               = self.get_boundary_conditions(), \
			      annotate          = annotate, \
			      solver_parameters = params['solver'])

		# calculate total time to compute
		t_tot = time() - t0
		m     = t_tot / 60.0
		h     = m / 60.0
		m     = m % 60
		s     = t_tot % 60
		ms    = (s % 1) * 1000

		print_text("time to solve: %02d:%02d:%02d:%03d" % (h,m,s,ms), 'red', 1)

		# update the unknown container in self.model :
		self.update_model_var(self.get_unknown(), annotate=annotate)

	def update_model_var(self, u, annotate=False):
		"""
		Update the appropriate unknown variable in ``self.model`` to ``u``.
		"""
		raiseNotDefined()



