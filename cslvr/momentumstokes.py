from dolfin               import *
from dolfin_adjoint       import *
from copy                 import deepcopy
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d3model        import D3Model
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
from cslvr.momentumbp     import MomentumBPBase
from cslvr.helper         import Boundary
from time                 import time
import numpy                  as np
import sys






class MomentumStokesBase(Momentum):
	"""
	"""
	def __init__(self, model, solve_params=None,
		           linear=False, use_lat_bcs=False,
		           use_pressure_bc=True, stabilized=False):
		"""
		"""
		s = "::: INITIALIZING STOKES VELOCITY BASE PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != D3Model:
			s = ">>> MomentumStokes REQUIRES A 'D3Model' INSTANCE, NOT %s <<<"
			print_text(s % type(model) , 'red', 1)
			sys.exit(1)

		self.stabilized = stabilized

		#===========================================================================
		# define variational problem :

		# function space is available for later use :
		if stabilized:
			print_text("    - using stabilized elements -", cls=self)
			self.Qe = model.STABe
			self.Q  = model.STAB
		else:
			print_text("    - using Taylor-Hood elements -", cls=self)
			if model.order < 2:
				s = ">>> MomentumStokes Taylor-Hood ELEMENT REQUIRES A 'D3Model'" \
				    + "INSTANCE OF ORDER GREATER THAN 1 <<<"
				print_text(s , 'red', 1)
				sys.exit(1)
			else:
				self.Qe = model.THe
				self.Q  = model.TH

		# function assigner goes from the U function solve to u vector
		self.assu  = FunctionAssigner(model.V, self.Q.sub(0))#model.V)#

		# the pressure may be solved using a lower order function space, and
		# so the FunctionAssigner must be made correctly :
		if stabilized:  self.assp = FunctionAssigner(model.Q, self.Q.sub(1))
		else:           self.assp = FunctionAssigner(model.Q, model.Q)

		"""
		# iterate through the facets and mark each if on a boundary :

		# 1: outflow
		# 2: inflow
		self.ff      = MeshFunction('size_t', model.mesh, 2, 0)
		for f in facets(model.mesh):
			n_f      = f.normal()
			x_m      = f.midpoint().x()
			y_m      = f.midpoint().y()
			z_m      = f.midpoint().z()
			u_x_m    = model.u_x_ob(x_m, y_m, z_m)
			u_y_m    = model.u_y_ob(x_m, y_m, z_m)
			u_z_m    = model.u_z_ob(x_m, y_m, z_m)
			u_dot_n  = u_x_m*n_f.x() + u_y_m*n_f.y() + u_z_m*n_f.z()

			if   u_dot_n >= 0 and f.exterior():   self.ff[f] = 1  # outflow
			elif u_dot_n <  0 and f.exterior():   self.ff[f] = 2  # inflow

		self.ds         = Measure('ds', subdomain_data=self.ff)
		self.dGamma_in  = Boundary(self.ds, [2], 'inflow')
		self.dGamma_out = Boundary(self.ds, [1], 'outflow')

		u_x_bc_in  = DirichletBC(self.Q.sub(0), model.u_x_ob, self.ff, 2)
		u_y_bc_in  = DirichletBC(self.Q.sub(1), model.u_y_ob, self.ff, 2)
		u_z_bc_in  = DirichletBC(self.Q.sub(2), model.u_z_ob, self.ff, 2)
		p_bc_out   = DirichletBC(self.Q.sub(3), model.p,    self.ff, 1)
		"""

		# set boundary conditions :
		u_x_bc_l   = DirichletBC(self.Q.sub(0).sub(0), model.u_x_ob, model.ff,
		                         model.GAMMA_L_UDR)
		u_x_bc_s   = DirichletBC(self.Q.sub(0).sub(0), model.u_x_ob, model.ff,
		                         model.GAMMA_U_GND)
		u_x_bc_b   = DirichletBC(self.Q.sub(0).sub(0), model.u_x_ob, model.ff,
		                         model.GAMMA_B_GND)
		u_y_bc_l   = DirichletBC(self.Q.sub(0).sub(1), model.u_y_ob, model.ff,
		                         model.GAMMA_L_UDR)
		u_y_bc_s   = DirichletBC(self.Q.sub(0).sub(1), model.u_y_ob, model.ff,
		                         model.GAMMA_U_GND)
		u_y_bc_b   = DirichletBC(self.Q.sub(0).sub(1), model.u_y_ob, model.ff,
		                         model.GAMMA_B_GND)
		u_z_bc_l   = DirichletBC(self.Q.sub(0).sub(2), model.u_z_ob, model.ff,
		                         model.GAMMA_L_UDR)
		u_z_bc_s   = DirichletBC(self.Q.sub(0).sub(2), model.u_z_ob, model.ff,
		                         model.GAMMA_U_GND)
		u_z_bc_b   = DirichletBC(self.Q.sub(0).sub(2), model.u_z_ob, model.ff,
		                         model.GAMMA_B_GND)
		p_bc_l     = DirichletBC(self.Q.sub(1), model.p, model.ff,
		                         model.GAMMA_L_UDR)
		p_bc_s     = DirichletBC(self.Q.sub(1), model.p, model.ff,
		                         model.GAMMA_U_GND)
		p_bc_b     = DirichletBC(self.Q.sub(1), model.p, model.ff,
		                         model.GAMMA_B_GND)
		u_dot_n_bc_b = DirichletBC(self.Q.sub(0).sub(0), model.B_ring, model.ff,
		                           model.GAMMA_B_GND)

		# momenturm functions :
		self.set_unknown(Function(self.Q, name = 'U'))
		self.set_trial_function(TrialFunction(self.Q))
		self.set_test_function(TestFunction(self.Q))

		mom_bcs = [u_x_bc_l, u_x_bc_s, u_x_bc_b,
		           u_y_bc_l, u_y_bc_s, u_y_bc_b,
		           u_z_bc_l, u_z_bc_s, u_z_bc_b,
		           p_bc_s,   p_bc_b,   p_bc_l]
		mom_bcs = []#[u_x_bc_in, u_y_bc_in, u_z_bc_in]#, p_bc_out]
		mom_bcs = []#[u_dot_n_bc_b]#[u_x_bc_l, u_y_bc_l, u_z_bc_l]
		self.u_dot_n_bc_b = u_dot_n_bc_b
		self.set_boundary_conditions(mom_bcs)

		# finally, set up all the other variables and call the child class's
		# ``initialize`` method :
		super(MomentumStokesBase, self).__init__(model           = model,
		                                         solve_params    = solve_params,
		                                         linear          = linear,
		                                         use_lat_bcs     = use_lat_bcs,
		                                         use_pressure_bc = use_pressure_bc,
		                                         stabilized      = stabilized)

	def assemble_transformation_tensor(self):
		"""
		To write.
		"""
		print_text("    - assembling transformation tensor T -", cls=self)
		model = self.model

		T_x = Function(self.Q, name='T_x')
		T_y = Function(self.Q, name='T_y')
		T_z = Function(self.Q, name='T_z')

		n_b             = model.n_b
		n_x, n_y, n_z   = n_b.split(True)

		t_0_e = Expression(('-n_z', '0.0', 'n_x'), n_z=n_z, n_x=n_x, \
		                   element=model.V.ufl_element())
		t_0 = interpolate(t_0_e, model.V)
		t_0 = model.normalize_vector(t_0)

		t_x, t_y, t_z = t_0.split(True)

		t_1_x = "t_y * n_z - n_y * t_z"
		t_1_y = "n_x * t_z - t_x * n_z"
		t_1_z = "t_x * n_y - n_x * t_y"

		t_1_e = Expression((t_1_x, t_1_y, t_1_z), \
		                   n_x=n_x, n_y=n_y, n_z=n_z, \
		                   t_x=t_x, t_y=t_y, t_z=t_z, \
		                   element=model.V.ufl_element())
		t_1 = interpolate(t_1_e, model.V)

		model.save_xdmf(n_b, 'n_b')
		model.save_xdmf(t_0, 't_0')
		model.save_xdmf(t_1, 't_1')

		self.T_x_bc = DirichletBC(self.Q.sub(0), n_b, model.ff, model.GAMMA_B_GND)
		self.T_y_bc = DirichletBC(self.Q.sub(0), t_0, model.ff, model.GAMMA_B_GND)
		self.T_z_bc = DirichletBC(self.Q.sub(0), t_1, model.ff, model.GAMMA_B_GND)

		self.T_x_bc.apply(T_x.vector())
		self.T_y_bc.apply(T_y.vector())
		self.T_z_bc.apply(T_z.vector())

		m_x_bc = DirichletBC(self.Q.sub(0).sub(0), 1, model.ff, model.GAMMA_B_GND)
		m_y_bc = DirichletBC(self.Q.sub(0).sub(1), 2, model.ff, model.GAMMA_B_GND)
		m_z_bc = DirichletBC(self.Q.sub(0).sub(2), 3, model.ff, model.GAMMA_B_GND)

		b_x_dofs = np.array(m_x_bc.get_boundary_values().keys(), dtype=np.intc)
		b_y_dofs = np.array(m_y_bc.get_boundary_values().keys(), dtype=np.intc)
		b_z_dofs = np.array(m_z_bc.get_boundary_values().keys(), dtype=np.intc)

		b_x_dofs.sort()
		b_y_dofs.sort()
		b_z_dofs.sort()

		T_i = T_x.vector().get_local()
		T_j = T_y.vector().get_local()
		T_k = T_z.vector().get_local()

		# Create a matrix full of zeros by integrating over empty domain.
		# (No elements are flagged with ID 999.)  For some reason this seems to
		# be the only way I can find to create A that allows BCs to be set
		# without errors :
		T  = assemble(inner(self.get_test_function(),
		                    self.get_trial_function())*dx(999))
		T  = as_backend_type(T).mat()

		# allocate memory for the non-zero elements :
		T.setPreallocationNNZ([3,3])
		T.setUp()

		# The following can be uncommented for this code to work even if you don't
		# know how many nonzeros per row to allocate:
		#T.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

		# set to identity matrix first :
		Istart, Iend = T.getOwnershipRange()
		for i in range(Istart, Iend): T[i,i] = 1.0

		# then set the valuse of the transformation tensor :
		for i,j,k in zip(b_x_dofs, b_y_dofs, b_z_dofs):
			T[i,i] = T_i[i]
			T[i,j] = T_i[j]
			T[i,k] = T_i[k]
			T[j,i] = T_j[i]
			T[j,j] = T_j[j]
			T[j,k] = T_j[k]
			T[k,i] = T_k[i]
			T[k,j] = T_k[j]
			T[k,k] = T_k[k]
		T.assemble()

		# save the dolfin matrix :
		self.T = Matrix(PETScMatrix(T))
		self.b_x_dofs = b_x_dofs

	def get_velocity(self):
		"""
		Return the velocity :math:`\underline{u} = [u_x\ u_y\ u_z]^{\intercal}`
		extracted from unknown function returned by
		:func:`~momentumstokes.MomentumStokesBase.get_unknown`.
		"""
		u_x, u_y, u_z, p = self.get_unknown()
		return as_vector([u_x, u_y, u_z])

	def solve(self, annotate=False):
		"""
		Perform the Newton solve of the full-Stokes equations
		"""
		model  = self.model
		params = self.solve_params

		# zero out self.velocity for good convergence for any subsequent solves,
		# e.g. model.L_curve() :
		model.assign_variable(self.get_unknown(), DOLFIN_EPS)

		# solve as defined in ``physics.Physics.solve()`` :
		super(Momentum, self).solve(annotate)

		#params['solver']['newton_solver']['linear_solver'] = 'gmres'
		#precond = 'fieldsplit'
		#model.home_rolled_newton_method(self.resid, self.U, self.mom_Jac,
		#                                self.mom_bcs, atol=1e-6, rtol=rtol,
		#                                relaxation_param=alpha, max_iter=maxit,
		#                                method=params['solver']['newton_solver']['linear_solver'], preconditioner=precond,
		#                                bp_Jac=self.bp_Jac,
		#                                bp_R=self.bp_R)

	def update_model_var(self, u, annotate=False):
		"""
		Update the two horizontal components of velocity in ``self.model.u``
		to those given by ``u``.
		"""
		u, p = u.split()

		if not self.stabilized:  p = interpolate(p, self.model.Q)

		self.assu.assign(self.model.u, u, annotate=annotate)
		self.assp.assign(self.model.p, p, annotate=annotate)

		print_min_max(self.model.u, 'model.u', cls=self)
		print_min_max(self.model.p, 'model.p', cls=self)






class MomentumDukowiczStokesReduced(MomentumBPBase):
	"""
	"""
	# FIXME: this fails completely with model.order > 1.
	def initialize(self, model, solve_params=None,
		             linear=False, use_lat_bcs=False, use_pressure_bc=True):
		"""
		Here we set up the problem, and do all of the differentiation and
		memory allocation type stuff.
		"""

		s = "::: INITIALIZING DUKOWICZ REDUCED FULL-STOKES PHYSICS :::"
		print_text(s, cls=self)

		# NOTE: not sure why this is ever changed, but the model.assimilate_data
		#       method throws an error if I don't do this :
		parameters["adjoint"]["stop_annotating"] = False

		S          = model.S
		B          = model.B
		B_ring     = model.B_ring
		z          = model.x[2]
		rho_i      = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		h          = model.h
		n          = model.N
		D          = model.D

		dOmega     = model.dOmega()
		dGamma     = model.dGamma()
		dGamma_b   = model.dGamma_b()
		dGamma_s   = model.dGamma_s()
		dGamma_l   = model.dGamma_l()
		dGamma_bg  = model.dGamma_bg()
		dGamma_bw  = model.dGamma_bw()
		dGamma_ld  = model.dGamma_ld()
		dGamma_ltu = model.dGamma_ltu()
		dGamma_lt  = model.dGamma_lt()

		# new constants :
		p0         = 101325
		T0         = 288.15
		M          = 0.0289644

		#===========================================================================
		# define variational problem :

		# momenturm variables :
		u_h    = self.get_unknown()
		du_h   = self.get_trial_function()
		v_h    = self.get_test_function()
		Vd     = self.get_viscous_dissipation(linear)

		# upper surface gradient and lower surface normal :
		grad_S_h  = as_vector([S.dx(0), S.dx(1)])
		n_h       = as_vector([n[0],    n[1]])
		n_z       = n[2]

		# function assigner goes from the U function solve to U3 vector
		# function used to save :
		v_x, v_y  = v_h
		u_x, u_y  = u_h
		u_z       = self.u_z

		# three-dimensional velocity vector :
		u      = as_vector([u_x, u_y, u_z])

		# potential energy (note u_z does not contribute after derivation) :
		Pe     = - rho_i * g * (dot(u_h, grad_S_h) + u_z)

		# dissipation by sliding :
		u_z_b  = (- B_ring - dot(u_h,n_h)) / n_z
		Sl_gnd = - 0.5 * beta * (u_x**2 + u_y**2 + u_z_b**2)

		# pressure boundary :
		f_w    = rho_i*g*(S - z) - rhosw*g*D
		Pb     = f_w * dot(u, n)

		# action :
		A      = (Vd - Pe)*dOmega - Sl_gnd*dGamma_bg - Pb*dGamma_bw

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
			eta_l    = self.get_viscosity(linear=True)
			sig_l    = self.stress_tensor(model.u, model.p, eta_l)
			A -= dot(dot(sig_l, n), u) * dGamma_ld

		# the first variation of the action in the direction of a
		# test function ; the extremum :
		self.set_residual(derivative(A, u_h, v_h))

	def get_velocity(self):
		"""
		Return the velocity :math:`\underline{u} = [u_x\ u_y\ u_z]^{\intercal}`
		with horizontal compoents taken from the unknown function returned by
		:func:`~momentumbp.MomentumBPBase.get_unknown` and vertical component from
		``self.u_z``.
		"""
		u_x, u_y = self.get_unknown()
		return as_vector([u_x, u_y, self.u_z])

	def strain_rate_tensor(self, u):
		"""
		return the Dukowicz reduced-Stokes strain-rate tensor for the
		velocity ``u``.
		"""
		print_text("    - using Dukowicz reduced-Stokes strain-rate tensor -", \
		           cls=self)
		u_x, u_y, u_z  = u
		epi            = 0.5 * (grad(u) + grad(u).T)
		epi22          = -u_x.dx(0) - u_y.dx(1)          # incompressibility
		return as_matrix([[epi[0,0],  epi[0,1],  epi[0,2]],
		                  [epi[1,0],  epi[1,1],  epi[1,2]],
		                  [epi[2,0],  epi[2,1],  epi22]])

	def default_solve_params(self):
		"""
		Returns a set of default solver parameters that yield good performance
		"""
		nparams = {'newton_solver' :
		          {
		            'linear_solver'            : 'cg',
		            'preconditioner'           : 'hypre_amg',
		            'relative_tolerance'       : 1e-9,
		            'relaxation_parameter'     : 0.8,
		            'maximum_iterations'       : 50,
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
		m_params  = {'solver'               : nparams,
		             'solve_vert_velocity'  : True,
		             'solve_pressure'       : True,
		             'vert_solve_method'    : 'mumps'}
		return m_params

	def solve(self, annotate=False):
		"""
		Perform the Newton solve of the reduced full-Stokes equations
		"""
		model  = self.model
		params = self.solve_params

		# solve non-linear system :
		rtol     = params['solver']['newton_solver']['relative_tolerance']
		maxit    = params['solver']['newton_solver']['maximum_iterations']
		alpha    = params['solver']['newton_solver']['relaxation_parameter']
		lin_slv  = params['solver']['newton_solver']['linear_solver']
		precon   = params['solver']['newton_solver']['preconditioner']
		err_conv = params['solver']['newton_solver']['error_on_nonconvergence']
		s    = "::: solving Dukowicz full-Stokes reduced equations with %i max" + \
		         " iterations and step size = %.1f :::"
		print_text(s % (maxit, alpha), cls=self)

		# zero out self.velocity for good convergence for any subsequent solves,
		# e.g. model.L_curve() :
		#model.assign_variable(self.get_unknown(), DOLFIN_EPS)
		#model.assign_variable(self.u_z,     DOLFIN_EPS)

		def cb_ftn():  self.solve_vert_velocity(annotate)

		# compute solution :
		#solve(self.resid == 0, self.U, J = self.mom_Jac, bcs = self.mom_bcs,
		#      annotate = annotate, solver_parameters = params['solver'])
		model.home_rolled_newton_method(self.get_residual(), self.get_unknown(), \
		                                self.get_jacobian(), \
		                                self.get_boundary_conditions(),
		                                atol=1e-6, rtol=rtol,
		                                relaxation_param=alpha, max_iter=maxit,
		                                method=lin_slv, preconditioner=precon,
		                                cb_ftn=cb_ftn)

		# solve for pressure if desired :
		if params['solve_pressure']:    self.solve_pressure(annotate)

		# update the model's momentum container :
		self.update_model_var(self.get_unknown(), annotate=annotate)

	def update_model_var(self, u, annotate=False):
		"""
		Update the two horizontal components of velocity in ``self.model.u``
		to those given by ``u``.
		"""
		u_model = self.model.u
		self.assu.assign([u_model.sub(0), u_model.sub(1)], u, annotate=annotate)
		print_min_max(self.model.u, 'model.u', cls=self)






class MomentumDukowiczStokes(MomentumStokesBase):
	"""
	"""
	def initialize(self, model, solve_params=None,
		             linear=False, use_lat_bcs=False,
		             use_pressure_bc=True, stabilized=True):
		"""
		Here we set up the problem, and do all of the differentiation and
		memory allocation type stuff.
		"""
		s = "::: INITIALIZING DUKOWICZ-STOKES PHYSICS :::"
		print_text(s, cls=self)

		S          = model.S
		z          = model.x[2]
		rho_i      = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		h          = model.h
		n          = model.N
		D          = model.D
		B          = model.B
		B_ring     = model.B_ring
		dBdt       = model.dBdt

		dOmega     = model.dOmega()
		dGamma     = model.dGamma()
		dGamma_b   = model.dGamma_b()
		dGamma_bg  = model.dGamma_bg()
		dGamma_bw  = model.dGamma_bw()
		dGamma_ld  = model.dGamma_ld()
		dGamma_ltu = model.dGamma_ltu()

		#===========================================================================
		# define variational problem :

		# momenturm functions :
		U               = self.get_unknown()
		trial_function  = self.get_trial_function()
		Phi             = self.get_test_function()
		eta             = self.get_viscosity(linear)
		Vd              = self.get_viscous_dissipation(linear)

		v_x, v_y, v_z, q  = Phi
		u_x, u_y, u_z, p  = U

		# create velocity vector :
		u       = as_vector([u_x, u_y, u_z])
		v       = as_vector([v_x, v_y, v_z])
		f       = as_vector([0,   0,   -rho_i*g])
		I       = Identity(3)

		# potential energy :
		Pe     = - rho_i * g * u_z

		# dissipation by sliding :
		ut     = u - dot(u,n)*n                     # tangential comp. of velocity
		Sl_gnd = - 0.5 * beta * dot(ut, ut)

		# incompressibility constraint :
		Pc     = + p * div(u)

		# impenetrability constraint :
		n_mag  = sqrt(1 + dot(grad(B), grad(B)))    # outward normal vector mag
		u_n    = - dBdt / n_mag - B_ring            # normal component of velocity
		Nc     = - p * (dot(u,n) - u_n)

		# pressure boundary :
		Pb_w   = - rhosw*g*D * dot(u,n)             # water pressure
		Pb_l   = - rho_i*g*(S - z) * dot(u,n)       # hydrostatic ice pressure

		# action :
		A      = + (Vd - Pe - Pc)*dOmega \
		         - Nc*dGamma_b - Sl_gnd*dGamma_bg - Pb_w*dGamma_bw

		if (not model.use_periodic and use_pressure_bc):
			s = "    - using water pressure lateral boundary condition -"
			print_text(s, cls=self)
			A -= Pb_w*dGamma_ltu

		if (not model.use_periodic and model.mark_divide and not use_lat_bcs):
			s = "    - using internal divide lateral pressure boundary condition -"
			print_text(s, cls=self)
			A -= Pb_l*dGamma_ld

		# add lateral boundary conditions :
		if use_lat_bcs:
			s = "    - using internal divide lateral stress natural boundary" + \
			    " conditions -"
			print_text(s, cls=self)
			eta_l    = self.get_viscosity(linear=True)
			sig_l    = self.stress_tensor(model.u, model.p, eta_l)
			A       -= dot(dot(sig_l, n), u) * dGamma_ld

		# the first variation of the action integral A w.r.t. U in the
		# direction of a test function Phi; the extremum :
		resid = derivative(A, U, Phi)

		# stabilized form is identical to TH with the addition the following terms :
		if stabilized:
			def epsilon(u): return 0.5*(grad(u) + grad(u).T)
			def sigma(u,p): return 2*eta * epsilon(u) - p*I
			def L(u,p):     return -div(sigma(u,p))
			#tau         = h**2 / (12 * A**(-1/n) * rho_i**2)
			tau    = Constant(1e-6) * h**2 / (2*eta + DOLFIN_EPS)
			resid += inner(L(v,q), tau*(L(u,p) - f)) * dOmega

		# if the model is linear, replace the ``Function`` with a ``TrialFunction``:
		if linear: resid = replace(resid, {U : trial_function})

		# set this Physics instance's residual :
		self.set_residual(resid)






class MomentumNitscheStokes(MomentumStokesBase):
	"""
	"""
	def initialize(self, model, solve_params=None,
		             linear=False, use_lat_bcs=False,
		             use_pressure_bc=True, stabilized=True):
		"""
		Here we set up the problem, and do all of the differentiation and
		memory allocation type stuff.
		"""
		s = "::: INITIALIZING NITSCHE-STOKES PHYSICS :::"
		print_text(s, cls=self)

		# save the solver parameters :
		self.stabilized   = stabilized

		S          = model.S
		B          = model.B
		dBdt       = model.dBdt
		B_ring     = model.B_ring
		z          = model.x[2]
		rho_i      = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		h          = model.h
		n          = model.N
		D          = model.D

		dOmega     = model.dOmega()
		dGamma     = model.dGamma()
		dGamma_b   = model.dGamma_b()
		dGamma_bg  = model.dGamma_bg()
		dGamma_bw  = model.dGamma_bw()
		dGamma_ld  = model.dGamma_ld()
		dGamma_ltu = model.dGamma_ltu()

		# new constants :
		p0         = 101325
		T0         = 288.15
		M          = 0.0289644

		#===========================================================================
		# define variational problem :

		# momenturm functions :
		U               = self.get_unknown()
		trial_function  = self.get_trial_function()
		Phi             = self.get_test_function()
		eta             = self.get_viscosity(linear)

		# get velocity, pressure spaces :
		v, q  = split(Phi)
		u, p  = split(U)

		gamma = Constant(1e13)
		f     = Constant((0.0, 0.0, -rho_i * g))
		I     = Identity(3)
		n_mag = sqrt(1 + dot(grad(B), grad(B)))    # outward normal vector mag
		u_n   = - dBdt / n_mag - B_ring            # normal component of velocity

		# pressure boundary :
		ut    = u - dot(u,n)*n    # tangential component of velocity
		Pb_w  = rhosw*g*D         # water pressure
		Pb_l  = rho_i*g*(S - z)   # hydrostatic ice pressure

		def epsilon(u): return 0.5*(grad(u) + grad(u).T)
		def sigma(u,p): return 2*eta * epsilon(u) - p*I
		def L(u,p):     return -div(sigma(u,p))
		def G(u,p):     return dot(sigma(u,p), n)

		B_o = + inner(sigma(u,p),grad(v))*dOmega \
		      + div(u)*q*dOmega

		B_g = -            dot(v,n)        * dot(n,G(u,p)) * dGamma_b \
		      -           (dot(u,n) - u_n) * dot(n,G(v,q)) * dGamma_b \
		      + gamma/h * (dot(u,n) - u_n) * dot(v,n)      * dGamma_b \
		      + beta * dot(ut, v) * dGamma_b \
		      + p * dot(v,n)      * dGamma_b

		F   = + dot(f,v) * dOmega

		# define the residual :
		resid = B_o + B_g - F

		# stabilized form is identical to TH with the addition the following terms :
		if stabilized:
			# intrinsic-time stabilization parameter :
			tau    = Constant(1e-9) * h**2 / (2*eta + DOLFIN_EPS)
			resid += inner(L(v,q), tau*(L(u,p) - f)) * dOmega

		if (not model.use_periodic and use_pressure_bc):
			s = "    - using water pressure lateral boundary condition -"
			print_text(s, cls=self)
			resid += Pb_w * dot(v,n) * dGamma_ltu

		if (not model.use_periodic and model.mark_divide and not use_lat_bcs):
			s = "    - using internal divide lateral pressure boundary condition -"
			print_text(s, cls=self)
			resid += Pb_l * dot(v,n) * dGamma_ld

		# add lateral boundary conditions :
		elif use_lat_bcs:
			s = "    - using internal divide lateral stress natural boundary" + \
			    " conditions -"
			print_text(s, cls=self)
			eta_l  = self.get_viscosity(linear=True)
			sig_l  = self.stress_tensor(model.u, model.p, eta_l)
			resid += dot(dot(sig_l, n), v) * dGamma_ld

		# if the model is linear, replace the ``Function`` with a ``TrialFunction``:
		if linear: resid = replace(resid, {U : trial_function})

		# set this Physics instance's residual :
		self.set_residual(resid)

		# Form for use in constructing preconditioner matrix
		#self.bp_R = inner(grad(u), grad(v))*dOmega + p*q*dOmega
		#self.bp_Jac = derivative(self.bp_R, U, trial_function)






class MomentumStokes(MomentumStokesBase):
	"""
	"""
	def initialize(self, model, solve_params=None,
		             linear=False, use_lat_bcs=False,
		             use_pressure_bc=True, stabilized=True):
		"""
		Here we set up the problem, and do all of the differentiation and
		memory allocation type stuff.
		"""
		s = "::: INITIALIZING STOKES PHYSICS :::"
		print_text(s, cls=self)

		S          = model.S
		B          = model.B
		dBdt       = model.dBdt
		B_ring     = model.B_ring
		z          = model.x[2]
		rho_i      = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		h          = model.h
		n          = model.N
		D          = model.D

		dOmega     = model.dOmega()
		dGamma     = model.dGamma()
		dGamma_b   = model.dGamma_b()
		dGamma_bg  = model.dGamma_bg()
		dGamma_bw  = model.dGamma_bw()
		dGamma_ld  = model.dGamma_ld()
		dGamma_ltu = model.dGamma_ltu()

		# new constants :
		p0         = 101325
		T0         = 288.15
		M          = 0.0289644

		#===========================================================================
		# define variational problem :

		# momenturm functions :
		if linear:
			U              = self.get_trial_function()
			trial_function = U
		else:
			U              = self.get_unknown()
			trial_function = self.get_trial_function()

		# momenturm functions :
		Phi = self.get_test_function()
		eta = self.get_viscosity(linear)

		# get velocity, pressure spaces :
		v, q  = split(Phi)
		u, p  = split(U)

		# form essential parts :
		f     = Constant((0.0, 0.0, -rho_i * g))
		I     = Identity(3)
		n_mag = sqrt(1 + dot(grad(B), grad(B)))    # outward normal vector mag
		u_n   = - dBdt / n_mag - B_ring            # normal component of velocity

		# pressure boundary :
		ut    = u - dot(u,n)*n                     # tang. component of velocity
		Pb_w  = rhosw*g*D                          # water pressure
		Pb_l  = rho_i*g*(S - z)                    # hydrostatic ice pressure

		def epsilon(u): return 0.5 * (grad(u) + grad(u).T)
		def tau(u):     return 2 * eta * epsilon(u)
		def sigma(u,p): return tau(u) - p*I
		def L(u,p):     return -div(sigma(u,p))
		def G(u,p):     return dot(sigma(u,p), n)

		#self.A  = inner(sigma(u,p),grad(v))*dOmega
		#self.B  = div(u)*q*dOmega

		# split the stress tensor into components for matrix manipulations :
		self.A     = inner(tau(u), grad(v)) * dOmega
		self.S     = beta * dot(ut, v) * dGamma_b
		self.P     = p * dot(v,n) * dGamma_b
		self.B     = - p * div(v) * dOmega
		self.BT    = - q * div(u) * dOmega
		self.f     = dot(f,v) * dOmega

		# define the residual :
		resid = self.A + self.S + self.B + self.BT - self.f

		# stabilized form is identical to TH with the addition the following terms :
		if stabilized:
			# intrinsic-time stabilization parameter :
			tau_stz     = Constant(1e-9) * h**2 / (2*eta + DOLFIN_EPS)
			self.B_stz  = inner(L(v,q), tau_stz*L(u,p)) * dOmega
			self.f_stz  = inner(L(v,q), tau_stz*f) * dOmega
			resid      += self.B_stz - self.f_stz

		if (not model.use_periodic and use_pressure_bc):
			s = "    - using water pressure lateral boundary condition -"
			print_text(s, cls=self)
			resid += Pb_w * dot(v,n) * dGamma_ltu

		if (not model.use_periodic and model.mark_divide and not use_lat_bcs):
			s = "    - using internal divide lateral pressure boundary condition -"
			print_text(s, cls=self)
			resid += Pb_l * dot(v,n) * dGamma_ld

		# add lateral boundary conditions :
		elif use_lat_bcs:
			s = "    - using internal divide lateral stress natural boundary" + \
			    " conditions -"
			print_text(s, cls=self)
			eta_l  = self.get_viscosity(linear=True)
			sig_l  = self.stress_tensor(model.u, model.p, eta_l)
			resid += dot(dot(sig_l, n), v) * dGamma_ld

		# set this Physics instance's residual :
		self.set_residual(resid)

		# Form for use in constructing preconditioner matrix
		#self.bp_R = inner(grad(u), grad(v))*dOmega + p*q*dOmega
		#self.bp_Jac = derivative(self.bp_R, U, trial_function)

	def solve(self, annotate=False):
		"""
		Perform the Newton solve of the full-Stokes equations
		"""
		model  = self.model
		params = self.solve_params

		self.assemble_transformation_tensor()

		# zero out self.velocity for good convergence for any subsequent solves,
		# e.g. model.L_curve() :
		model.assign_variable(self.get_unknown(), DOLFIN_EPS)

		A      = assemble(self.A)
		B      = assemble(self.B)
		BT     = assemble(self.B)
		S      = assemble(self.S)
		f      = assemble(self.f)

		T      = as_backend_type(self.T).mat()
		A      = as_backend_type(A).mat()
		B      = as_backend_type(B).mat()
		BT     = as_backend_type(BT).mat()
		S      = as_backend_type(S).mat()
		f      = as_backend_type(f).vec()

		A_n    = T.matMult(A).matTransposeMult(T)
		B_n    = T.matMult(B).matTransposeMult(T)
		BT_n   = T.transposeMatMult(BT).matMult(T)
		S_n    = T.matMult(S).matTransposeMult(T)
		f_n    = T * f

		self.A_n  = A_n
		self.B_n  = B_n
		self.BT_n = BT_n
		self.S_n  = S_n
		self.f_n  = f_n

		F      = Matrix(PETScMatrix(A_n + B_n + S_n + BT_n))
		f      = Vector(PETScVector(f_n))

		self.u_dot_n_bc_b.apply(f)

		as_backend_type(F).mat().zeroRowsColumns(self.b_x_dofs, 1.0)
		as_backend_type(F).mat().assemblyBegin()
		as_backend_type(F).mat().assemblyEnd()
		self.u_dot_n_bc_b.apply(f)

		solve(F, self.get_unknown().vector(), f, annotate=False)

		#params['solver']['newton_solver']['linear_solver'] = 'gmres'
		#precond = 'fieldsplit'
		#model.home_rolled_newton_method(self.resid, self.U, self.mom_Jac,
		#                                self.mom_bcs, atol=1e-6, rtol=rtol,
		#                                relaxation_param=alpha, max_iter=maxit,
		#                                method=params['solver']['newton_solver']['linear_solver'], preconditioner=precond,
		#                                bp_Jac=self.bp_Jac,
		#                                bp_R=self.bp_R)

		self.update_model_var(self.get_unknown(), annotate=annotate)

	def update_model_var(self, u, annotate=False):
		"""
		Update the two horizontal components of velocity in ``self.model.u``
		to those given by ``u``.
		"""
		Tm  = as_backend_type(self.T).mat()
		uv  = as_backend_type(u.vector()).vec()
		u_n = Tm.transpose() * uv
		u_n = Vector(PETScVector(u_n))

		u.vector().set_local(u_n.get_local())
		u.vector().apply('insert')

		super(MomentumStokes, self).update_model_var(u, annotate)



