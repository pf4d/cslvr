from __future__ import division
from dolfin               import *
from dolfin_adjoint       import *
from cslvr.inputoutput    import print_text, print_min_max
from cslvr.d2model        import D2Model
from cslvr.latmodel       import LatModel
from cslvr.physics        import Physics
from cslvr.momentum       import Momentum
import sys






class MomentumPlaneStrainBase(Momentum):
	"""
	"""
	def __init__(self, model, solve_params=None,
		           linear=False, use_lat_bcs=False,
		           use_pressure_bc=True, stabilized=False):
		"""
		"""
		s = "::: INITIALIZING PLANE-STRAIN VELOCITY BASE PHYSICS :::"
		print_text(s, cls=self)

		if type(model) != LatModel:
			s = ">>> MomentumPlaneStrain REQUIRES A 'LatModel' INSTANCE, NOT %s <<<"
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
		self.assu  = FunctionAssigner(model.V, [self.Q.sub(0), self.Q.sub(1)])
		if stabilized:
			self.assp  = FunctionAssigner(model.Q, self.Q.sub(2))
		else:
			self.assp  = FunctionAssigner(model.Q, model.Q)

		"""
		# iterate through the facets and mark each if on a boundary :

		# 1: outflow
		# 2: inflow
		self.ff      = MeshFunction('size_t', model.mesh, 2, 0)
		for f in facets(model.mesh):
			n_f      = f.normal()
			x_m      = f.midpoint().x()
			z_m      = f.midpoint().y()
			u_x_m    = model.u_x_ob(x_m, z_m)
			u_z_m    = model.u_z_ob(x_m, z_m)
			u_dot_n  = u_x_m*n_f.x() + u_z_m*n_f.y()

			if   u_dot_n >= 0 and f.exterior():   self.ff[f] = 1  # outflow
			elif u_dot_n <  0 and f.exterior():   self.ff[f] = 2  # inflow

		self.ds         = Measure('ds', subdomain_data=self.ff)
		self.dGamma_in  = Boundary(self.ds, [2], 'inflow')
		self.dGamma_out = Boundary(self.ds, [1], 'outflow')

		u_x_bc_in  = DirichletBC(self.Q.sub(0), model.u_x_ob, self.ff, 2)
		u_z_bc_in  = DirichletBC(self.Q.sub(1), model.u_z_ob, self.ff, 2)
		p_bc_out   = DirichletBC(self.Q.sub(2), model.p,      self.ff, 1)
		"""

		# set boundary conditions :
		u_x_bc_l   = DirichletBC(self.Q.sub(0), model.u_x_ob, model.ff,
		                         model.GAMMA_L_UDR)
		u_x_bc_s   = DirichletBC(self.Q.sub(0), model.u_x_ob, model.ff,
		                         model.GAMMA_U_GND)
		u_x_bc_b   = DirichletBC(self.Q.sub(0), model.u_x_ob, model.ff,
		                         model.GAMMA_B_GND)
		u_z_bc_l   = DirichletBC(self.Q.sub(1), model.u_z_ob, model.ff,
		                         model.GAMMA_L_UDR)
		u_z_bc_s   = DirichletBC(self.Q.sub(1), model.u_z_ob, model.ff,
		                         model.GAMMA_U_GND)
		u_z_bc_b   = DirichletBC(self.Q.sub(1), model.u_z_ob, model.ff,
		                         model.GAMMA_B_GND)
		p_bc_l     = DirichletBC(self.Q.sub(2), model.p, model.ff,
		                         model.GAMMA_L_UDR)
		p_bc_s     = DirichletBC(self.Q.sub(2), model.p, model.ff,
		                         model.GAMMA_U_GND)
		p_bc_b     = DirichletBC(self.Q.sub(2), model.p, model.ff,
		                         model.GAMMA_B_GND)
		u_dot_n_bc_b = DirichletBC(self.Q.sub(0), model.B_ring, model.ff,
		                           model.GAMMA_B_GND)

		# momenturm functions :
		self.set_unknown(Function(self.Q, name = 'G'))
		self.set_trial_function(TrialFunction(self.Q))
		self.set_test_function(TestFunction(self.Q))

		n_x, n_z   = model.n_b

		t_11 = Function(model.Q, name='t_11')
		t_12 = Function(model.Q, name='t_12')
		t_21 = Function(model.Q, name='t_21')
		t_22 = Function(model.Q, name='t_22')

		model.assign_variable(t_11, 1.0)
		model.assign_variable(t_22, 1.0)
		model.assign_variable(t_12, 0.0)
		model.assign_variable(t_21, 0.0)

		T   = as_matrix([[t_11, t_12],
		                 [t_21, t_22]])
		#T = Identity(2)

		#T   = as_matrix([[n_x, n_y,  n_z],
		#                 [0,   n_z, -n_y],
		#                 [n_z, 0,   -n_x]])
		self.T = T

		mom_bcs = [u_x_bc_l, u_x_bc_s, u_x_bc_b,
		           u_z_bc_l, u_z_bc_s, u_z_bc_b,
		           p_bc_s,   p_bc_b,   p_bc_l]
		mom_bcs = []#[u_x_bc_in, u_z_bc_in]#, p_bc_out]
		mom_bcs = []#[u_dot_n_bc_b]#[u_x_bc_l, u_z_bc_l]
		self.set_boundary_conditions(mom_bcs)

		# finally, set up all the other variables and call the child class's
		# ``initialize`` method :
		super(MomentumPlaneStrainBase, self).__init__(model      = model,
		                                         solve_params    = solve_params,
		                                         linear          = linear,
		                                         use_lat_bcs     = use_lat_bcs,
		                                         use_pressure_bc = use_pressure_bc,
		                                         stabilized      = stabilized)

	def get_velocity(self):
		r"""
		Return the velocity :math:`\underline{u} = [u_x\ 0\ u_z]^{\intercal}`
		extracted from unknown function returned by
		:func:`~momentumstokes.MomentumStokesBase.get_unknown`.
		"""
		u_x, u_z, p = self.get_unknown()
		return as_vector([u_x, u_z])

	def update_model_var(self, u, annotate=False):
		"""
		Update the two horizontal components of velocity in ``self.model.u``
		to those given by ``u``.
		"""
		#T = self.T
		u_x, u_z, p = u.split()
		u = [u_x, u_z]

		if not self.stabilized:   p = project(p, self.model.Q, annotate=annotate)

		self.assu.assign(self.model.u, u, annotate=annotate)
		self.assp.assign(self.model.p, p, annotate=annotate)

		print_min_max(self.model.u, 'model.u', cls=self)
		print_min_max(self.model.p, 'model.p', cls=self)






class MomentumDukowiczPlaneStrain(MomentumPlaneStrainBase):
	"""
	"""
	def initialize(self, model, solve_params=None, stabilized=False,
	               linear=False, use_lat_bcs=False, use_pressure_bc=True):
		"""
		"""
		s = "::: INITIALIZING DUKOWICZ-PLANE-STRAIN PHYSICS :::"
		print_text(s, cls=self)

		mesh       = model.mesh
		S          = model.S
		z          = model.x[1]
		rho_i      = model.rho_i
		rhosw      = model.rhosw
		g          = model.g
		beta       = model.beta
		h          = model.h
		n          = model.N
		D          = model.D
		B_ring     = model.B_ring

		dOmega     = model.dOmega()
		dGamma_b   = model.dGamma_b()
		dGamma_bw  = model.dGamma_bw()
		dGamma_ltu = model.dGamma_ltu()
		dGamma_ld  = model.dGamma_ld()

		#===========================================================================
		# define variational problem :

		# momenturm functions :
		U   = self.get_unknown()
		dU  = self.get_trial_function()
		Phi = self.get_test_function()
		eta = self.get_viscosity(linear)
		Vd  = self.get_viscous_dissipation(linear)

		v_x, v_z, q  = Phi
		u_x, u_z, p  = U

		# create velocity vector :
		# collect the velocity vector to be of the same dimension os the model :
		u     = as_vector([u_x, u_z])
		v     = as_vector([v_x, v_z])
		f     = as_vector([0,   -rho_i*g])
		I     = Identity(self.model.dim)

		# potential energy :
		Pe   = - rho_i * g * u_z

		# dissipation by sliding :
		ut   = u - dot(u,n)*n
		Sl   = - 0.5 * beta * dot(ut, ut)

		# incompressibility constraint :
		Pc   = + p * div(u)

		# inpenetrability constraint :
		Nc   = - p * dot(u,n)

		# pressure boundary :
		Pb_w   = - rhosw*g*D * dot(u,n)
		Pb_l   = - rho_i*g*(S - z) * dot(u,n)

		# action :
		A      = (Vd - Pe - Pc)*dOmega - Nc*dGamma_b - Sl*dGamma_b - Pb_w*dGamma_bw

		if (not model.use_periodic and use_pressure_bc):
			s = "    - using water pressure lateral boundary condition -"
			print_text(s, cls=self)
			A -= Pb_w*dGamma_ltu

		if (not model.use_periodic and model.mark_divide and not use_lat_bcs):
			s = "    - using internal divide lateral pressure boundary condition -"
			print_text(s, cls=self)
			A -= Pb_l*dGamma_ld

		# add lateral boundary conditions :
		elif use_lat_bcs:
			s = "    - using internal divide lateral stress natural boundary" + \
			    " conditions -"
			print_text(s, cls=self)
			eta_l  = self.get_viscosity(linear=True)
			sig_l  = self.stress_tensor(model.u, model.p, eta_l)
			A     -= dot(dot(sig_l, n), u) * dGamma_ld

		# the first variation of the action integral A w.r.t. U in the
		# direction of a test function Phi; the extremum :
		resid = derivative(A, U, Phi)

		# stabilized form is identical to TH with the addition the following terms :
		if stabilized:
			def epsilon(u): return 0.5*(grad(u) + grad(u).T)
			def sigma(u,p): return 2*eta * epsilon(u) - p*I
			def L(u,p):     return -div(sigma(u,p))
			#tau    = h**2 / (12 * A**(-1/n) * rho_i**2)
			tau    = Constant(1e-12) * h**2 / (2*eta + DOLFIN_EPS)
			resid += inner(L(v,q), tau*(L(u,p) - f)) * dOmega

		# if the model is linear, replace the ``Function`` with a ``TrialFunction``:
		if linear: resid = replace(resid, {U : dU})

		# set this Physics instance's residual :
		self.set_residual(resid)



