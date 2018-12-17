from __future__ import division
from builtins import object
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.utilities.codegen  import ccode
from cslvr.inputoutput        import print_text, print_min_max
from cslvr.momentumbp         import MomentumBP, MomentumDukowiczBP
from cslvr.momentumstokes     import MomentumNitscheStokes, MomentumStokes, \
                                     MomentumDukowiczStokes
from dolfin                   import Expression, interpolate, \
                                     as_vector, as_matrix
import matplotlib.pyplot          as plt
import sympy                      as sp
import numpy                      as np






class Verification(object):

	def __init__(self):
		"""
		"""
		print_text("::: INITIALIZING Verification OBJECT :::", cls=self)

		# the coordinates as expressed in FEniCS C++ code :
		x,y,z  = sp.symbols('x[0], x[1], x[2]')
		self.x = x
		self.y = y
		self.z = z

	def init_expressions(self, s, b, u_xs, u_xb, dsdt, dbdt, s_ring, b_ring, lam):
		"""
		"""
		print_text("::: INITIALIZING ANALYTIC EXPRESSIONS :::", cls=self)

		x,y,z = self.x, self.y, self.z

		# form the fundamental expressions :
		self.s      = s(x,y)
		self.b      = b(x,y)
		self.u_xs   = u_xs(x,y)
		self.u_xb   = u_xb(x,y)
		self.dsdt   = dsdt(x,y)
		self.dbdt   = dbdt(x,y)
		self.s_ring = s_ring(x,y)
		self.b_ring = b_ring(x,y)
		self.lam    = lam

		#===========================================================================
		# form the derived expressions :

		# thickness :
		self.h        = self.s - self.b

		# relative depth :
		self.a_b      = (self.s - z) / self.h

		# relative depth :
		self.a_s      = (z - self.b) / self.h

		# x-derivative of relative depth :
		self.da_b_dx  = self.a_b.diff(x, 1)

		# y-derivative of relative depth :
		self.da_b_dy  = self.a_b.diff(y, 1)

		# x-derivative of upper surface
		self.dsdx     = self.s.diff(x, 1)

		# y-derivative of upper surface
		self.dsdy     = self.s.diff(y, 1)

		# x-derivative of lower surface
		self.dbdx     = self.b.diff(x, 1)

		# y-derivative of lower surface
		self.dbdy     = self.b.diff(y, 1)

		# x-derivative of thickness :
		self.dhdx     = self.h.diff(x, 1)

		# y-derivative of thickness :
		self.dhdy     = self.h.diff(y, 1)

		# rate of change of thickness :
		self.dhdt     = self.dsdt - self.dbdt

		# outward-pointing-normal-vector magnitude at upper surface :
		self.n_mag_s  = sp.sqrt( 1 + self.dsdx**2 + self.dsdy**2 )

		# outward-pointing-normal-vector magnitude at lower surface :
		self.n_mag_b  = sp.sqrt( 1 + self.dbdx**2 + self.dbdy**2 )

		# thickness forcing term :
		self.h_ring   = self.n_mag_s * self.s_ring + self.n_mag_b * self.b_ring

		# x-derivative of the x-component of velocity at the upper surface :
		self.du_xs_dx = self.u_xs.diff(x, 1)

		# y-derivative of the x-component of velocity at the upper surface :
		self.du_xs_dy = self.u_xs.diff(y, 1)

		# x-derivative of the x-component of velocity at the lower surface :
		self.du_xb_dx = self.u_xb.diff(x, 1)

		# y-derivative of the x-component of velocity at the lower surface :
		self.du_xb_dy = self.u_xb.diff(y, 1)

		# x-component of velocity :
		self.u_x      = + (self.u_xs - self.u_xb) * (1 - self.a_b**self.lam) \
		                + self.u_xb

		# first integrand of u_y() :
		self.A_11h    = self.h * self.du_xs_dx + self.dhdt + self.u_xs * self.dhdx

		# possibly elliptic upper surface-mass-balance integrand of u_y() :
		self.A_12h    = - self.n_mag_s * self.s_ring

		# possibly elliptic lower surface-mass-balance integrand of u_y() :
		self.A_13h    = - self.n_mag_b * self.b_ring

		# fourth integrand of u_y() :
		self.A_2h     = + self.h * (self.du_xb_dx - self.du_xs_dx) \
		                + (self.u_xs - self.u_xb) * self.dbdx

		# fifth integrand of u_y() :
		self.A_3h     = (self.u_xs - self.u_xb) * self.dsdx

		# y-component of velocity :
		self.u_y      = - 1/self.h * sp.integrate( self.A_11h, y) \
		                - 1/self.h * sp.integrate( self.A_12h, y) \
		                - 1/self.h * sp.integrate( self.A_13h, y) \
		                - 1/self.h * (1 - self.a_s)**self.lam \
		                           * sp.integrate( self.A_2h, y) \
		                - 1/self.h * (1 - self.a_s)**(self.lam - 1) \
		                           * (self.a_s - 1) \
		                           * sp.integrate( self.A_3h, y)

		# y-component of velocity at the upper surface :
		self.u_ys     = self.u_y.replace(z, self.s)

		# y-component of velocity at the lower surface :
		self.u_yb     = self.u_y.replace(z, self.b)

		# z-component of velocity at the upper surface :
		self.u_zs     = - self.n_mag_s * self.s_ring + self.dsdt \
		                + self.u_xs * self.dsdx + self.u_ys * self.dsdy

		# z-component of velocity at the upper surface :
		self.u_zb     = + self.n_mag_b * self.b_ring + self.dbdt \
		                + self.u_xb * self.dbdx + self.u_yb * self.dbdy

		# z-component of velocity :
		self.u_z      = self.a_s * self.u_zs + self.a_b * self.u_zb

		# velocity vector :
		self.u        = sp.Matrix([self.u_x, self.u_y, self.u_z])

		# velocity vector on upper surface :
		self.u_s      = sp.Matrix([self.u_xs, self.u_ys, self.u_zs])

		# velocity vector on lower surface :
		self.u_b      = sp.Matrix([self.u_xb, self.u_yb, self.u_zb])

		# velocity magnitude :
		self.u_mag    = sp.sqrt(self.u_x**2 + self.u_y**2 + self.u_z**2)

		# x derivative of x-component of velocity :
		self.du_x_dx  = self.u_x.diff(x, 1)

		# y derivative of x-component of velocity :
		self.du_x_dy  = self.u_x.diff(y, 1)

		# z derivative of x-component of velocity :
		self.du_x_dz  = self.u_x.diff(z, 1)

		# x derivative of y-component of velocity :
		self.du_y_dx  = self.u_y.diff(x, 1)

		# y derivative of y-component of velocity :
		self.du_y_dy  = self.u_y.diff(y, 1)

		# z derivative of y-component of velocity :
		self.du_y_dz  = self.u_y.diff(z, 1)

		# x-derivative of z-component of velocity :
		self.du_z_dx  = self.u_z.diff(x, 1)

		# y-derivative of z-component of velocity :
		self.du_z_dy  = self.u_z.diff(y, 1)

		# z-derivative of z-component of velocity :
		self.du_z_dz  = self.u_z.diff(z, 1)

		# velocity divergence :
		self.div_u    = self.du_x_dx + self.du_y_dy + self.du_z_dz

	def init_beta(self, beta):
		"""
		"""
		self.beta   = beta(self.x, self.y)

	def init_T(self, T):
		"""
		"""
		self.T      = T(self.x, self.y, self.z)

	# set the finite element model :
	def set_model(self, model):
		"""
		"""
		self.model = model
		self.Qe    = model.Qe
		self.Ve    = model.Ve

	def init_r2_stress_balance(self):
		"""
		"""
		print_text("::: initializing R^2 stress balance terms:::", cls=self)

		# vertically-integrated velocity divergence :
		self.int_div_u  = sp.integrate( self.div_u, (self.z, self.b, self.s))

		# vertically-averaged x-component of velocity :
		#self.u_xbar     = 1/self.h * sp.integrate( self.u_x, (z, self.b, self.s))
		self.u_xbar     = sp.integrate( self.u_x, (self.z, self.b, self.s))

		# vertically-averaged y-component of velocity :
		#self.u_ybar     = 1/self.h * sp.integrate( self.u_y, (z, self.b, self.s))
		self.u_ybar     = sp.integrate( self.u_y, (self.z, self.b, self.s))

		# vertically-averaged z-component of velocity :
		#self.u_zbar     = 1/self.h * sp.integrate( self.u_z, (z, self.b, self.s))
		self.u_zbar     = sp.integrate( self.u_z, (self.z, self.b, self.s))

		# x-derivative of vertically-integrated x-component of velocity :
		self.du_xbar_dx = self.u_xbar.diff(self.x, 1)

		# y-derivative of vertically-integrated x-component of velocity :
		self.du_xbar_dy = self.u_xbar.diff(self.y, 1)

		# x-derivative of vertically-integrated y-component of velocity :
		self.du_ybar_dx = self.u_ybar.diff(self.x, 1)

		# y-derivative of vertically-integrated y-component of velocity :
		self.du_ybar_dy = self.u_ybar.diff(self.y, 1)

		# thickness flux :
		#self.div_hu     = + self.u_xbar * self.dhdx + self.u_ybar * self.dhdy \
		#                  + self.h * (self.du_xbar_dx + self.du_ybar_dy)
		self.div_hu     = self.du_xbar_dx + self.du_ybar_dy

		# Leibniz's rule/thickness-balance residual check :
		self.leibniz_resid  = self.dhdt + self.div_hu - self.h_ring

		# inverse method for calculating _b_ring :
		self.b_ring_inverse = (self.dhdt + self.div_hu - self.n_mag_s*self.s_ring) \
		                      / self.n_mag_b

		# error in assuming a flat lower surface for deriving basal mass balance :
		self.epsilon_b      = (self.n_mag_b - 1) * self.b_ring_inverse

	def init_r3_stress_balance(self, momentum):
		"""
		"""
		print_text("::: initializing R^3 stress balance terms:::", cls=self)

		x, y, z = self.x, self.y, self.z

		# collect constants from FE model :
		a_T_l   = sp.Rational(self.model.a_T_l(0))
		a_T_u   = sp.Rational(self.model.a_T_u(0))
		Q_T_l   = sp.Rational(self.model.Q_T_l(0))
		Q_T_u   = sp.Rational(self.model.Q_T_u(0))
		R       = sp.Rational(self.model.R(0))
		n       = sp.Rational(self.model.n(0))
		eps_reg = sp.Rational(self.model.eps_reg(0))
		rho_i   = sp.Rational(self.model.rho_i(0))
		g       = sp.Rational(self.model.g(0))

		# x-coordinate unit vector :
		self.i_hat  = sp.Matrix([1, 0, 0])

		# y-coordinate unit vector :
		self.j_hat  = sp.Matrix([0, 1, 0])

		# z-coordinate unit vector :
		self.k_hat  = sp.Matrix([0, 0, 1])

		# gradient of upper surface :
		self.grad_s = sp.Matrix([self.dsdx, self.dsdy, 0])

		# gradient of lower surface :
		self.grad_b = sp.Matrix([self.dbdx, self.dbdy, 0])

		# outward-pointing-normal vector at upper surface :
		self.n_s    = (self.k_hat - self.grad_s) / self.n_mag_s

		# outward-pointing-normal vector at lower surface :
		self.n_b    = (self.grad_b - self.k_hat) / self.n_mag_b

		# outward-pointing-normal vector at north surface :
		self.n_N    = + 1 * self.j_hat

		# outward-pointing-normal vector at south surface :
		self.n_S    = - 1 * self.j_hat

		# outward-pointing-normal vector at east surface :
		self.n_E    = + 1 * self.i_hat

		# outward-pointing-normal vector at west surface :
		self.n_W    = - 1 * self.i_hat

		# deformation gradient tensor :
		self.grad_u = sp.Matrix([[self.du_x_dx, self.du_x_dy, self.du_x_dz],
		                         [self.du_y_dx, self.du_y_dy, self.du_y_dz],
		                         [self.du_z_dx, self.du_z_dy, self.du_z_dz]])

		# "full-Stokes" :
		if    momentum.__class__ == MomentumStokes \
		   or momentum.__class__ == MomentumNitscheStokes \
		   or momentum.__class__ == MomentumDukowiczStokes:
			s   = "    - using full-Stokes formulation -"
			# strain-rate tensor :
			self.epsdot = 0.5 * (self.grad_u + self.grad_u.T)

		# first-order Blatter-Pattyn approximation :
		elif    momentum.__class__ == MomentumBP \
		     or momentum.__class__ == MomentumDukowiczBP:
			s   = "    - using Blatter-Pattyn formulation -"
			# strain-rate tensor :
			epi         = 0.5 * (self.grad_u + self.grad_u.T)
			epi02       = 0.5*self.du_x_dz
			epi12       = 0.5*self.du_y_dz
			epi22       = -self.du_x_dx - self.du_y_dy  # incompressibility
			self.epsdot = sp.Matrix([[epi[0,0],  epi[0,1],  epi02],
			                         [epi[1,0],  epi[1,1],  epi12],
			                         [epi02,     epi12,     epi22]])

		print_text(s, cls=self)

		# effective strain-rate squared :
		self.epsdot2      = self.epsdot.multiply(self.epsdot)
		self.epsdot_eff   = 0.5 * ( + self.epsdot2[0,0] \
		                            + self.epsdot2[1,1] \
		                            + self.epsdot2[2,2])

		# flow enhancement factor :
		def E(x,y,z):
			return 1.0

		# water content :
		def W(x,y,z):
			return 0.0

		# TODO : allow temperature to vary in space
		def a_T(x,y,z):
			if self.T < 263.15:
				return a_T_l
			else:
				return a_T_u

		def Q_T(x,y,z):
			if self.T < 263.15:
				return Q_T_l
			else:
				return Q_T_u

		def W_T(x,y,z):
			W_e = W(x,y,z)
			if W_e < 0.01:
				return W_e
			else:
				return 0.01

		# flow-rate factor :
		self.A = E(x,y,z) * a_T(x,y,z) * (1 + 181.25 * W_T(x,y,z)) \
		                               * sp.exp(-Q_T(x,y,z) / (R * self.T))

		# viscosity :
		self.eta = 0.5 * self.A**(-1/n) * (self.epsdot_eff + eps_reg)**((1-n)/(2*n))

		# gravity forcing term (rhs) :
		self.f   = sp.Matrix([0, 0, -rho_i*g])

		# deviatoric stress tensor :
		self.tau = 2.0 * self.eta * self.epsdot

		# deviatoric stress on upper surface :
		self.tau_s = self.tau.replace(z, self.s)

		# deviatoric stress on lower surface :
		self.tau_b = self.tau.replace(z, self.b)

		# elements of the deviatoric-stress divergence tensor :
		self.dtau_xx_dx = self.tau[0,0].diff(x,1)
		self.dtau_xy_dy = self.tau[0,1].diff(y,1)
		self.dtau_xz_dz = self.tau[0,2].diff(z,1)
		self.dtau_yx_dx = self.tau[1,0].diff(x,1)
		self.dtau_yy_dy = self.tau[1,1].diff(y,1)
		self.dtau_yz_dz = self.tau[1,2].diff(z,1)
		self.dtau_zx_dx = self.tau[2,0].diff(x,1)
		self.dtau_zy_dy = self.tau[2,1].diff(y,1)
		self.dtau_zz_dz = self.tau[2,2].diff(z,1)

		# deviatoric-stress divergence tensor :
		div_tau = [self.dtau_xx_dx + self.dtau_xy_dy + self.dtau_xz_dz,
		           self.dtau_yx_dx + self.dtau_yy_dy + self.dtau_yz_dz,
		           self.dtau_zx_dx + self.dtau_zy_dy + self.dtau_zz_dz]
		self.div_tau = sp.Matrix(div_tau)

		# pressure :
		#self.p = + rho_i * g * (self.s - z) \
		#         - 2.0 * self.eta * ( self.du_x_dx + self.du_y_dy )
		#self.p = + self.a_s * self.tau_s.multiply(self.n_s).dot(self.n_s) \
		#         + self.a_b * self.tau_b.multiply(self.n_b).dot(self.n_b)
		#self.p = + rho_i * g * (self.s - z)
		#self.p = sp.Rational(0.0)
		self.p = + rho_i * g * (self.s - z) + 0.5 * rho_i * self.u.dot(self.u)

		# pressure derivative in the x direction :
		self.dpdx      = self.p.diff(x,1)

		# pressure derivative in the y direction :
		self.dpdy      = self.p.diff(y,1)

		# pressure derivative in the z direction :
		self.dpdz      = self.p.diff(z,1)

		# pressure gradient :
		self.grad_p    = sp.Matrix([self.dpdx, self.dpdy, self.dpdz])

		# stress divergence tensor :
		self.div_sigma = self.div_tau - self.grad_p

		# cauchy stress tensor :
		self.sigma     = self.tau - self.p*sp.eye(3)

		# stress tensor at upper surface :
		self.sigma_s   = self.sigma.replace(z, self.s)

		# stress tensor at lower surface :
		self.sigma_b   = self.sigma.replace(z, self.b)

		# traction on upper surface :
		self.sigma_dot_n_s = self.sigma_s.multiply(self.n_s)

		# traction on lower surface :
		self.sigma_dot_n_b = self.sigma_b.multiply(self.n_b)

		# traction on north surface :
		self.sigma_dot_n_N = self.sigma.multiply(self.n_N)

		# traction on south surface :
		self.sigma_dot_n_S = self.sigma.multiply(self.n_S)

		# traction on east surface :
		self.sigma_dot_n_E = self.sigma.multiply(self.n_E)

		# traction on west surface :
		self.sigma_dot_n_W = self.sigma.multiply(self.n_W)

		# velocity flux on lower surface :
		self.u_dot_n_b = self.u_b.dot(self.n_b)

		# tangental component of velocity at lower surface :
		self.ut        = self.u_b - self.u_dot_n_b * self.n_b

		# interior compensatory forcing rhs term :
		self.f_int     = - self.div_sigma - self.f

		# exterior compensatory forcing rhs term over lower surface :
		self.f_ext_b   = self.sigma_dot_n_b \
		                 + self.beta * self.ut \
		                 + self.p * self.n_b

		# exterior compensatory forcing rhs term over upper surface :
		self.f_ext_s   = self.sigma_dot_n_s

	def get_u(self):
		"""
		"""
		print_text("::: forming R^3 velocity vector :::", cls=self)
		u_x_e = ccode(self.u_x)
		u_y_e = ccode(self.u_y)
		u_z_e = ccode(self.u_z)
		return Expression((u_x_e, u_y_e, u_z_e), element=self.Ve)

	def get_div_u(self):
		"""
		"""
		print_text("::: forming R^3 velocity divergence :::", cls=self)
		return Expression(ccode(self.div_u), element=self.Qe)

	def get_p(self):
		"""
		"""
		print_text("::: forming p expression :::", cls=self)
		return Expression(ccode(self.p), element=self.Qe)

	def get_S(self):
		"""
		"""
		print_text("::: forming S expression :::", cls=self)
		return Expression(ccode(self.s), element=self.Qe)

	def get_B(self):
		"""
		"""
		print_text("::: forming B expression :::", cls=self)
		return Expression(ccode(self.b), element=self.Qe)

	def get_S_ring(self):
		"""
		"""
		print_text("::: forming S_ring expression :::", cls=self)
		return Expression(ccode(self.s_ring), element=self.Qe)

	def get_B_ring(self):
		"""
		"""
		print_text("::: forming B_ring expression :::", cls=self)
		return Expression(ccode(self.b_ring), element=self.Qe)

	def get_dSdt(self):
		"""
		"""
		print_text("::: forming dSdt expression :::", cls=self)
		return Expression(ccode(self.dsdt), element=self.Qe)

	def get_dBdt(self):
		"""
		"""
		print_text("::: forming dBdt expression :::", cls=self)
		return Expression(ccode(self.dbdt), element=self.Qe)

	def get_T(self):
		"""
		"""
		print_text("::: forming T expression :::", cls=self)
		return Expression(ccode(self.T), element=self.Qe)

	def get_beta(self):
		"""
		"""
		print_text("::: forming beta expression :::", cls=self)
		return Expression(ccode(self.beta), element=self.Qe)

	def get_u_dot_n_b(self, coef=1):
		"""
		"""
		print_text("::: forming u.n expression :::", cls=self)
		return Expression(ccode(coef*self.u_dot_n_b), element=self.Qe)

	def get_compensatory_interior_rhs(self):
		"""
		"""
		print_text("::: forming R^3 stress balance interior compensatory " + \
		           "forcing terms:::", cls=self)
		comp_x_e = ccode(self.f_int[0])
		comp_y_e = ccode(self.f_int[1])
		comp_z_e = ccode(self.f_int[2])
		return Expression((comp_x_e, comp_y_e, comp_z_e), element=self.Ve)

	def get_compensatory_upper_surface_exterior_rhs(self):
		"""
		"""
		print_text("::: forming R^3 stress balance exterior compensatory " + \
		           "forcing terms over upper surface :::", cls=self)
		comp_x_e = ccode(self.f_ext_s[0])
		comp_y_e = ccode(self.f_ext_s[1])
		comp_z_e = ccode(self.f_ext_s[2])
		return Expression((comp_x_e, comp_y_e, comp_z_e), element=self.Ve)

	def get_compensatory_lower_surface_exterior_rhs(self):
		"""
		"""
		print_text("::: forming R^3 stress balance exterior compensatory " + \
		           "forcing terms over lower surface :::", cls=self)
		comp_x_e = ccode(self.f_ext_b[0])
		comp_y_e = ccode(self.f_ext_b[1])
		comp_z_e = ccode(self.f_ext_b[2])
		return Expression((comp_x_e, comp_y_e, comp_z_e), element=self.Ve)

	def get_stress_tensor(self):
		"""
		"""
		print_text("::: forming R^3 stress tensor :::", cls=self)
		sig_xx = Expression(ccode(self.sigma[0,0]), element=self.Qe)
		sig_xy = Expression(ccode(self.sigma[0,1]), element=self.Qe)
		sig_xz = Expression(ccode(self.sigma[0,2]), element=self.Qe)
		sig_yx = Expression(ccode(self.sigma[1,0]), element=self.Qe)
		sig_yy = Expression(ccode(self.sigma[1,1]), element=self.Qe)
		sig_yz = Expression(ccode(self.sigma[1,2]), element=self.Qe)
		sig_zx = Expression(ccode(self.sigma[2,0]), element=self.Qe)
		sig_zy = Expression(ccode(self.sigma[2,1]), element=self.Qe)
		sig_zz = Expression(ccode(self.sigma[2,2]), element=self.Qe)
		return as_matrix([[sig_xx, sig_xy, sig_xz],
		                  [sig_yx, sig_yy, sig_yz],
		                  [sig_zx, sig_zy, sig_zz]])

	def init_r3_membrane_stress_balance(self, momentum):
		"""
		"""
		# signed angle in radians of the horizontal velocity from the x axis :
		self.xy_velocity_angle = sp.atan2(self.u_x, self.u_y)

		# rotate about the z-axis :
		def z_rotation_matrix(self):
			c     = sp.cos(self.xy_velocity_angle)
			s     = sp.sin(self.xy_velocity_angle)
			Rz    = sp.Matrix([[c, -s, 0],
			                   [s,  c, 0],
			                   [0,  0, 1]])
			return Rz

		def rotate_tensor(self, M, R):
			if M.rank() == 3:
				Mr = R.multiply(M.multiply(R.T))
			elif M.rank() == 2:
				Mr = R.multipy(M)
			else:
				s   = ">>> METHOD 'rotate_tensor' REQUIRES RANK 2 OR 1 TENSOR <<<"
				print_text(s, 'red', 1)
				sys.exit(1)
			return Mr

		# directional derivative in direction of flow :
		def d_di(self, u):  return dot(grad(u), U_n)

		# directional derivative in direction across flow :
		def d_dj(self, u):  return dot(grad(u), U_t)

		def membrane_stress_tensor(self):
			R        = self.z_rotation_matrix()
			tau_r    = self.rotate_tensor(self.tau, R)
			#tau_r_s  = rotate_tensor(tau(x,y,z=self.s), R)
			#tau_r_b  = rotate_tensor(tau(x,y,z=self.b), R)

			## get upper-surface deviatoric stress :
			#tau_ii_s = tau_r_s[0,0]
			#tau_ij_s = tau_r_s[0,1]
			#tau_iz_s = tau_r_s[0,2]
			#tau_ji_s = tau_r_s[1,0]
			#tau_jj_s = tau_r_s[1,1]
			#tau_jz_s = tau_r_s[1,2]
			#tau_zi_s = tau_r_s[2,0]
			#tau_zj_s = tau_r_s[2,1]
			#tau_zz_s = tau_r_s[2,2]

			## get lower-surface deviatoric stress :
			#tau_ii_b = tau_r_b[0,0]
			#tau_ij_b = tau_r_b[0,1]
			#tau_iz_b = tau_r_b[0,2]
			#tau_ji_b = tau_r_b[1,0]
			#tau_jj_b = tau_r_b[1,1]
			#tau_jz_b = tau_r_b[1,2]
			#tau_zi_b = tau_r_b[2,0]
			#tau_zj_b = tau_r_b[2,1]
			#tau_zz_b = tau_r_b[2,2]

			# vertically integrate deviatoric stress (membrane stress) :
			N_ii = sp.integrate(tau_r[0,0], (self.z, self.b, self.s))
			N_ij = sp.integrate(tau_r[0,1], (self.z, self.b, self.s))
			N_iz = sp.integrate(tau_r[0,2], (self.z, self.b, self.s))
			N_ji = sp.integrate(tau_r[1,0], (self.z, self.b, self.s))
			N_jj = sp.integrate(tau_r[1,1], (self.z, self.b, self.s))
			N_jz = sp.integrate(tau_r[1,2], (self.z, self.b, self.s))
			N_zi = sp.integrate(tau_r[2,0], (self.z, self.b, self.s))
			N_zj = sp.integrate(tau_r[2,1], (self.z, self.b, self.s))
			N_zz = sp.integrate(tau_r[2,2], (self.z, self.b, self.s))

			N    = sp.Matrix([[N_ii, N_ij, N_iz],
			                  [N_ji, N_jj, N_jz],
			                  [N_zi, N_zj, N_zz]])
			return N

	def color(self):
		"""
		return the default color for this class.
		"""
		return '199'

	# verify the solution before proceeding :
	def verify_analytic_solution(self, nx, ny, Lx, Ly):
		"""
		"""
		s = "::: verifying analytic solution :::"
		print_text(s, cls=self)

		# create a genreic box mesh, we'll fit it to geometry below :
		x_a   = np.linspace(0, Lx, nx, dtype='float128')
		y_a   = np.linspace(0, Ly, ny, dtype='float128')
		X,Y   = np.meshgrid(x_a, y_a)

		# form the 2D vertically-integrated relations :
		self.init_r2_stress_balance()

		# vertically integrated velocity divergence evaluated numerically :
		int_div_u = lambdify((self.x, self.y), self.int_div_u,     "numpy")
		lei_resid = lambdify((self.x, self.y), self.leibniz_resid, "numpy")

		print_min_max(int_div_u(X,Y), 'int_z div(u)           ')
		print_min_max(lei_resid(X,Y), 'dHdt + div(Hu) - ring_H')



