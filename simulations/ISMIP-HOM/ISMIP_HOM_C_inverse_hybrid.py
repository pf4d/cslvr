from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *

#set_log_active(False)

out_dir = './ISMIP_HOM_C_inverse_hybrid_results/'

alpha = 0.1 * pi / 180
L     = 40000

p1    = Point(0.0, 0.0)
p2    = Point(L,   L)
mesh  = RectangleMesh(p1, p2, 25, 25)

model = HybridModel(mesh, out_dir = out_dir + 'initial/', use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha,
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha,
                     element=model.Q.ufl_element())
beta    = Expression('1000 - 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)                           # all grounded ice
model.init_beta(beta)                          # friction
model.init_E(1.0)                              # no enhancement factor
model.init_A(1e-16)

mom = MomentumHybrid(model)
mom.solve(annotate=False)

u     = Function(model.Q)
v     = Function(model.Q)
assign(u, model.u.sub(0))
assign(v, model.u.sub(1))
u_o   = u.vector().array()
v_o   = v.vector().array()
n     = len(u_o)
#U_e   = model.get_norm(as_vector([model.u, model.v]), 'linf') / 500
U_e   = 0.18
print_min_max(U_e, 'U_e')

u_error = U_e * random.randn(n)
v_error = U_e * random.randn(n)
u_ob    = u_o + u_error
v_ob    = v_o + v_error

model.init_U_ob(u_ob, v_ob)

model.save_xdmf(model.u,   'U_true')
model.save_xdmf(model.U_ob, 'U_ob')
model.save_xdmf(model.beta, 'beta_true')

model.init_beta(30.0**2)
#model.init_beta_SIA()
#model.save_xdmf(model.beta, 'beta_SIA')

model.set_out_dir(out_dir + 'inversion/')

def deriv_cb(I, dI, beta):
	model.save_xdmf(model.beta, 'beta_control')

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
	mom.solve_params['solve_vert_velocity'] = True
	mom.solve(annotate=False)

	# save the optimal velocity and beta fields for viewing with paraview :
	model.save_xdmf(model.u,   'U_opt')
	model.save_xdmf(model.beta, 'beta_opt')

# after every completed adjoining, save the state of these functions :
adj_save_vars = [model.beta, model.u]

# form the cost functional :
mom.form_obj_ftn(integral=model.OMEGA_GND, kind='log_L2_hybrid',
                 g1=0.01, g2=5000)

# form the regularization functional :
mom.form_reg_ftn(model.beta, integral=model.OMEGA_GND, kind='TV',
                 alpha=1.0)

# solving the incomplete adjoint is more efficient :
mom.linearize_viscosity()

# optimize for beta :
mom.optimize_U_ob(control           = model.beta,
                  bounds            = (1e-5, 1e7),
                  method            = 'ipopt',
                  max_iter          = 20,
                  adj_save_vars     = adj_save_vars,
                  adj_callback      = deriv_cb,
                  post_adj_callback = adj_post_cb_ftn)



