from varglas        import D3Model, MomentumDukowiczStokesReduced, \
                           print_text, print_min_max
from varglas.energy import Enthalpy
from scipy          import random
from fenics         import *
from dolfin_adjoint import *

#set_log_active(False)

out_dir = './dukowicz_reduced/'

alpha = 0.1 * pi / 180
L     = 40000

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 25, 25, 10)

model = D3Model(out_dir = out_dir + 'initial/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('1000 - 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(beta)
model.init_T_surface(268.0)
model.init_T(268.0)
model.init_q_geo(model.ghf)
model.init_E(1.0)

mom = MomentumDukowiczStokesReduced(model, isothermal=False)
nrg = Enthalpy(model)

def cb_ftn():
  nrg.solve_basal_melt_rate()
  nrg.calc_bulk_density()
  model.save_pvd(model.U3,    'U3')
  model.save_pvd(model.p,     'p')
  model.save_pvd(model.theta, 'theta')
  model.save_pvd(model.T,     'T')
  model.save_pvd(model.W,     'W')
  model.save_pvd(model.Mb,    'Mb')
  model.save_pvd(model.rho_b, 'rho_b')

model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=1e-6, max_iter=15)

u,v,w = model.U3.split(True)
u_o   = u.vector().array()
v_o   = v.vector().array()
n     = len(u_o)
#U_e   = model.get_norm(as_vector([model.u, model.v]), 'linf')[1] / 500
U_e   = 0.18
print_min_max(U_e, 'U_e')
  
u_error = U_e * random.randn(n)
v_error = U_e * random.randn(n)
u_ob    = u_o + u_error
v_ob    = v_o + v_error

model.init_U_ob(u_ob, v_ob)

model.save_pvd(model.U3,   'U_true')
model.save_pvd(model.U_ob, 'U_ob')
model.save_pvd(model.beta, 'beta_true')

#model.init_beta(30.0)
model.init_beta_SIA()
model.save_pvd(model.beta, 'beta_SIA')

mom = MomentumDukowiczStokesReduced(model, linear=True, isothermal=False)
mom.solve_params['solver']['newton_solver']['relaxation_parameter']    = 1.0
mom.solve_params['solver']['newton_solver']['maximum_iterations']      = 10
mom.solve(annotate=True)

model.set_out_dir(out_dir = out_dir + 'inverted/')
  
J = mom.form_obj_ftn('log_lin_hybrid', integral=model.GAMMA_S_GND,
                     g1=0.01, g2=1000)
R = mom.form_reg_ftn(model.beta, 'Tikhonov', integral=model.GAMMA_B_GND,
                     alpha=10000.0)
I = J + R

controls = File(out_dir + "beta_control.pvd")
beta_viz = Function(model.Q, name="beta_control")
  
def eval_cb(j, m):
  #mom.print_eval_ftns()
  #print_min_max(mom.U, 'U')
  print_min_max(j, 'I')

def deriv_cb(j, dj, m):
  print_min_max(dj, 'dJdb')
  print_min_max(m,  'beta')
  beta_viz.assign(m)
  controls << beta_viz

def hessian_cb(j, ddj, m):
  print_min_max(ddj, 'd/db dJ/db')

m = FunctionControl('beta')
F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                      derivative_cb_post = deriv_cb,
                      hessian_cb = hessian_cb)
  
problem = MinimizationProblem(F, bounds=(0, 4000))
parameters = {"acceptable_tol"     : 1.0e-200,
              "maximum_iterations" : 1000,
              "linear_solver"      : "ma97"}

solver = IPOPTSolver(problem, parameters=parameters)
b_opt = solver.solve()

#m_opt = minimize(F, method="L-BFGS-B", tol=2e-8, bounds=(10, 100),
#                 options={"disp"    : True,
#                          "maxiter" : 100})

model.save_pvd(b_opt, 'b_opt')

#A = DolfinAdjointSolver(model, config)
#A.solve()



