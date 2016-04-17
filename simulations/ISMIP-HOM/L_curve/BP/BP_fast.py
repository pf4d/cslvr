from varglas          import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *

#set_log_active(False)

out_dir = './results_BP/'

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
model.init_b(model.A0(0)**(-1/model.n(0)))

nparams = {'newton_solver' : {'linear_solver'            : 'cg',
                              'preconditioner'           : 'hypre_amg',
                              'relative_tolerance'       : 1e-8,
                              'relaxation_parameter'     : 1.0,
                              'maximum_iterations'       : 25,
                              'error_on_nonconvergence'  : False}}
m_params  = {'solver'               : nparams,
             'solve_vert_velocity'  : True,
             'solve_pressure'       : True,
             'vert_solve_method'    : 'mumps'}

mom = MomentumBP(model, m_params, isothermal=True)
mom.solve(annotate=False)

u     = Function(model.Q)
v     = Function(model.Q)
assign(u, model.U3.sub(0))
assign(v, model.U3.sub(1))
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

model.init_beta(30.0**2)
#model.init_beta_SIA()
#model.save_pvd(model.beta, 'beta_SIA')

mom = MomentumBP(model, m_params, linear=True, isothermal=True)
mom.solve(annotate=True)

model.set_out_dir(out_dir = out_dir + 'inverted/')
  
J = mom.form_obj_ftn(integral=model.dSrf, kind='log_lin_hybrid', 
                     g1=0.01, g2=1000)
R = mom.form_reg_ftn(model.beta, integral=model.dBed, kind='Tikhonov', 
                     alpha=10000.0)
I = J# + R

controls = File(out_dir + "beta_control.pvd")
beta_viz = Function(model.Q, name="beta_control")
  
def eval_cb(I, beta):
  #       commented out because the model variables are not updated by 
  #       dolfin-adjoint (yet) :
  #mom.print_eval_ftns()
  #print_min_max(mom.U, 'U')
  print_min_max(I,    'I')
  print_min_max(beta, 'beta')

def deriv_cb(I, dI, beta):
  print_min_max(I,     'I')
  print_min_max(dI,    'dI/dbeta')
  print_min_max(beta,  'beta')
  beta_viz.assign(beta)
  controls << beta_viz

def hessian_cb(I, ddI, beta):
  print_min_max(ddI, 'd/db dI/db')

m = FunctionControl('beta')
F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                      derivative_cb_post = deriv_cb,
                      hessian_cb = hessian_cb)

#m_opt = minimize(F, method="L-BFGS-B", tol=2e-8, bounds=(0, 4000),
#                 options={"disp"    : True,
#                          "maxiter" : 200})
  
problem = MinimizationProblem(F, bounds=(0, 4000))
parameters = {"acceptable_tol"     : 1.0e-200,
              "maximum_iterations" : 200,
              "linear_solver"      : "ma97"}

solver = IPOPTSolver(problem, parameters=parameters)
b_opt = solver.solve()
print_min_max(b_opt, 'b_opt')

model.assign_variable(model.beta, b_opt)
mom.solve(annotate=False)
mom.print_eval_ftns()

model.save_pvd(model.beta, 'beta_opt')
model.save_pvd(model.U3,   'U_opt')



