from varglas          import D3Model, MomentumDukowiczStokes, print_text, \
                             print_min_max
from varglas.energy   import Enthalpy 
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *

#set_log_active(False)

out_dir = './fast/'

#hs = [1000, 2000, 4000, 8000, 16000, 32000]
#Hs = [250,  500,  750,  1000, 2000,  3000]
#Gs = [0.1,  0.25, 0.5,  1,    2,     4]
#
#for h in hs:
#  for H in Hs:
#    for g in Gs:
#      pass

n     = 25
h     = 1000.0
g     = 0.5

H     = 1000.0
L     = n*h
alpha = g * pi / 180

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 25, 25, 10)

model = D3Model(out_dir = out_dir + 'initial/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - H', alpha=alpha, H=H, 
                     element=model.Q.ufl_element())
beta    = Expression('H - H * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)', H=H,
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(beta)
model.init_b(model.A0(0)**(-1/model.n(0)))

nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
                              'relative_tolerance'       : 1e-8,
                              'relaxation_parameter'     : 1.0,
                              'maximum_iterations'       : 25,
                              'error_on_nonconvergence'  : False}}
m_params  = {'solver'      : nparams}

mom = MomentumDukowiczStokes(model, m_params, isothermal=True)
mom.solve(annotate=False)

u,v,w = model.U3.split(True)
u_s   = model.vert_extrude(u, d='down')
v_s   = model.vert_extrude(v, d='down')
sigma = 100.0
U_mag = model.get_norm(as_vector([u_s, v_s]), 'linf')[1]
n     = len(U_mag)
U_avg = sum(U_mag) / n
U_e   = U_avg / sigma
print_min_max(U_e, 'U_error')

u_o     = u.vector().array()
v_o     = v.vector().array()
n       = len(u_o)
u_error = U_e * random.randn(n)
v_error = U_e * random.randn(n)
u_ob    = u_o + u_error
v_ob    = v_o + v_error

model.assign_variable(u, u_ob)
model.assign_variable(v, v_ob)


print_min_max(u_error, 'u_error')
print_min_max(v_error, 'v_error')


model.init_U_ob(u, v)

model.save_pvd(model.U3,   'U_true')
model.save_pvd(model.U_ob, 'U_ob')
model.save_pvd(model.beta, 'beta_true')

model.init_beta(30.0**2)
#model.init_beta_SIA()
#model.save_pvd(model.beta, 'beta_SIA')

mom = MomentumDukowiczStokes(model, m_params, linear=True, isothermal=True)
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



