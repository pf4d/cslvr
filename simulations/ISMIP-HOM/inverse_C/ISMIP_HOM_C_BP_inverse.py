from varglas.D3Model  import D3Model
from varglas.momentum import MomentumBP
from varglas.energy   import Enthalpy 
from varglas.io       import print_text, print_min_max
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *

#set_log_active(False)

out_dir = './results_new/'

alpha = 0.1 * pi / 180
L     = 40000

parameters['form_compiler']['quadrature_degree']  = 2
parameters["std_out_all_processes"]               = False

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 25, 25, 10)

model = D3Model(out_dir = out_dir + 'initial/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('sqrt(1000 - 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L))',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)
model.initialize_variables()

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(beta)
model.init_T_surface(268.0)
model.init_T(268.0)
model.init_q_geo(model.ghf)
model.init_E(1.0)
model.init_full_b()

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

e_params  = {'solver'               : 'mumps',
             'use_surface_climate'  : False}

mom = MomentumBP(model, m_params)
nrg = Enthalpy(model, e_params)

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

model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=1e-6, max_iter=3)

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

nparams['newton_solver']['relaxation_parameter']    = 1.0
nparams['newton_solver']['maximum_iterations']      = 10

model.init_beta(30.0)

mom = MomentumBP(model, m_params, linear=True)
mom.solve(annotate=True)

model.set_out_dir(out_dir = out_dir + 'inverted/')
  
J = mom.get_obj_ftn('log_lin_hybrid', integral=model.GAMMA_S_GND,
                    g1=0.01, g2=1000)
R = mom.get_reg_ftn(model.beta, 'Tikhonov', integral=model.GAMMA_B_GND,
                    alpha=1.0)
I = J + R

L = mom.Lagrangian()

H = mom.Hamiltonian(I)

dHdc = mom.dHdc(I, L, model.beta)

mom.solve_adjoint_momentum(H)

model.save_pvd(mom.Lam, 'Lam')
  
controls = File(out_dir + "control_iterations.pvd")
a_viz = Function(model.Q, name="ControlVisualisation")
def eval_cb(j, a):
  a_viz.assign(a)
  controls << a_viz


m = Control(model.beta)
F = ReducedFunctional(Functional(I), m, eval_cb=eval_cb)
  
problem = MinimizationProblem(F, bounds=(10, 100))
parameters = {"acceptable_tol"     : 1.0e-200,
              "maximum_iterations" : 10,
              "linear_solver"      : "ma97"}

solver = IPOPTSolver(problem, parameters=parameters)
b_opt = solver.solve()

#m_opt = minimize(F, method="L-BFGS-B", tol=2e-8, bounds=(10, 100),
#                 options={"disp"    : True,
#                          "maxiter" : 100})

model.save_pvd(model.beta, 'beta')
model.save_pvd(b_opt,      'b_opt')
model.save_pvd(mom.U,      'U')

#A = DolfinAdjointSolver(model, config)
#A.solve()



