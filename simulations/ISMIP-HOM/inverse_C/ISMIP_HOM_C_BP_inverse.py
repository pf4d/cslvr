from varglas.BPModel import BPModel
#from varglas.solvers import DolfinAdjointSolver
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_text, print_min_max
from scipy           import random
from fenics          import *
from dolfin_adjoint  import *

#set_log_active(False)

out_dir = './results_new/'

alpha = 0.1 * pi / 180
L     = 40000

nparams = NonlinearVariationalSolver.default_parameters()
#nparams = default_nonlin_solver_params()
#nparams['newton_solver']['linear_solver']           = 'cg'
#nparams['newton_solver']['preconditioner']          = 'hypre_amg'
#nparams['newton_solver']['relative_tolerance']      = 1e-8
#nparams['newton_solver']['relaxation_parameter']    = 1.0
#nparams['newton_solver']['maximum_iterations']      = 25
#nparams['newton_solver']['error_on_nonconvergence'] = False
parameters['form_compiler']['quadrature_degree']    = 2

config = default_config()
config['log_history']                     = False
config['output_path']                     = out_dir + 'initial/'
config['periodic_boundary_conditions']    = True
config['velocity']['newton_params']       = nparams
config['velocity']['vert_solve_method']   = 'mumps'
config['velocity']['solve_pressure']      = True

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 25, 25, 10)

model = BPModel(config)
model.set_mesh(mesh)

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
#model.init_viscosity_mode('isothermal')
model.init_viscosity_mode('full')

model.init_momentum()
#model.init_energy()
#model.init_age()

model.solve_momentum(annotate=False)
model.U.interpolate(Constant((0,0)), annotate=False)
model.solve_momentum()
#model.solve_vert_velocity()
#model.solve_energy()
#model.solve_age()
#model.solve_pressure()

#model.thermo_solve(rtol=1e-10, max_iter=2)

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

config['output_path']  = out_dir + 'inverted/'

#nparams['newton_solver']['relaxation_parameter']    = 1.0
#nparams['newton_solver']['maximum_iterations']      = 10
  
#model.init_viscosity_mode('linear')
model.init_beta(30.0)

J = model.get_obj_ftn('log_lin_hybrid', integral=model.GAMMA_S_GND,
                      g1=0.01, g2=1000)
R = model.get_reg_ftn(model.beta, 'Tikhonov', integral=model.GAMMA_B_GND,
                      alpha=1.0)
I = J*dt[FINISH_TIME] + R*dt[START_TIME]

#L = model.Lagrangian()
#
#H = model.Hamiltonian(I)
# 
#dHdc = model.dHdc(I, L, model.beta)
# 
#model.solve_adjoint_momentum(H)
#
#model.save_pvd(model.Lam,  'Lam')

F = ReducedFunctional(Functional(I),
                      Control(model.beta, value=model.beta))

m_opt = minimize(F, method="L-BFGS-B", tol=2e-8, bounds=(10, 100),
                 options={"disp"    : True})#,
#                          "maxiter" : 25})

model.save_pvd(model.beta, 'beta')
model.save_pvd(model.U3,   'U')

#A = DolfinAdjointSolver(model, config)
#A.solve()



