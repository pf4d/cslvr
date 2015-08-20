from varglas.model   import Model
from varglas.solvers import SteadySolver, AdjointSolver
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_text, print_min_max
from fenics          import *
from scipy           import random

#set_log_active(False)

alpha = 0.1 * pi / 180
L     = 40000

nparams = default_nonlin_solver_params()
#nparams['newton_solver']['linear_solver']           = 'mumps'
nparams['newton_solver']['linear_solver']           = 'cg'
nparams['newton_solver']['preconditioner']          = 'hypre_amg'
nparams['newton_solver']['relative_tolerance']      = 1e-8
nparams['newton_solver']['relaxation_parameter']    = 1.0
nparams['newton_solver']['maximum_iterations']      = 25
nparams['newton_solver']['error_on_nonconvergence'] = False
parameters['form_compiler']['quadrature_degree']    = 2

config = default_config()
config['log_history']                     = False
config['output_path']                     = './results_new/initial/'
config['model_order']                     = 'BP'
config['use_dukowicz']                    = False
config['periodic_boundary_conditions']    = True
config['velocity']['newton_params']       = nparams
config['velocity']['vert_solve_method']   = 'mumps'

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 25, 25, 10)

model = Model(config)
model.set_mesh(mesh)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('sqrt(1000 - 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L))',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.set_geometry(surface, bed, deform=True)
model.initialize_variables()

model.init_viscosity_mode('isothermal')
model.init_beta(beta)

model.save_pvd(model.beta, 'beta_true')

F = SteadySolver(model, config)
F.solve()

u_o = model.u.vector().array()
v_o = model.v.vector().array()
#U_e = model.get_norm(as_vector([model.u, model.v]), 'linf')[1] / 500
U_e = 0.18
print_min_max(U_e, 'U_e')
n   = len(u_o)
  
u_error = U_e * random.randn(n)
v_error = U_e * random.randn(n)
u_ob    = u_o + u_error
v_ob    = v_o + v_error

model.init_U_ob(u_ob, v_ob)
model.save_pvd(model.beta, 'beta')
model.save_pvd(model.U_ob, 'U_ob')

config['output_path']                     = './results_new/inverted/'
config['velocity']['solve_vert_velocity'] = False
config['adjoint']['objective_function']   = 'log_lin_hybrid'
config['adjoint']['gamma1']               = 0.01
config['adjoint']['gamma2']               = 1000
config['adjoint']['bounds']               = (10, 100.0)
config['adjoint']['control_variable']     = model.beta
config['adjoint']['alpha']                = 1.0
config['adjoint']['max_fun']              = 100
config['adjoint']['control_domain']       = 'bed'
config['velocity']['solve_vert_velocity'] = False

nparams['newton_solver']['relaxation_parameter']    = 1.0
nparams['newton_solver']['maximum_iterations']      = 10
  
model.init_viscosity_mode('linear')
model.init_beta(30.0)

A = AdjointSolver(model, config)
A.solve()
