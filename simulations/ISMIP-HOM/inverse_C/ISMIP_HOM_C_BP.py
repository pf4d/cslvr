from varglas.model              import Model
from varglas.solvers            import SteadySolver, AdjointSolver
from varglas.physical_constants import IceParameters
from varglas.helper             import default_nonlin_solver_params, \
                                       default_config
from scipy                      import random
from fenics                     import *

set_log_active(True)

alpha = 0.1 * pi / 180
L     = 80000
nx    = 50
ny    = 50 
nz    = 10

model = Model()
model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, ymin=0, ymax=L, 
                            generate_pbcs=True)

Surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
Bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
Beta    = Expression('sqrt(1000 + 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L))',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.set_geometry(Surface, Bed, deform=True)
model.set_parameters(IceParameters())
model.calculate_boundaries()
model.initialize_variables()

nparams = default_nonlin_solver_params()
nparams['newton_solver']['linear_solver']           = 'mumps'
nparams['newton_solver']['preconditioner']          = 'default'
nparams['newton_solver']['relaxation_parameter']    = 0.8
nparams['newton_solver']['maximum_iterations']      = 16
nparams['newton_solver']['relative_tolerance']      = 1e-5
nparams['newton_solver']['error_on_nonconvergence'] = False
parameters['form_compiler']['quadrature_degree']    = 2

config = default_config()
config['output_path']                   = './results/initial/'
config['periodic_boundary_conditions']  = True
config['velocity']['newton_params']     = nparams
config['velocity']['beta0']             = Beta
config['adjoint']['objective_function'] = 'linear'
config['adjoint']['bounds']             = (0.0, 500.0)
config['adjoint']['control_variable']   = model.beta

F = SteadySolver(model,config)
F.solve()

File('results/beta_true.pvd') << model.beta

u_o = model.u.vector().array()
v_o = model.v.vector().array()
U_e = 5.0

config['output_path'] = './results/00/'
#model.eps_reg = 1e-5

A = AdjointSolver(model,config)

for i in range(1):
  u_error = U_e*random.randn(len(u_o))
  v_error = U_e*random.randn(len(v_o))
  u_obs   = u_o + u_error
  v_obs   = v_o + v_error
  u_obs_f = Function(model.Q)
  v_obs_f = Function(model.Q)
  model.assign_variable(u_obs_f, u_obs)
  model.assign_variable(v_obs_f, v_obs)
  U_ob    = project(sqrt(u_obs_f**2 + v_obs_f**2), model.Q)
  config['output_path']   = './results/%02i/' % i
  config['velocity']['init_beta_from_U_ob'] = True
  config['velocity']['U_ob']                = U_ob
  #model.assign_variable(model.beta, sqrt(1000.0))
  A.set_target_velocity(u=u_obs, v=v_obs)
  A.solve()



