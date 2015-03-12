from varglas.model   import Model
from varglas.solvers import SteadySolver, AdjointSolver
from varglas.helper  import default_nonlin_solver_params, default_config
from fenics          import *
from scipy           import random

#set_log_active(False)

alpha = 0.1 * pi / 180
L     = 40000

nparams = default_nonlin_solver_params()
nparams['newton_solver']['linear_solver']           = 'cg'
nparams['newton_solver']['preconditioner']          = 'hypre_amg'
nparams['newton_solver']['relative_tolerance']      = 1e-10
nparams['newton_solver']['relaxation_parameter']    = 1.0
nparams['newton_solver']['maximum_iterations']      = 20
nparams['newton_solver']['error_on_nonconvergence'] = False
parameters['form_compiler']['quadrature_degree']    = 2

config = default_config()
config['output_path']                     = './results/initial/'
config['periodic_boundary_conditions']    = True
config['velocity']['newton_params']       = nparams

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 30, 30, 10)

model = Model(config)
model.set_mesh(mesh)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('sqrt(1000 + 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L))',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.set_geometry(surface, bed, deform=True)
model.initialize_variables()

model.init_beta(beta)

F = SteadySolver(model, config)
F.solve()

model.save_pvd(model.beta, 'beta_true')

u_o = model.u.vector().array()
v_o = model.v.vector().array()
U_e = 0.5
n   = len(u_o)

model.init_beta(30.0)

config['adjoint']['objective_function']   = 'log'
config['adjoint']['bounds']               = (1.0, 100.0)
config['adjoint']['control_variable']     = model.beta
config['adjoint']['alpha']                = 0.0
config['adjoint']['max_fun']              = 200
#config['velocity']['viscosity_mode']      = 'linear'
#config['velocity']['eta_gnd']             = model.eta_gnd
#config['velocity']['eta_shf']             = model.eta_shf

nparams['newton_solver']['relative_tolerance']      = 1e-3
nparams['newton_solver']['relaxation_parameter']    = 0.7
nparams['newton_solver']['maximum_iterations']      = 20

for i in range(1):
  u_error = U_e * random.randn(n)
  v_error = U_e * random.randn(n)
  u_ob    = u_o + u_error
  v_ob    = v_o + v_error
  
  config['output_path'] = './results/%02i/' % i
  
  model.init_U_ob(u_ob, v_ob)
  model.save_pvd(model.U_ob, 'U_ob')
  
  A = AdjointSolver(model, config)
  A.solve()



