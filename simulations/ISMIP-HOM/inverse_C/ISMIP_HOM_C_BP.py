from varglas.model              import Model
from varglas.solvers            import SteadySolver, AdjointSolver
from varglas.physical_constants import IceParameters
from varglas.helper             import default_nonlin_solver_params
from scipy                      import random
from fenics                     import set_log_active, File, Expression, pi, \
                                       sin, tan, parameters, project

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
Beta2   = Expression(  '1000 + 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.set_geometry(Surface, Bed, deform=True)
model.set_parameters(IceParameters())
model.initialize_variables()

nparams = default_nonlin_solver_params()
nparams['newton_solver']['linear_solver']           = 'mumps'
nparams['newton_solver']['preconditioner']          = 'default'
nparams['newton_solver']['relaxation_parameter']    = 0.8
nparams['newton_solver']['maximum_iterations']      = 20
nparams['newton_solver']['relative_tolerance']      = 1e-5
nparams['newton_solver']['error_on_nonconvergence'] = False
parameters['form_compiler']['quadrature_degree']    = 2

config = { 'mode'                         : 'steady',
           't_start'                      : None,
           't_end'                        : None,
           'time_step'                    : None,
           'output_path'                  : './results/initial/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : True,
           'log': True,
           'coupled' : 
           { 
             'on'                  : False,
             'inner_tol'           : 0.0,
             'max_iter'            : 1
           },
           'velocity' : 
           { 
             'on'                  : True,
             'newton_params'       : nparams,
             'viscosity_mode'      : 'isothermal',
             'b_linear'            : None,
             'use_T0'              : False,
             'T0'                  : None,
             'A0'                  : 1e-16,
             'beta2'               : Beta2,
             'r'                   : 0.0,
             'E'                   : 1,
             'approximation'       : 'fo',
             'boundaries'          : None
           },
           'enthalpy' : 
           { 
             'on'                  : False,
             'use_surface_climate' : False,
             'T_surface'           : None,
               
           },
           'free_surface' :
           { 
             'on'                  : False,
             'thklim'              : None,
             'use_pdd'             : False,
             'observed_smb'        : None,
           },  
           'age' : 
           { 
             'on'                  : False,
             'use_smb_for_ela'     : False,
             'ela'                 : None,
           },
           'surface_climate' : 
           { 
             'on'                  : False,
             'T_ma'                : None,
             'T_ju'                : None,
             'beta_w'              : None,
             'sigma'               : None,
             'precip'              : None
           },
           'adjoint' :
           { 
             'alpha'               : 0.0,
             'max_fun'             : 20,
             'objective_function'  : 'linear',
             'animate'             : False,
             'bounds'              : (0.0, 5000.0),
             'control_variable'    : model.beta2,
             'regularization_type' : 'Tikhonov'
           }}

model.eps_reg = 1e-5

F = SteadySolver(model,config)
F.solve()

File('results/beta2_obs.pvd') << model.beta2

u_o = project(model.u, model.Q).vector().array()
v_o = project(model.v, model.Q).vector().array()
U_e = 10.0

model.print_min_max(u_o, 'u_o')
model.print_min_max(v_o, 'v_o')

config['output_path'] = './results/00/'

A = AdjointSolver(model,config)

for i in range(1):
  config['output_path']   = './results/%02i/' % i
  model.assign_variable(model.beta2, 1000.0)
  u_error = U_e*random.randn(len(u_o))
  v_error = U_e*random.randn(len(v_o))
  u_obs   = u_o + u_error
  v_obs   = v_o + v_error
  A.set_target_velocity(u=u_obs, v=v_obs)
  A.solve()



