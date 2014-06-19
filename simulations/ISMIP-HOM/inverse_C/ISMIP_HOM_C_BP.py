import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.model              import Model
from src.solvers            import SteadySolver, AdjointSolver
from src.physical_constants import IceParameters
from src.helper             import default_nonlin_solver_params
from dolfin                 import set_log_active, pi, Expression, tan, sin, \
                                   pi, File, as_vector
from scipy                  import random

set_log_active(True)

alpha = 0.1 * pi / 180
L=80000

nx = 50
ny = 50 
nz = 10


model = Model()
model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, ymin=0, ymax=L, 
                            deform=True, generate_pbcs=True)

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
nparams['newton_solver']['linear_solver']           = 'gmres'
nparams['newton_solver']['preconditioner']          = 'hypre_amg'
#nparams['newton_solver']['linear_solver']           = 'mumps'
nparams['newton_solver']['relaxation_parameter']    = 0.7
nparams['newton_solver']['maximum_iterations']      = 20
nparams['newton_solver']['error_on_nonconvergence'] = False

config = { 'mode'                         : 'steady',
           't_start'                      : None,
           't_end'                        : None,
           'time_step'                    : None,
           'output_path'                  : './results/',
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
             'alpha'               : [0.0],
             'beta'                : 0.0,
             'max_fun'             : 20,
             'objective_function'  : 'linear',
             'animate'             : False,
             'bounds'              : None,
             'control_variable'    : None,
             'regularization_type' : 'Tikhonov'
           }}

F = SteadySolver(model,config)
F.solve()

model.eps_reg = 1e-5
config['adjoint']['control_variable'] = [model.beta2]
config['adjoint']['bounds']           = [(0.0,5000.0)]
File('results/beta2_obs.xml') << model.beta2
File('results/beta2_obs.pvd') << model.beta2

A = AdjointSolver(model,config)
u_o = model.u.vector().get_local()
v_o = model.v.vector().get_local()
U_e = 10.0

for i in range(50):
  config['output_path'] = 'results/run_'+str(i)+'/'
  model.beta2.vector()[:] = 1000.
  u_error = U_e*random.randn(len(u_o))
  v_error = U_e*random.randn(len(v_o))
  model.u_o.vector().set_local(u_o+u_error)
  model.v_o.vector().set_local(v_o+v_error)
  model.u_o.vector().apply('insert')
  model.v_o.vector().apply('insert')
  A.solve()
