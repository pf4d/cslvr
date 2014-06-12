import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.model              import Model
from src.solvers            import SteadySolver
from src.physical_constants import IceParameters
from src.helper             import default_nonlin_solver_params
from dolfin                 import set_log_active, File, Expression, pi
from pylab                  import sin, tan, deg2rad

set_log_active(True)

alpha   = deg2rad(0.5)
lengths = [40000]
for L in lengths:

  class Surface(Expression):
    def __init__(self):
      pass
    def eval(self,values,x):
      values[0] = - x[0] * tan(alpha)


  class Bed(Expression):
    def __init__(self):
      pass
    def eval(self,values,x):
      values[0] = - x[0] * tan(alpha) \
                  - 1000.0 \
                  + 500.0 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)

  config = { 'mode'                         : 'steady',
             'output_path'                  : './results_BP/',
             'wall_markers'                 : [],
             'periodic_boundary_conditions' : True,
             't_start'                      : None,
             't_end'                        : None,
             'time_step'                    : None,
             'log'                          : True,
             'coupled' : 
             { 
               'on'                  : False,
               'inner_tol'           : 0.0,
               'max_iter'            : 1
             },
             'velocity' : 
             { 
               'on'                  : True,
               'newton_params'       : default_nonlin_solver_params(),
               'viscosity_mode'      : 'isothermal',
               'b_linear'            : None,
               'use_T0'              : False,
               'T0'                  : None,
               'A0'                  : 1e-16,
               'beta2'               : 1e3,
               'r'                   : 1,
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
               'alpha'               : None,
               'beta'                : None,
               'max_fun'             : None,
               'objective_function'  : 'logarithmic',
               'animate'             : False
             }}

  model = Model()
  model.set_geometry(Surface(), Bed())

  nx = 20
  ny = 20 
  nz = 5

  model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, 
                              ymin=0, ymax=L, generate_pbcs=True)

  model.set_parameters(IceParameters())
  model.initialize_variables()
   
  newt_params = config['velocity']['newton_params']
  if L in [5000,10000,20000,40000]:
    newt_params['newton_solver']['preconditioner']       = 'default'
    newt_params['newton_solver']['relaxation_parameter'] = 0.7
  else:
    newt_params['linear_solver']                         = 'gmres'
    newt_params['preconditioner']                        = 'hypre_amg'
    newt_params['newton_solver']['relaxation_parameter'] = 0.8

  F = SteadySolver(model, config)
  F.solve()

