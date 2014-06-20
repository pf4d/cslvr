import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.model              import Model
from src.solvers            import TransientSolver
from src.physical_constants import IceParameters
from src.helper             import default_nonlin_solver_params
from dolfin                 import set_log_active, File, Expression, pi, \
                                   sin, tan, cos, exp

set_log_active(True)

theta = -3.0 * pi / 180
L     = 100000.0
H     = 1000.0
a0    = 100
sigma = 10000

nx = 50
ny = 50
nz = 10

model = Model()
model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, ymin=0, ymax=L,
                            generate_pbcs = True)

Surface = Expression('tan(theta) * x[0]', theta=theta,
                     element=model.Q.ufl_element())
class Bed(Expression):
  def __init__(self, element=None):
    pass
  def eval(self, values, x):
    y_0       = -H + a0 * (exp(-((x[0]-L/2.)**2 + (x[1]-L/2.)**2) / sigma**2))
    values[0] = sin(theta)/cos(theta) * (x[0] + sin(theta)*y_0) \
                + cos(theta)*y_0
Bed = Bed(element=model.Q.ufl_element())

SMB = Expression('0.0', element=model.Q.ufl_element())

model.set_geometry(Surface, Bed, deform=True)

model.set_parameters(IceParameters())
model.initialize_variables()
model.n = 1.0

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 1.0
nonlin_solver_params['newton_solver']['relative_tolerance']   = 1.0
#nonlin_solver_params['newton_solver']['linear_solver']        = 'mumps'
#nonlin_solver_params['newton_solver']['preconditioner']       = 'default'
nonlin_solver_params['newton_solver']['linear_solver']        = 'gmres'
nonlin_solver_params['newton_solver']['preconditioner']       = 'hypre_amg'

config = { 'mode'                         : 'transient',
           't_start'                      : 0.0,
           't_end'                        : 500.0,
           'time_step'                    : 2.0,
           'output_path'                  : './results/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : True,
           'log'                          : True,
           'coupled' : 
           { 
             'on'                         : False,
             'inner_tol'                  : 0.0,
             'max_iter'                   : 1
           },                             
           'velocity' :                   
           {                              
             'on'                         : True,
             'newton_params'              : nonlin_solver_params,
             'viscosity_mode'             : 'isothermal',
             'b_linear'                   : None,
             'use_T0'                     : False,
             'T0'                         : None,
             'A0'                         : 2.140373e-7,
             'beta2'                      : (1*2.140373e-7*1000)**-1.,
             'r'                          : 0.0,
             'E'                          : 1,
             'approximation'              : 'fo',
             'boundaries'                 : None
           },                             
           'enthalpy' :                   
           {                              
             'on'                         : False,
             'use_surface_climate'        : False,
             'T_surface'                  : None,
             
           },
           'free_surface' :
           { 
             'on'                         : True,
             'lump_mass_matrix'           : False,
             'use_shock_capturing'        : False,
             'thklim'                     : 10.0,
             'use_pdd'                    : False,
             'observed_smb'               : SMB,
             'static_boundary_conditions' : False
           },  
           'age' : 
           { 
             'on'                         : False,
             'use_smb_for_ela'            : False,
             'ela'                        : None,
           },                             
           'surface_climate' :            
           {                              
             'on'                         : False,
             'T_ma'                       : None,
             'T_ju'                       : None,
             'beta_w'                     : None,
             'sigma'                      : None,
             'precip'                     : None
           },                             
           'adjoint' :                    
           {                              
             'alpha'                      : None,
             'beta'                       : None,
             'max_fun'                    : None,
             'objective_function'         : 'logarithmic',
             'animate'                    : False
           }}

T = TransientSolver(model,config)
T.solve()

File('./results_stokes/S.xml') << model.S

