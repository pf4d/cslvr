import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model
import src.solvers
import src.physical_constants
import src.helper
from pylab  import sin, cos, exp, deg2rad
from dolfin import Expression, File, set_log_active

set_log_active(True)

theta = deg2rad(-3.0)
L     = 100000.
H     = 1000.0
a0    = 100
sigma = 10000

class Surface(Expression):
  def __init__(self):
    pass
  def eval(self, values, x):
    values[0] = sin(theta) / cos(theta) * x[0]

class Bed(Expression):
  def __init__(self):
    pass
  def eval(self, values, x):
    y_0       = -H + a0 * (exp(-((x[0]-L/2.)**2 + (x[1]-L/2.)**2) / sigma**2))
    values[0] = sin(theta)/cos(theta) * (x[0] + sin(theta)*y_0) + cos(theta)*y_0

class SMB(Expression):
  def eval(self, values, x):
    values[0] = 0.0

nonlin_solver_params = src.helper.default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 1.0
nonlin_solver_params['newton_solver']['relative_tolerance']   = 1.0
nonlin_solver_params['newton_solver']['linear_solver']                         = 'mumps'
nonlin_solver_params['newton_solver']['preconditioner']                        = 'default'

config = { 'mode'                         : 'transient',
           'output_path'                  : './results/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : True,
           't_start'                      : 0.0,
           't_end'                        : 500.0,
           'time_step'                    : 2.0,
           'log'                          : True,
           'coupled' : 
           { 
             'on'        : False,
             'inner_tol' : 0.0,
             'max_iter'  : 1
           },
           'velocity' : 
           { 
             'on'             : True,
             'newton_params'  : nonlin_solver_params,
             'viscosity_mode' : 'isothermal',
             'b_linear'       : None,
             'use_T0'         : False,
             'T0'             : None,
             'A0'             : 2.140373e-7,
             'beta2'          : 1e3,
             'r'              : 1.0,
             'E'              : 1,
             'approximation'  : 'stokes',
             'boundaries'     : None
           },
           'enthalpy' : 
           { 
             'on'                  : False,
             'use_surface_climate' : False,
             'T_surface'           : None,
           },
           'free_surface' :
           { 
             'on'                         : True,
             'lump_mass_matrix'           : False,
             'use_shock_capturing'        : False,
             'thklim'                     : 10.0,
             'use_pdd'                    : False,
             'observed_smb'               : SMB(),
             'static_boundary_conditions' : False
           },  
           'age' : 
           { 
             'on'              : False,
             'use_smb_for_ela' : False,
             'ela'             : None,
           },
           'surface_climate' : 
           { 
             'on'     : False,
             'T_ma'   : None,
             'T_ju'   : None,
             'beta_w' : None,
             'sigma'  : None,
             'precip' : None
           },
           'adjoint' :
           { 
             'alpha'              : None,
             'beta'               : None,
             'max_fun'            : None,
             'objective_function' : 'logarithmic',
             'animate'            : False
           }}

model = src.model.Model()
model.set_geometry(Surface(), Bed())

nx = 20
ny = 20
nz = 5

model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, ymin=0, ymax=L,
                            generate_pbcs = True)
model.set_parameters(src.physical_constants.IceParameters())
model.initialize_variables()
model.n = 1.0

T = src.solvers.TransientSolver(model, config)
T.solve()

File('./results/u.xml') << model.u
File('./results/v.xml') << model.v
File('./results/w.xml') << model.w
File('./results/S.xml') << model.S

