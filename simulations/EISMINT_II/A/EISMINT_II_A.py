import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.model              import Model
from src.helper             import default_nonlin_solver_params
from src.solvers            import SteadySolver, TransientSolver
from src.physical_constants import IceParameters
from dolfin                 import Expression, sqrt, set_log_active, \
                                   parameters, Constant, File
from meshes.mesh_factory    import MeshFactory

set_log_active(True)

L     = 750000.0
S_0   = 10.0
S_b   = 1e-5
R_el  = 450000.0
M_max = 0.5

T_min = 238.15
S_T = 1.67e-5


class MassBalance(Expression):
  def eval(self,values,x):
    values[0] = min(M_max,S_b*(R_el-sqrt(x[0]**2 + x[1]**2))) 

class SurfaceTemperature(Expression):
  def eval(self,values,x):
    values[0] =  T_min + S_T*sqrt(x[0]**2 + x[1]**2)

mesh  = MeshFactory.get_circle()

model = Model()
model.set_mesh(mesh, deform=False)

Surface = Expression('S_0', S_0 = S_0, element = model.Q.ufl_element())
Bed     = Expression('0.0', element = model.Q.ufl_element())

model.set_geometry(Surface, Bed)
model.deform_mesh_to_geometry()

model.set_parameters(IceParameters())
model.initialize_variables()

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 1.0
nonlin_solver_params['newton_solver']['absolute_tolerance']   = 1.0
nonlin_solver_params['newton_solver']['linear_solver']        = 'gmres'
nonlin_solver_params['newton_solver']['preconditioner']       = 'hypre_amg'
parameters['form_compiler']['quadrature_degree']              = 2


config = { 'mode'                         : 'steady',
           't_start'                      : 0.0,
           't_end'                        : 100.0,
           'time_step'                    : 10.0,
           'output_path'                  : './results/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : False,
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
             'viscosity_mode' : 'full',
             'b_linear'       : None,
             'use_T0'         : True,
             'T0'             : 268.0,
             'A0'             : 1e-16,
             'beta2'          : 1e5,
             'r'              : 1.0,
             'E'              : 1.0,
             'approximation'  : 'fo',
             'boundaries'     : None
           },
           'enthalpy' : 
           { 
             'on'                  : True,
             'use_surface_climate' : False,
             'T_surface'           : SurfaceTemperature(),
             'q_geo'               : 0.042*60**2*24*365,
             'lateral_boundaries'  : None
             
           },
           'free_surface' :
           { 
             'on'                         : True,
             'lump_mass_matrix'           : False,
             'use_shock_capturing'        : True,
             'thklim'                     : 10.0,
             'use_pdd'                    : False,
             'observed_smb'               : MassBalance(),
             'static_boundary_conditions' :False
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

F = SteadySolver(model,config)
F.solve()

T = TransientSolver(model,config)
T.solve()

File('./results/u.xml') << model.u
File('./results/v.xml') << model.v
File('./results/w.xml') << model.w
File('./results/S.xml') << model.S
File('./results/T.xml') << model.T
