import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model
import src.solvers
import src.physical_constants
import src.helper
import pylab
import dolfin
import scipy.io
from data.data_factory   import DataFactory
from meshes.mesh_factory import MeshFactory
from src.utilities       import DataInput

dolfin.set_log_active(True)

vara = DataFactory.get_searise(thklim = 50.0)

mesh                    = MeshFactory.get_greenland_coarse()
flat_mesh               = MeshFactory.get_greenland_coarse() 
mesh.coordinates()[:,2] = mesh.coordinates()[:,2]/1000.0

dd                 = DataInput(None, vara, mesh=mesh)

Surface            = dd.get_spline_expression('h')
Bed                = dd.get_spline_expression('b')
SMB                = dd.get_spline_expression('adot')
SurfaceTemperature = dd.get_spline_expression('T')
BasalHeatFlux      = dd.get_spline_expression('q_geo')
U_observed         = dd.get_spline_expression('U_ob')
Tn                 = vara['Tn']['map_data']
           
nonlin_solver_params = src.helper.default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.7
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver']                            = 'gmres'
nonlin_solver_params['preconditioner']                           = 'hypre_amg'

config = { 'mode'                         : 'transient',
           't_start'                      : 0.0,
           't_end'                        : 50.0,
           'time_step'                    : 0.5,
           'output_path'                  : './results/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : False,
           'log'                          : True, 
           'coupled' : 
           { 
             'on'        : False,
             'inner_tol' : 0.0,
             'max_iter'  : 3
           },
           'velocity' : 
           { 
             'on'             : True,
             'newton_params'  : nonlin_solver_params,
             'viscosity_mode' : 'full',
             'b_linear'       : None,
             'use_T0'         : True,
             'T0'             : 273.0,
             'A0'             : None,
             'beta2'          : 1.0,
             'r'              : 1.0,
             'E'              : 1.0,
             'approximation'  : 'fo',
             'boundaries'     : None
           },
           'enthalpy' : 
           { 
             'on'                  : False,
             'T_surface'           : SurfaceTemperature,
             'q_geo'               : BasalHeatFlux,
             'use_surface_climate' : False,
             'lateral_boundaries'  : None
           },
           'free_surface' :
           { 
             'on'                         : True,
             'observed_smb'               : SMB,
             'lump_mass_matrix'           : True,
             'thklim'                     : 50.0,
             'use_pdd'                    : False,
             'use_shock_capturing'        : False,
             'static_boundary_conditions' : True
           },  
           'age' : 
           { 
             'on'               : False,
             'use_smb_for_ela'  : False,
             'ela'              : None,
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
             'alpha'              : 0.0,
             'beta'               : 100.0,
             'max_fun'            : 20,
             'objective_function' : 'logarithmic',
           }}

model = src.model.Model()

model.set_geometry(Surface, Bed)
model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(src.physical_constants.IceParameters())
model.initialize_variables()

F = src.solvers.SteadySolver(model,config)
F.solve()

config['velocity']['use_T0'] = False
config['velocity']['newton_params']['newton_solver']['relaxation_parameter'] = 0.8

T = src.solvers.TransientSolver(model,config)
T.solve()

