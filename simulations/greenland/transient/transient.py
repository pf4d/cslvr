from varglas.data.data_factory  import DataFactory
from varglas.mesh.mesh_factory  import MeshFactory
from varglas.utilities          import DataInput
from varglas.model              import Model
from varglas.solvers            import SteadySolver, TransientSolver
from varglas.physical_constants import IceParameters
from varglas.helper             import default_nonlin_solver_params
from fenics                     import set_log_active, parameters

set_log_active(True)

thklim = 50.0

vara = DataFactory.get_searise(thklim = thklim)
mesh = MeshFactory.get_greenland_coarse()
dsr  = DataInput(vara, mesh=mesh)

S     = dsr.get_spline_expression('S')
B     = dsr.get_spline_expression('B')
SMB   = dsr.get_spline_expression('adot')
T_s   = dsr.get_spline_expression('T')
q_geo = dsr.get_spline_expression('q_geo')
U_ob  = dsr.get_spline_expression('U_ob')
Tn    = vara['Tn']['map_data']

# create the model :
model = Model()

model.set_mesh(mesh)
model.set_geometry(S, B, deform=True)
model.set_parameters(IceParameters())
model.calculate_boundaries()
model.initialize_variables()

# specify solver parameters : 
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.7
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['newton_solver']['linear_solver']           = 'gmres'
nonlin_solver_params['newton_solver']['preconditioner']          = 'hypre_amg'
parameters['form_compiler']['quadrature_degree']                 = 2

# varglas config dictionary :
config = { 'mode'                         : 'transient',
           't_start'                      : 0.0,
           't_end'                        : 50.0,
           'time_step'                    : 0.5,
           'output_path'                  : './results/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : False,
           'log'                          : True, 
           'velocity' : 
           { 
             'on'                         : True,
             'newton_params'              : nonlin_solver_params,
             'viscosity_mode'             : 'full',
             'b_linear'                   : None,
             'use_T0'                     : True,
             'T0'                         : 273.0,
             'A0'                         : None,
             'beta2'                      : 1.0,
             'r'                          : 1.0,
             'E'                          : 1.0,
             'approximation'              : 'fo',
             'boundaries'                 : None
           },
           'coupled' : 
           { 
             'on'                         : False,
             'inner_tol'                  : 0.0,
             'max_iter'                   : 3
           },
           'enthalpy' : 
           { 
             'on'                         : False,
             'T_surface'                  : T_s,
             'q_geo'                      : q_geo,
             'use_surface_climate'        : False,
             'lateral_boundaries'         : None
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
             'alpha'                      : 0.0,
             'beta'                       : 100.0,
             'max_fun'                    : 20,
             'objective_function'         : 'logarithmic',
           }}

F = SteadySolver(model, config)
F.solve()

config['velocity']['use_T0'] = False
np = config['velocity']['newton_params']['newton_solver']
np['relaxation_parameter'] = 0.8

T = TransientSolver(model,config)
T.solve()



