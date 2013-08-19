import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model              as model
import src.solvers            as solvers
import src.physical_constants as pc
from data.data_factory import DataFactory
from src.helper        import default_nonlin_solver_params
from src.utilities     import DataInput
from dolfin            import *

set_log_active(True)

var_measures = DataFactory.get_ant_measures()
var_bedmap1 = DataFactory.get_lebrocq()
var_bedmap2 = DataFactory.get_bedmap2()

mesh                  = Mesh('../meshes/antarctica_50H_5l.xml')
flat_mesh             = Mesh('../meshes/antarctica_50H_5l.xml')

dm  = DataInput(None, var_measures, mesh=mesh,flip=True)
db1 = DataInput(None, var_bedmap1, mesh=mesh)
db2 = DataInput(None, var_bedmap2, mesh=mesh, flip=True)

thklim = 50.0

db2.set_data_min("H", thklim, thklim)
db2.set_data_min("h", 0.0,0.0)

db2.set_data_max("H",30000.,thklim)
db2.set_data_max("h",30000.,0.0)

db2.data['b'] = db2.data['h']-db2.data['H']

db1.data['srfTemp'] += 273.15
db1.data['q_geo'] *= 60**2*24*365


Surface            = db2.get_spline_expression("h")
Bed                = db2.get_spline_expression("b")
SurfaceTemperature = db1.get_spline_expression("srfTemp")
BasalHeatFlux      = db1.get_spline_expression("q_geo")
U_observed         = dm.get_spline_expression("v_mag")


#===============================================================================
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 0.5
nonlin_solver_params['newton_solver']['relative_tolerance'] = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations'] = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver'] = 'mumps'
nonlin_solver_params['preconditioner'] = 'default'

config = { 'mode' : 'steady',
           'coupled' : 
               { 'on' : False,
                 'inner_tol': 0.0,
                 'max_iter' : 5
               },
           't_start' : None,
           't_end' : None,
           'time_step' : None,
           'velocity' : 
               { 'on' : True,
                 'newton_params' : nonlin_solver_params,
                 'viscosity_mode' : 'full',
                 'b_linear' : None,
                 'use_T0': True,
                 'T0' : 268.0,
                 'A0' : 1e-16,
                 'beta2' : 2.0,
                 'r' : 1.0,
                 'E' : 1.0,
                 'approximation' : 'fo',
                 'boundaries' : None
               },
           'enthalpy' : 
               { 'on': False,
                 'use_surface_climate': False,
                 'T_surface' : SurfaceTemperature,
                 'q_geo' : BasalHeatFlux,
                 'lateral_boundaries' : None
                 
               },
           'free_surface' :
               { 'on': False,
                 'lump_mass_matrix': True,
                 'thklim': 10.0,
                 'use_pdd': False,
                 'observed_smb': None,
               },  
           'age' : 
               { 'on': False,
                 'use_smb_for_ela': False,
                 'ela': None,
               },
            'surface_climate' : 
               { 'on': False,
                 'T_ma': None,
                 'T_ju': None,
                 'beta_w': None,
                 'sigma': None,
                 'precip': None
               },
            'adjoint' :
               { 'alpha' : 1e3,
                 'beta' : 0.0,
                 'max_fun' : 50,
                 'objective_function' : 'logarithmic',
                 'bounds':(0.,20.)
               },
            'output_path' : './results_coarse/',
            'wall_markers' : [],
            'periodic_boundary_conditions' : False,
            'log': True }

model = model.Model()
model.set_geometry(Surface,Bed)

model.set_mesh(mesh,flat_mesh=flat_mesh,deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()

F = solvers.SteadySolver(model,config)
#dolfin.File('results_coarse/beta2_opt.xml') >> model.beta2
F.solve()

visc = project(model.eta)
config['velocity']['viscosity_mode'] = 'linear'
config['velocity']['b_linear'] = visc
config['velocity']['newton_params']['newton_solver']['relaxation_parameter'] = 1.0

config['enthalpy']['on'] = False
config['surface_climate']['on'] = False
config['coupled']['on'] = False
config['velocity']['use_T0'] = False

A = solvers.AdjointSolver(model,config)
A.set_target_velocity(U = U_observed)
File('results_coarse/U_obs.pvd') << model.U_o
#dolfin.File('results_coarse/beta2_opt.xml') >> model.beta2
A.solve()

