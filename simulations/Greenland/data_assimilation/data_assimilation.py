#!/usr/bin/env python
import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model              as model
import src.solvers            as solvers
import src.physical_constants as pc
from data.data_factory   import DataFactory
from meshes.mesh_factory import MeshFactory
from src.helper          import default_nonlin_solver_params
from src.utilities       import DataInput
from dolfin              import *

set_log_active(True)

vara = DataFactory.get_searise()

mesh                    = MeshFactory.get_greenland_detailed()
flat_mesh               = MeshFactory.get_greenland_detailed()

dd                 = DataInput(None, vara, mesh=mesh)

Surface            = dd.get_spline_expression('h')
Bed                = dd.get_spline_expression('b')
Smb                = dd.get_spline_expression('adot')
SurfaceTemperature = dd.get_spline_expression('T')
BasalHeatFlux      = dd.get_spline_expression('q_geo')
U_observed         = dd.get_spline_expression('U_ob')

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 0.5
nonlin_solver_params['newton_solver']['relative_tolerance'] = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations'] = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver']                            = 'mumps'
nonlin_solver_params['preconditioner']                           = 'default'

config = { 'mode'                         : 'steady',
           't_start'                      : None,
           't_end'                        : None,
           'time_step'                    : None,
           'output_path'                  : './results_2/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : False,
           'log'                          : True,
           'coupled' : 
           { 
             'on'        : True,
             'inner_tol' : 0.0,
             'max_iter'  : 5
           },
           'velocity' : 
           { 
             'on'             : True,
             'newton_params'  : nonlin_solver_params,
             'viscosity_mode' : 'full',
             'b_linear'       : None,
             'use_T0'         : True,
             'T0'             : 268.0,
             'A0'             : None,
             'beta2'          : 2.0,
             'r'              : 1.0,
             'E'              : 1.0,
             'approximation'  : 'stokes',
             'boundaries'     : None
           },
           'enthalpy' : 
           { 
             'on'                  : True,
             'use_surface_climate' : False,
             'T_surface'           : SurfaceTemperature,
             'q_geo'               : BasalHeatFlux,
             'lateral_boundaries'  : None
           },
           'free_surface' :
           { 
             'on'               : False,
             'lump_mass_matrix' : True,
             'thklim'           : 10.0,
             'use_pdd'          : False,
             'observed_smb'     : None,
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
             'alpha'              : 0.0,
             'beta'               : 0.0,
             'max_fun'            : 100,
             'objective_function' : 'logarithmic',
             'control_variable'   : None,
             'bounds'             : [(0,20)]
           }}



model = model.Model()
model.set_geometry(Surface, Bed)

model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()
model.eps_reg = 1e-5

F = solvers.SteadySolver(model,config)
File('results/beta2_opt.xml') >> model.beta2
F.solve()
#model.adot = Smb

visc    = project(model.eta)
vel_par = config['velocity']
vel_par['viscosity_mode']                                         = 'linear'
vel_par['b_linear']                                               = visc
vel_par['newton_params']['newton_solver']['relaxation_parameter'] = 1.0

config['enthalpy']['on']        = False
config['surface_climate']['on'] = False
config['coupled']['on']         = False
config['velocity']['use_T0']    = False
config['adjoint']['control_variable']    = [model.beta2,model.U_o]

A = solvers.AdjointSolver(model,config)
A.set_target_velocity(U = U_observed)
U_loc = model.U_o.vector().get_local()
U_loc[U_loc<0] = 0.0
model.U_o.vector().set_local(U_loc)
U_o_min = Function(model.Q)
U_o_min.vector().set_local(model.U_o.vector().get_local()*0.8)
U_o_max = Function(model.Q)
U_o_max.vector().set_local(model.U_o.vector().get_local()*1.2)
config['adjoint']['bounds'] = [(0,20),(U_o_min,U_o_max)]
#File('./results/beta2_opt.xml') >> model.beta2
A.solve()

