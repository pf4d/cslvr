import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

import src.solvers            as solvers
import src.physical_constants as pc
import src.model              as model
from meshes.mesh_factory  import MeshFactory
from data.data_factory    import DataFactory
from src.helper           import default_nonlin_solver_params
from src.utilities        import DataInput, DataOutput
from dolfin               import *
from pylab                import sqrt, copy
from time                 import time

# make the directory if needed :
out_dir   = './results/'

set_log_active(True)
#set_log_level(PROGRESS)

thklim = 10.0

# collect the raw data :
searise  = DataFactory.get_searise(thklim = thklim)
measure  = DataFactory.get_gre_measures()
#meas_shf = DataFactory.get_shift_gre_measures()
bamber   = DataFactory.get_bamber(thklim = thklim)
fm_qgeo  = DataFactory.get_gre_qgeo_fox_maule()
#sec_qgeo = DataFactory.get_gre_qgeo_secret()
merged   = DataFactory.get_gre_merged()

# define the meshes :
mesh      = Mesh('../../../meshes/greenland/greenland_coarse_mesh.xml')
flat_mesh = Mesh('../../../meshes/greenland/greenland_coarse_mesh.xml')
mesh.coordinates()[:,2]      /= 1000.0
flat_mesh.coordinates()[:,2] /= 1000.0


# create data objects to use with varglas :
dsr     = DataInput(None, searise,  mesh=mesh)
dbm     = DataInput(None, bamber,   mesh=mesh)
#dms     = DataInput(None, measure,  mesh=mesh)
#dmss    = DataInput(None, meas_shf, mesh=mesh)
dfm     = DataInput(None, fm_qgeo,  mesh=mesh)
#dsq     = DataInput(None, sec_qgeo, mesh=mesh)
dmg     = DataInput(None, merged,   mesh=mesh)
#dbv     = DataInput("results/", ("Ubmag_measures.mat", "Ubmag.mat"), mesh=mesh)

# change the projection of the measures data to fit with other data :
#dms.change_projection(dsr)
dmg.change_projection(dsr)

# get the expressions used by varglas :
Thickness          = dbm.get_spline_expression('H')
Surface            = dbm.get_spline_expression('h')
Bed                = dbm.get_spline_expression('b')
SurfaceTemperature = dsr.get_spline_expression('T')
BasalHeatFlux      = dfm.get_spline_expression('q_geo')
adot               = dsr.get_spline_expression('adot')
U_observed         = dmg.get_spline_expression('v_mag')

model = model.Model()
model.set_geometry(Surface, Bed)
model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()


# specifify non-linear solver parameters :
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.7
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['absolute_tolerance']      = 1e2
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['newton_solver']['linear_solver']           = 'mumps'
nonlin_solver_params['newton_solver']['preconditioner']          = 'default'
parameters['form_compiler']['quadrature_degree']                 = 2

config = { 'mode'                         : 'steady',
           't_start'                      : None,
           't_end'                        : None,
           'time_step'                    : None,
           'output_path'                  : out_dir,
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : False,
           'log'                          : True,
           'coupled' : 
           { 
             'on'       : True,
             'inner_tol': 0.0,
             'max_iter' : 5
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
             'beta2'          : 0.5,
             'r'              : 1.0,
             'E'              : 1.0,
             'approximation'  : 'fo',
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
             'thklim'           : thklim,
             'use_pdd'          : False,
             'observed_smb'     : adot,
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
             'alpha'               : [Thickness**2],
             'beta'                : 0.0,
             'max_fun'             : 20,
             'objective_function'  : 'logarithmic',
             'bounds'              : [(0,20)],
             'control_variable'    : None,
             'regularization_type' : 'Tikhonov'
           }}

model.eps_reg = 1e-5

F = solvers.SteadySolver(model,config)
t01 = time()
F.solve()
tf1 = time()

params = config['velocity']['newton_params']['newton_solver']
params['relaxation_parameter']         = 1.0
config['velocity']['viscosity_mode']   = 'linear'
config['velocity']['b_linear']         = project(model.eta, model.Q)
config['enthalpy']['on']               = False
config['surface_climate']['on']        = False
config['coupled']['on']                = False
config['velocity']['use_T0']           = False
config['adjoint']['control_variable']  = [model.beta2]

A = solvers.AdjointSolver(model,config)
A.set_target_velocity(U = U_observed)
t02 = time()
A.solve()
tf2 = time()

File(out_dir + 'S.xml')       << model.S
File(out_dir + 'B.xml')       << model.B
File(out_dir + 'u.xml')       << model.u
File(out_dir + 'v.xml')       << model.v
File(out_dir + 'w.xml')       << model.w
File(out_dir + 'beta2.xml')   << model.beta2
File(out_dir + 'eta.xml')     << model.eta
