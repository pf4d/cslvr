#
# Assimilation run times (16 cores, 20 function evals):
# =====================================================
#
#  mesh type          | # elements | run 1    | run 2    | run 3    | run 4    
#  -------------------+------------+----------+----------+----------+----------
#  10 layers 1xGrid   |    1187610 |          | 00:26:31 | 00:27:40 | 00:27:46 
#  20 layers 1xGrid   |    2374020 | 01:13:09 | 01:18:18 | 01:17:33 | 01:18:29 
#  30 layers 1xGrid   |    4265526 |    /     |          |          |          
#  10 layers 2.5xGrid |    2114880 | 01:17:34 | 01:19:21 | 01:22:38 | 01:35:02 
#  10 layers 3xGrid   |    2680020 | 01:41:14 | 01:51:47 | 02:06:10m| 01:51:46
#  10 layers 3.5xGrid |    3992850 |    /     |          |          | 
#
#

import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

import src.solvers            as solvers
import src.physical_constants as pc
import src.model              as model
from meshes.mesh_factory import MeshFactory
from data.data_factory   import DataFactory
from src.helper          import default_nonlin_solver_params
from src.utilities       import DataInput, DataOutput
from dolfin              import *
from pylab               import sqrt, copy
from time                import time

set_log_active(True)
#set_log_level(PROGRESS)

thklim = 200.0

# collect the raw data :
searise  = DataFactory.get_searise()
measure  = DataFactory.get_gre_measures()
meas_shf = DataFactory.get_shift_gre_measures()
bamber   = DataFactory.get_bamber(thklim = thklim)
fm_qgeo  = DataFactory.get_gre_qgeo_fox_maule()
sec_qgeo = DataFactory.get_gre_qgeo_secret()

# define the meshes :
#mesh      = Mesh('../meshes/mesh.xml')
#flat_mesh = Mesh('../meshes/mesh.xml')
#mesh      = MeshFactory.get_greenland_detailed()
#flat_mesh = MeshFactory.get_greenland_detailed()
#mesh      = Mesh('meshes/3dmesh_10_layers.xml')
#flat_mesh = Mesh('meshes/3dmesh_10_layers.xml')
#mesh      = Mesh('meshes/3dmesh_20_layers.xml')
#flat_mesh = Mesh('meshes/3dmesh_20_layers.xml')
#mesh      = Mesh('meshes/3dmesh_10_layers_5xGrid.xml')
#flat_mesh = Mesh('meshes/3dmesh_10_layers_5xGrid.xml')
#mesh      = Mesh('meshes/3dmesh_10_layers_2.5xGrid.xml')
#flat_mesh = Mesh('meshes/3dmesh_10_layers_2.5xGrid.xml')
#mesh      = Mesh('meshes/3dmesh_10_layers_3xGrid.xml')
#flat_mesh = Mesh('meshes/3dmesh_10_layers_3xGrid.xml')
#mesh      = Mesh('meshes/3dmesh_10_layers_3.5xGrid.xml')
#flat_mesh = Mesh('meshes/3dmesh_10_layers_3.5xGrid.xml')
#mesh      = Mesh('meshes/3dmesh_30_layers.xml')
#flat_mesh = Mesh('meshes/3dmesh_30_layers.xml')
#mesh      = Mesh('meshes/3dmesh_10_layers_2xGrid.xml')
#flat_mesh = Mesh('meshes/3dmesh_10_layers_2xGrid.xml')
mesh.coordinates()[:,2]      /= 100000.0
flat_mesh.coordinates()[:,2] /= 100000.0


# create data objects to use with varglas :
dsr     = DataInput(None, searise,  mesh=mesh, create_proj=True)
dbm     = DataInput(None, bamber,   mesh=mesh)
#dms     = DataInput(None, measure,  mesh=mesh, create_proj=True, flip=True)
#dmss    = DataInput(None, meas_shf, mesh=mesh, flip=True)
dfm     = DataInput(None, fm_qgeo,  mesh=mesh)
dsq     = DataInput(None, sec_qgeo, mesh=mesh)
#dbv     = DataInput("results/", ("Ubmag_measures.mat", "Ubmag.mat"), mesh=mesh)

# change the projection of the measures data to fit with other data :
#dms.change_projection(dsr)

# inspect the data values :
do      = DataOutput('results_pre/')
#do.write_one_file('h',              dbm.get_projection('h'))
#do.write_one_file('Ubmag_measures', dbv.get_projection('Ubmag_measures'))
#do.write_one_file('sq_qgeo',        dsq.get_projection('q_geo'))
#do.write_one_file('sr_qgeo',        dsr.get_projection('q_geo'))
#exit(0)

# get the expressions used by varglas :
Thickness          = dbm.get_spline_expression('H_n')
Surface            = dbm.get_spline_expression('h_n')
Bed                = dbm.get_spline_expression('b')
SurfaceTemperature = dsr.get_spline_expression('T')
#BasalHeatFlux      = dsr.get_spline_expression('q_geo')
BasalHeatFlux      = dsq.get_spline_expression('q_geo')
#BasalHeatFlux      = dfm.get_spline_expression('q_geo')
adot               = dsr.get_spline_expression('adot')
U_observed         = dsr.get_spline_expression('U_ob')


# specifify non-linear solver parameters :
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.5
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver']                            = 'mumps'
nonlin_solver_params['preconditioner']                           = 'default'

# make the directory if needed :
i = int(sys.argv[1])
#dir_b   = './results_new_mesh_20_layers_sr/0'
#dir_b   = './results_new_mesh_20_layers_sq/0'
#dir_b   = './results_new_mesh_20_layers_42/0'
#dir_b   = './results_new_mesh_10_layers_sr/0'
#dir_b   = './results_new_mesh_10_layers_sq/0'
#dir_b   = './results_10_layers_alphaH2_sq/0'
#dir_b   = './results_20_layers_alphaH2_sq/0'
#dir_b   = './results_30_layers_alphaH2_sq/0'
#dir_b   = './results_10_layers_alphaH2_5xGrid_sq/0'
#dir_b   = './results_10_layers_alphaH2_2.5xGrid_sq/0'
#dir_b   = './results_10_layers_alphaH2_3xGrid_sq/0'
#dir_b   = './results_10_layers_alphaH2_3.5xGrid_sq/0'
#dir_b   = './results_detailed_sr/0'
#dir_b   = './results_detailed_fm/0'
#dir_b   = './results_detailed_sq/0'

# make the directory if needed :
out_dir = dir_b + str(i) + '/'
d       = os.path.dirname(out_dir)
if not os.path.exists(d):
  os.makedirs(d)

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
             'alpha'               : [Thickness**2],
             'beta'                : 0.0,
             'max_fun'             : 20,
             'objective_function'  : 'logarithmic',
             'control_variable'    : None,
             'bounds'              : [(0,20)],
             'regularization_type' : 'Tikhonov'
           }}


model = model.Model()
model.set_geometry(Surface, Bed)

model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()
model.eps_reg = 1e-5
#config['adjoint']['alpha'] = model.S - model.B

F = solvers.SteadySolver(model,config)
if i != 0: File(dir_b + str(i-1) + '/beta2_opt.xml') >> model.beta2
t01 = time()
F.solve()
tf1 = time()
if i != 0: model.adot = adot

visc    = project(model.eta)
vel_par = config['velocity']
vel_par['viscosity_mode']                                         = 'linear'
vel_par['b_linear']                                               = visc
vel_par['newton_params']['newton_solver']['relaxation_parameter'] = 1.0

config['enthalpy']['on']               = False
config['surface_climate']['on']        = False
config['coupled']['on']                = False
if i !=0: config['velocity']['use_T0'] = False
config['adjoint']['control_variable']  = [model.beta2]

A = solvers.AdjointSolver(model,config)
A.set_target_velocity(U = U_observed)
if i != 0: File(dir_b + str(i-1) + '/beta2_opt.xml') >> model.beta2
t02 = time()
A.solve()
tf2 = time()

File(dir_b + str(i) + '/Mb.pvd') << model.Mb

# calculate total time to compute
s = (tf1 - t01) + (tf2 - t02)
m = s / 60.0
h = m / 60.0
s = s % 60
m = m % 60
print "Total time to compute: \r%02d:%02d:%02d" % (h,m,s)

