#
# FEniCS 1.2.0 Assimilation run times (16 cores, 20 function evals):
# ==================================================================
#
#  mesh type          | # elements | run 1    | run 2    | run 3    | run 4    
#  -------------------+------------+----------+----------+----------+----------
#  10 layers 1xGrid   |    1187610 |    NA    | 00:26:31 | 00:27:40 | 00:27:46 
#  20 layers 1xGrid   |    2374020 | 01:13:09 | 01:18:18 | 01:17:33 | 01:18:29 
#  30 layers 1xGrid   |    4265526 |    /     |          |          |          
#  10 layers 2.5xGrid |    2114880 | 01:17:34 | 01:19:21 | 01:22:38 | 01:35:02 
#  10 layers 3xGrid   |    2680020 | 01:41:14 | 01:51:47 | 02:06:10m| 01:51:46
#  10 layers 3.5xGrid |    3992850 |    /     |          |          | 
#  10 layers DivGrid  |    2235810 | 00:57:36 | 01:01:16 | 01:01:15 | 01:00:03
#  10 layers DivGrid2 |    2613600 | 01:14:20 | 01:14:20 | 01:14:11 | 01:13:47
#  10 layers DivGrid3 |    2612400 | 01:07:32 | 01:12:31 | 01:12:51 | 01:12:18 
#
#
# FEniCS 1.3.0 Assimilation run times (16 cores, 20 function evals):
# ==================================================================
#
#  mesh type          | # elements | run 1    | run 2    | run 3    | run 4    
#  -------------------+------------+----------+----------+----------+----------
#  mesh_high.xml      |    3747930 | 02:13:59 | 02:14:34 | 02:12:27 |
#
#
# Assimilation run times (8 cores, 20 function evals):
# =====================================================
#
#  mesh type          | # elements | run 1    | run 2    | run 3    | run 4    
#  -------------------+------------+----------+----------+----------+----------
#  10 layers crude    |     370740 |  |  |  |  
#

import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

import src.solvers            as solvers
import src.physical_constants as pc
import src.model              as model
from meshes.mesh_factory  import MeshFactory
from data.data_factory    import DataFactory
from src.helper           import default_nonlin_solver_params, component_stress
from src.utilities        import DataInput, DataOutput
from dolfin               import *
from pylab                import sqrt, copy
from time                 import time

set_log_active(True)
#set_log_level(PROGRESS)

thklim = 200.0

# collect the raw data :
searise  = DataFactory.get_searise(thklim = thklim)
measure  = DataFactory.get_gre_measures()
meas_shf = DataFactory.get_shift_gre_measures()
bamber   = DataFactory.get_bamber(thklim = thklim)
fm_qgeo  = DataFactory.get_gre_qgeo_fox_maule()
sec_qgeo = DataFactory.get_gre_qgeo_secret()

# define the meshes :
mesh      = Mesh('meshes/mesh_high_new.xml')
flat_mesh = Mesh('meshes/mesh_high_new.xml')
mesh.coordinates()[:,2]      /= 100000.0
flat_mesh.coordinates()[:,2] /= 100000.0


# create data objects to use with varglas :
dsr     = DataInput(None, searise,  mesh=mesh)
dbm     = DataInput(None, bamber,   mesh=mesh)
#dms     = DataInput(None, measure,  mesh=mesh, flip=True)
#dmss    = DataInput(None, meas_shf, mesh=mesh, flip=True)
dfm     = DataInput(None, fm_qgeo,  mesh=mesh)
dsq     = DataInput(None, sec_qgeo, mesh=mesh)
#dbv     = DataInput("results/", ("Ubmag_measures.mat", "Ubmag.mat"), mesh=mesh)

# change the projection of the measures data to fit with other data :
#dms.change_projection(dsr)

# get the expressions used by varglas :
Thickness          = dbm.get_spline_expression('H')
Surface            = dbm.get_spline_expression('h')
Bed                = dbm.get_spline_expression('b')
SurfaceTemperature = dsr.get_spline_expression('T')
#BasalHeatFlux      = dsr.get_spline_expression('q_geo')
BasalHeatFlux      = dsq.get_spline_expression('q_geo')
#BasalHeatFlux      = dfm.get_spline_expression('q_geo')
adot               = dsr.get_spline_expression('adot')
U_observed         = dsr.get_spline_expression('U_ob')

# inspect the data values :
do    = DataOutput('results_pre/')
model = model.Model()
model.set_geometry(Surface, Bed)
model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()
#do.write_one_file('ff',             model.ff)
#do.write_one_file('h',              dbm.get_projection('h'))
#do.write_one_file('Ubmag_measures', dbv.get_projection('Ubmag_measures'))
#do.write_one_file('sq_qgeo',        dsq.get_projection('q_geo'))
#do.write_one_file('sr_qgeo',        dsr.get_projection('q_geo'))
#exit(0)


# specifify non-linear solver parameters :
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 1.0
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-14
nonlin_solver_params['newton_solver']['absolute_tolerance']      = 1e2
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver']                            = 'mumps'
nonlin_solver_params['preconditioner']                           = 'default'
#nonlin_solver_params['newton_solver']['linear_solver']           = 'mumps'
#nonlin_solver_params['newton_solver']['preconditioner']          = 'default'
parameters['form_compiler']['quadrature_degree']                 = 2

# make the directory if needed :
i = int(sys.argv[1])
#dir_b   = './results_sr/0'
#dir_b   = './results_fm/0'
dir_b   = './results_new/0'

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
             'on'               : True,
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
             'alpha'               : [Thickness**2/10**2],
             'beta'                : 0.0,
             'max_fun'             : 20,
             'objective_function'  : 'logarithmic',
             'bounds'              : [(0,20)],
             'control_variable'    : None,
             'regularization_type' : 'Tikhonov'
           }}

model.eps_reg = 1e-5
#config['adjoint']['alpha'] = model.S - model.B

F = solvers.SteadySolver(model,config)
if i != 0: File(dir_b + str(i-1) + '/beta2_opt.xml') >> model.beta2
t01 = time()
F.solve()
tf1 = time()
if i != 0: model.adot = adot

visc    = project(model.eta, model.Q)
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

File(out_dir + 'Mb.pvd')      << model.Mb
File(out_dir + 'S.pvd')       << model.S
File(out_dir + 'B.pvd')       << model.B
File(out_dir + 'u.pvd')       << model.u
File(out_dir + 'v.pvd')       << model.v
File(out_dir + 'w.pvd')       << model.w

tau_lon, tau_lat, tau_bas, tau_drv = component_stress(model)
#tau_tot = project(tau_lon + tau_lat + tau_bas - tau_drv)

File(out_dir + 'tau_lon.pvd') << tau_lon 
File(out_dir + 'tau_lat.pvd') << tau_lat 
File(out_dir + 'tau_bas.pvd') << tau_bas
File(out_dir + 'tau_drv.pvd') << tau_drv
#File(out_dir + 'tau_tot.pvd') << tau_tot 
#File(out_dir + 'mesh.xdmf')   << model.mesh

# functionality of HDF5 not completed by fenics devs :
#f = HDF5File(out_dir + 'u.h5', 'w')
#f.write(model.mesh,  'mesh')
#f.write(model.beta2, 'beta2')
#f.write(model.Mb,    'Mb')
#f.write(model.T,     'T')

# calculate total time to compute
s = (tf1 - t01) + (tf2 - t02)
m = s / 60.0
h = m / 60.0
s = s % 60
m = m % 60
print "Total time to compute: \r%02d:%02d:%02d" % (h,m,s)

