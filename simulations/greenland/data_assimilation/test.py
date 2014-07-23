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
out_dir = './test_results/'

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
#mesh      = Mesh('meshes/mesh_low.xml')
#flat_mesh = Mesh('meshes/mesh_low.xml')
mesh.coordinates()[:,2]      /= 100000.0
flat_mesh.coordinates()[:,2] /= 100000.0


# create data objects to use with varglas :
dsr     = DataInput(searise,  mesh=mesh)
dbm     = DataInput(bamber,   mesh=mesh)
#dms     = DataInput(measure,  mesh=mesh)
#dmss    = DataInput(meas_shf, mesh=mesh)
dfm     = DataInput(fm_qgeo,  mesh=mesh)
dsq     = DataInput(sec_qgeo, mesh=mesh)
#dbv     = DataInput(("Ubmag_measures.mat", "Ubmag.mat"), 
#                     direc = "results/", 
#                     mesh=mesh)

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
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.7
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['absolute_tolerance']      = 1e2
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
#nonlin_solver_params['linear_solver']                            = 'mumps'
#nonlin_solver_params['preconditioner']                           = 'default'
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

File(out_dir + 'S.xml')       << model.S
File(out_dir + 'B.xml')       << model.B
File(out_dir + 'u.xml')       << model.u
File(out_dir + 'v.xml')       << model.v
File(out_dir + 'w.xml')       << model.w
File(out_dir + 'beta2.xml')   << model.beta2

tau_lon, tau_lat, tau_bas, tau_drv, beta22 = model.component_stress()
tau_tot       = project(tau_lon + tau_lat + tau_bas - tau_drv)
tau_drv_m_bas = project(tau_drv - tau_bas)
tau_lat_p_lon = project(tau_lat + tau_lon)
tau_bas2      = project(tau_drv - tau_lon - tau_lat)
tau_drv2      = project(tau_bas + tau_lon + tau_lat)
intDivU       = project(model.vert_integrate(div(model.U)))

File(out_dir + 'tau_lon.pvd')       << tau_lon
File(out_dir + 'tau_lat.pvd')       << tau_lat
File(out_dir + 'tau_bas.pvd')       << tau_bas
File(out_dir + 'tau_drv.pvd')       << tau_drv
File(out_dir + 'tau_tot.pvd')       << tau_tot
File(out_dir + 'tau_lat_p_lon.pvd') << tau_lat_p_lon
File(out_dir + 'tau_drv_m_bas.pvd') << tau_drv_m_bas
File(out_dir + 'tau_bas2.pvd')      << tau_bas2
File(out_dir + 'tau_drv2.pvd')      << tau_drv2
File(out_dir + 'beta22.pvd')        << beta22
File(out_dir + 'intDivU.pvd')       << intDivU

# calculate total time to compute
s = tf1 - t01
m = s / 60.0
h = m / 60.0
s = s % 60
m = m % 60
print "Total time to compute: \r%02d:%02d:%02d" % (h,m,s)

