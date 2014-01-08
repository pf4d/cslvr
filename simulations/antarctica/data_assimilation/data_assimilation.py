# Assimilation run times (8 cores, 20 function evals):
# =====================================================
#
#  mesh type          | # elements | run 1    | run 2    | run 3    | run 4    
#  -------------------+------------+----------+----------+----------+----------
#  10 layers medium   |    1669740 |  |  |  |  
#  10 layers crude    |     624210 |  |  |  |  
#
import os
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

thklim = 50.0

measures  = DataFactory.get_ant_measures(resolution=450)
bedmap1   = DataFactory.get_bedmap1(thklim=thklim)
bedmap2   = DataFactory.get_bedmap2(thklim=thklim)

mesh      = Mesh('meshes/3dmesh_medium.xml')
flat_mesh = Mesh('meshes/3dmesh_medium.xml')

dm  = DataInput(None, measures, mesh=mesh)
db1 = DataInput(None, bedmap1,  mesh=mesh)
db2 = DataInput(None, bedmap2,  mesh=mesh)

db2.set_data_min("H", thklim, thklim)
db2.set_data_min("h", 0.0,0.0)

db2.set_data_max("H",30000.,thklim)
db2.set_data_max("h",30000.,0.0)

db2.data['b'] = db2.data['h']-db2.data['H']

print "getting expressions"
Surface            = db2.get_spline_expression("h_n")
Bed                = db2.get_spline_expression("b")
SurfaceTemperature = db1.get_spline_expression("srfTemp")
BasalHeatFlux      = db1.get_spline_expression("q_geo")
U_observed         = dm.get_spline_expression("v_mag")
print "done"

#===============================================================================
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.5
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver']                            = 'mumps'
nonlin_solver_params['preconditioner']                           = 'default'

# make the directory if needed :
i = int(sys.argv[1])
dir_b   = './results/0'

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
             'on'       : False,
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
             'A0'             : 1e-16,
             'beta2'          : 2.0,
             'r'              : 1.0,
             'E'              : 1.0,
             'approximation'  : 'fo',
             'boundaries'     : None
           },
           'enthalpy' : 
           { 
             'on'                  : False,
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
             'alpha'              : 1e3,
             'beta'               : 0.0,
             'max_fun'            : 50,
             'objective_function' : 'logarithmic',
             'bounds'             : (0.,20.)
           }}

model = model.Model()
model.set_geometry(Surface,Bed)

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

# functionality of HDF5 not completed by fenics devs :
#f = HDF5File(dir_b + str(i) + '/u.h5', 'w')
#f.write(model.mesh,  'mesh')
#f.write(model.beta2, 'beta2')
#f.write(model.Mb,    'Mb')
#f.write(model.T,     'T')

File(dir_b + str(i) + '/Mb.pvd')    << model.Mb
File(dir_b + str(i) + '/mesh.xdmf') << model.mesh

# calculate total time to compute
s = (tf1 - t01) + (tf2 - t02)
m = s / 60.0
h = m / 60.0
s = s % 60
m = m % 60
print "Total time to compute: \r%02d:%02d:%02d" % (h,m,s)

