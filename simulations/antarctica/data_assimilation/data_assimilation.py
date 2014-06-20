import os
import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model              as model
import src.solvers            as solvers
import src.physical_constants as pc
from data.data_factory   import DataFactory
from meshes.mesh_factory import MeshFactory
from src.helper          import *
from src.utilities       import DataInput
from dolfin              import *


# get the input args :
i = int(sys.argv[2])           # assimilation number
dir_b = sys.argv[1] + '/0'     # directory to save

# set the output directory :
out_dir = dir_b + str(i) + '/'

set_log_active(True)
#set_log_level(PROGRESS)

thklim = 200.0

measures  = DataFactory.get_ant_measures(res=900)
bedmap1   = DataFactory.get_bedmap1(thklim=thklim)
bedmap2   = DataFactory.get_bedmap2(thklim=thklim)

mesh = MeshFactory.get_antarctica_coarse()

dm  = DataInput(None, measures, mesh=mesh)
db1 = DataInput(None, bedmap1,  mesh=mesh)
db2 = DataInput(None, bedmap2,  mesh=mesh)

db2.set_data_val('H',   32767, thklim)
db2.set_data_val('S',   32767, 0.0)
dm.set_data_min('U_ob', 0.0,   0.0)

Thickness          = db2.get_spline_expression("H")
Surface            = db2.get_spline_expression("S")
Bed                = db2.get_spline_expression("B")
Mask               = db2.get_nearest_expression("mask")
SurfaceTemperature = db1.get_spline_expression("srfTemp")
BasalHeatFlux      = db1.get_spline_expression("q_geo")
adot               = db1.get_spline_expression("adot")
U_observed         = dm.get_spline_expression("U_ob")

model = model.Model()
model.set_mesh(mesh)
model.set_geometry(Surface, Bed, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()

# specifify non-linear solver parameters :
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.7
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-6
nonlin_solver_params['newton_solver']['absolute_tolerance']      = 1e2
nonlin_solver_params['newton_solver']['maximum_iterations']      = 25
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['newton_solver']['linear_solver']           = 'gmres'
nonlin_solver_params['newton_solver']['preconditioner']          = 'hypre_amg'
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
             'on'                  : True,
             'inner_tol'           : 0.0,
             'max_iter'            : 5
           },
           'velocity' : 
           { 
             'on'                  : True,
             'newton_params'       : nonlin_solver_params,
             'viscosity_mode'      : 'full',
             'b_linear'            : None,
             'use_T0'              : True,
             'T0'                  : 268.0,
             'A0'                  : 1e-16,
             'beta2'               : 2.0,
             'r'                   : 1.0,
             'E'                   : 1.0,
             'approximation'       : 'fo',
             'boundaries'          : None
           },
           'enthalpy' : 
           { 
             'on'                  : True,
             'use_surface_climate' : False,
             'T_surface'           : SurfaceTemperature,
             'q_geo'               : 0.042 * 60 * 60 * 24 * 365,#BasalHeatFlux,
             'lateral_boundaries'  : None
           },
           'free_surface' :
           { 
             'on'                  : False,
             'lump_mass_matrix'    : True,
             'thklim'              : thklim,
             'use_pdd'             : False,
             'observed_smb'        : adot,
           },  
           'age' : 
           { 
             'on'                  : False,
             'use_smb_for_ela'     : False,
             'ela'                 : None,
           },
           'surface_climate' : 
           { 
             'on'                  : False,
             'T_ma'                : None,
             'T_ju'                : None,
             'beta_w'              : None,
             'sigma'               : None,
             'precip'              : None
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

model.eps_reg = 1e-15

F = solvers.SteadySolver(model,config)
if i != 0: 
  File(dir_b + str(i-1) + '/beta2.xml') >> model.beta2
  config['velocity']['approximation'] = 'stokes'
t01 = time()
F.solve()
tf1 = time()

params = config['velocity']['newton_params']['newton_solver']
params['relaxation_parameter']         = 1.0
params['relative_tolerance']           = 1e-6
params['absolute_tolerance']           = 0.0
params['maximum_iterations']           = 12
config['velocity']['viscosity_mode']   = 'linear'
config['velocity']['b_linear']         = model.eta
config['enthalpy']['on']               = False
config['surface_climate']['on']        = False
config['coupled']['on']                = False
config['velocity']['use_T0']           = False
config['adjoint']['control_variable']  = [model.beta2]

A = solvers.AdjointSolver(model,config)
A.set_target_velocity(U = U_observed)
if i != 0: File(dir_b + str(i-1) + '/beta2.xml') >> model.beta2
t02 = time()
A.solve()
tf2 = time()
    
File(out_dir + 'S.xml')       << model.S
File(out_dir + 'B.xml')       << model.B
File(out_dir + 'u.xml')       << project(model.u, model.Q)
File(out_dir + 'v.xml')       << project(model.v, model.Q)
File(out_dir + 'w.xml')       << project(model.w, model.Q)
File(out_dir + 'beta2.xml')   << model.beta2
File(out_dir + 'eta.xml')     << project(model.eta, model.Q)

#XDMFFile(mesh.mpi_comm(), out_dir + 'mesh.xdmf')   << model.mesh
#
## save the state of the model :
#if i !=0: rw = 'a'
#else:     rw = 'w'
#f = HDF5File(out_dir + '3D_5H_stokes.h5', rw)
#f.write(model.mesh,  'mesh')
#f.write(model.beta2, 'beta2')
#f.write(model.Mb,    'Mb')
#f.write(model.T,     'T')
#f.write(model.S,     'S')
#f.write(model.B,     'B')
#f.write(model.U,     'U')
#f.write(model.eta,   'eta')

# calculate total time to compute
s = (tf1 - t01) + (tf2 - t02)
m = s / 60.0
h = m / 60.0
s = s % 60
m = m % 60
print "Total time to compute: \r%02d:%02d:%02d" % (h,m,s)

