import sys
import varglas.solvers            as solvers
import varglas.physical_constants as pc
import varglas.model              as model
from varglas.mesh.mesh_factory    import MeshFactory
from varglas.data.data_factory    import DataFactory
from varglas.helper               import default_nonlin_solver_params
from varglas.utilities            import DataInput, DataOutput
from fenics                       import *
from time                         import time


# get the input args :
i = int(sys.argv[2])           # assimilation number
dir_b = sys.argv[1] + '/0'     # directory to save

# set the output directory :
out_dir = dir_b + str(i) + '/'

set_log_active(True)
#set_log_level(PROGRESS)

thklim = 200.0

# collect the raw data :
searise  = DataFactory.get_searise(thklim = thklim)
bamber   = DataFactory.get_bamber(thklim = thklim)
fm_qgeo  = DataFactory.get_gre_qgeo_fox_maule()
#measure  = DataFactory.get_gre_measures()
rignot   = DataFactory.get_gre_rignot()

# define the mesh :
mesh = MeshFactory.get_greenland_coarse()

# create data objects to use with varglas :
dsr     = DataInput(None, searise,  mesh=mesh)
dbm     = DataInput(None, bamber,   mesh=mesh)
#dms     = DataInput(None, measure,  mesh=mesh)
dfm     = DataInput(None, fm_qgeo,  mesh=mesh)
drg     = DataInput(None, rignot,   mesh=mesh)

# change the projection of the measures data to fit with other data :
#dms.change_projection(dsr)
drg.change_projection(dsr)

# get the expressions used by varglas :
Thickness          = dbm.get_spline_expression('H')
Surface            = dbm.get_spline_expression('S')
Bed                = dbm.get_spline_expression('B')
SurfaceTemperature = dsr.get_spline_expression('T')
#BasalHeatFlux      = dsr.get_spline_expression('q_geo')
BasalHeatFlux      = dfm.get_spline_expression('q_geo')
adot               = dsr.get_spline_expression('adot')
#U_observed         = dsr.get_spline_expression('U_ob')
U_observed         = drg.get_spline_expression('U_ob')

# inspect the data values :
#do    = DataOutput('results_pre/')
#do.write_one_file('vmag',           drg.get_projection('U_ob'))
#do.write_one_file('h',              dbm.get_projection('H'))
#do.write_one_file('Ubmag_measures', dbv.get_projection('Ubmag_measures'))
#do.write_one_file('sr_qgeo',        dsr.get_projection('q_geo'))
#exit(0)

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
             'A0'                  : None,
             'beta2'               : 0.5,
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
             'q_geo'               : BasalHeatFlux,
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

