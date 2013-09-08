import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model              as model
import src.solvers            as solvers
import src.physical_constants as pc
import scipy.io
from data.data_factory import DataFactory
from src.helper        import default_nonlin_solver_params
from src.utilities     import DataInput, DataOutput
from dolfin            import *
from pylab             import sqrt, copy

set_log_active(True)

# collect the raw data :
searise  = DataFactory.get_searise()
measure  = DataFactory.get_gre_measures()
meas_shf = DataFactory.get_shift_gre_measures()
bamber   = DataFactory.get_bamber()

# define the meshes :
mesh      = Mesh('../meshes/mesh.xml')
flat_mesh = Mesh('../meshes/mesh.xml')

# create data objects to use with varglas :
dsr     = DataInput(None, searise,  mesh=mesh, create_proj=True)
dbm     = DataInput(None, bamber,   mesh=mesh)
dms     = DataInput(None, measure,  mesh=mesh, create_proj=True, flip=True)
dmss    = DataInput(None, meas_shf, mesh=mesh, flip=True)
dbv     = DataInput("results/", ("Ubmag_measures.mat",), mesh=mesh)

# change the projection of the measures data to fit with other data :
dms.change_projection(dsr)

# set the maximum ice thickness :
dbm.set_data_min('H', 10.0, 10.0)

# set value of bed to correspond with thickness change :
dbm.data['b'] = dbm.data['h'] - dbm.data['H']

# get the expressions used by varglas :
Surface            = dbm.get_spline_expression('h')
Bed                = dbm.get_spline_expression('b')
SurfaceTemperature = dsr.get_spline_expression('T')
BasalHeatFlux      = dsr.get_spline_expression('q_geo')
U_observed         = dbv.get_spline_expression('Ubmag_measures')

# specifify non-linear solver parameters :
nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter']    = 0.7
nonlin_solver_params['newton_solver']['relative_tolerance']      = 1e-3
nonlin_solver_params['newton_solver']['maximum_iterations']      = 20
nonlin_solver_params['newton_solver']['error_on_nonconvergence'] = False
nonlin_solver_params['linear_solver']                            = 'mumps'
nonlin_solver_params['preconditioner']                           = 'default'

config = { 'mode'                         : 'steady',
           't_start'                      : None,
           't_end'                        : None,
           'time_step'                    : None,
           'output_path'                  : './results_coarse/',
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
             'beta2'          : 10.0,
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
             'max_fun'            : 20,
             'objective_function' : 'logarithmic',
             'bounds'             : (0,20)
           }}



model = model.Model()
model.set_geometry(Surface, Bed)

model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()

F = solvers.SteadySolver(model,config)
#File('./results/beta2_opt.xml') >> model.beta2
F.solve()

visc    = project(model.eta)
vel_par = config['velocity']
vel_par['viscosity_mode']                                         = 'linear'
vel_par['b_linear']                                               = visc
vel_par['newton_params']['newton_solver']['relaxation_parameter'] = 1.0

config['enthalpy']['on']        = False
config['surface_climate']['on'] = False
config['coupled']['on']         = False
config['velocity']['use_T0']    = False

A = solvers.AdjointSolver(model,config)
A.set_target_velocity(U = U_observed)
#File('./results/beta2_opt.xml') >> model.beta2
A.solve()

#u = model.u.vector().get_local()
#v = model.v.vector().get_local()
#u[abs(u)>5000.0] = 5000.0
#v[abs(v)>5000.0] = 5000.0
#model.u.vector().set_local(u)
#model.v.vector().set_local(v)

#u = model.u_o.vector().get_local()
#v = model.v_o.vector().get_local()
#u[abs(u)>5000.0] = 5000.0
#v[abs(v)>5000.0] = 5000.0
#model.u_o.vector().set_local(u)
#model.v_o.vector().set_local(v)

#unit = Function(model.Q)
#unit.vector()[:] = 1.0
#RMS = sqrt(assemble(((model.u-model.u_o)**2 + (model.v-model.v_o)**2)*ds(2))/assemble(unit*ds(2)))
#print RMS


