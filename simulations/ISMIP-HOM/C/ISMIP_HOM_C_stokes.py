import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model
import src.solvers
import src.physical_constants
from src.helper import default_nonlin_solver_params
import pylab
import dolfin

dolfin.set_log_active(True)

alpha = pylab.deg2rad(0.1)
for L in [40000]:

    class Surface(dolfin.Expression):
        def __init__(self):
            pass
        def eval(self,values,x):
            values[0] = -x[0]*pylab.tan(alpha)

    class Bed(dolfin.Expression):
        def __init__(self):
            pass
        def eval(self,values,x):
            values[0] = -x[0]*pylab.tan(alpha) - 1000.0

    class Beta2(dolfin.Expression):
        def __init__(self):
            pass
        def eval(self,values,x):
            values[0] = + 1000 \
                        + 1000. * pylab.sin(2*pylab.pi*x[0]/L) \
                                * pylab.sin(2*pylab.pi*x[1]/L)

    nonlin_params = src.helper.default_nonlin_solver_params()
    
    config = { 'mode'                         : 'steady',
               'output_path'                  : './results_stokes/',
               'wall_markers'                 : [],
               'periodic_boundary_conditions' : True,
               't_start'                      : None,
               't_end'                        : None,
               'time_step'                    : None,
               'log'                          : True,
               'coupled' : 
               { 
                 'on'        : False,
                 'inner_tol' : 0.0,
                 'max_iter'  : 1
               },
               'velocity' : 
               { 
                 'on'             : True,
                 'newton_params'  : nonlin_params,
                 'viscosity_mode' : 'isothermal',
                 'b_linear'       : None,
                 'use_T0'         : False,
                 'T0'             : None,
                 'A0'             : 1e-16,
                 'beta2'          : Beta2(),
                 'r'              : 0.0,
                 'E'              : 1,
                 'approximation'  : 'stokes',
                 'boundaries'     : None
               },
               'enthalpy' : 
               { 
                 'on'                  : False,
                 'use_surface_climate' : False,
                 'T_surface'           : None,
               },
               'free_surface' :
               { 
                 'on'           : False,
                 'thklim'       : None,
                 'use_pdd'      : False,
                 'observed_smb' : None,
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
                 'alpha'              : None,
                 'beta'               : None,
                 'max_fun'            : None,
                 'objective_function' : 'logarithmic',
                 'animate'            : False
               }}

    model = src.model.Model()
    model.set_geometry(Surface(), Bed())
 
    nx = 25
    ny = 25
    nz = 7

    model.generate_uniform_mesh(nx,ny,nz,xmin=0,xmax=L,ymin=0,ymax=L,
                                generate_pbcs=True)
    model.set_parameters(src.physical_constants.IceParameters())
    model.initialize_variables()

    F = src.solvers.SteadySolver(model,config)
    F.solve()

