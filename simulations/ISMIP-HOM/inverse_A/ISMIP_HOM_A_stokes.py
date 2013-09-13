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

alpha = pylab.deg2rad(0.5)
lengths = [40000]
for L in lengths:

    class Surface(dolfin.Expression):
        def __init__(self):
            pass
        def eval(self,values,x):
            values[0] = -x[0]*pylab.tan(alpha)

	
    class Bed(dolfin.Expression):
        def __init__(self):
            pass
        def eval(self,values,x):
            values[0] = -x[0]*pylab.tan(alpha) - 1000.0 + 500.0*pylab.sin(2*pylab.pi*x[0]/L)*pylab.sin(2*pylab.pi*x[1]/L)

    nonlin_solver_params = default_nonlin_solver_params()
    nonlin_solver_params['newton_solver']['relaxation_parameter'] = 0.7
    nonlin_solver_params['linear_solver'] = 'mumps'

    config = { 'mode' : 'steady',
            'coupled' : 
                { 'on' : False,
                    'inner_tol': 0.0,
                    'max_iter' : 1
                },
            't_start' : None,
            't_end' : None,
            'time_step' : None,
            'velocity' : 
                { 'on' : True,
                    'newton_params' : nonlin_solver_params,
                    'viscosity_mode' : 'isothermal',
                    'b_linear' : None,
                    'use_T0': False,
                    'T0' : None,
                    'A0' : 1e-16,
                    'beta2' : 2.0,
                    'r' : 1.0,
                    'E' : 1,
                    'approximation' : 'stokes',
                    'boundaries' : None
                },
            'enthalpy' : 
                { 'on': False,
                    'use_surface_climate': False,
                    'T_surface' : None,
                    
                },
            'free_surface' :
                { 'on': False,
                    'thklim': None,
                    'use_pdd': False,
                    'observed_smb': None,
                },  
            'age' : 
                { 'on': False,
                    'use_smb_for_ela': False,
                    'ela': None,
                },
                'surface_climate' : 
                { 'on': False,
                    'T_ma': None,
                    'T_ju': None,
                    'beta_w': None,
                    'sigma': None,
                    'precip': None
                },
                'adjoint' :
                { 'alpha' : 0.0,
                    'beta' : 0.0,
                    'max_fun' : 100,
                    'objective_function' : 'kinematic',
                    'animate' : False,
                    'bounds' : [(0,5000.0)]
                },
                'output_path' : './results_stokes/',
                'wall_markers' : [],
                'periodic_boundary_conditions' : True,
                'log': True }

    model = src.model.Model()
    model.set_geometry(Surface(), Bed())

    nx = 20
    ny = 20
    nz = 6

    model.generate_uniform_mesh(nx,ny,nz,xmin=0,xmax=L,ymin=0,ymax=L,generate_pbcs=True)

    model.set_parameters(src.physical_constants.IceParameters())
    model.initialize_variables()

    F = src.solvers.SteadySolver(model,config)
    F.solve()
    model.eps_reg = 1e-5

    config['adjoint']['control_variable'] = [model.beta2]

    A = src.solvers.AdjointSolver(model,config)
    #model.beta2.vector()[:] = 1000.
    model.u_o.vector().set_local(model.u.vector().get_local())
    model.v_o.vector().set_local(model.v.vector().get_local())
    A.solve()

