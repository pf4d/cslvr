import sys
src_directory = '../../../'
sys.path.append(src_directory)

import src.model
import src.solvers
import src.physical_constants
import src.helper
import pylab
import dolfin

dolfin.set_log_active(True)

alpha = pylab.deg2rad(0.1)
L=80000

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
        values[0] = 1000 + 1000.*pylab.sin(2*pylab.pi*x[0]/L)*pylab.sin(2*pylab.pi*x[1]/L)

nparams = src.helper.default_nonlin_solver_params()
nparams['linear_solver'] = 'gmres'
nparams['preconditioner'] = 'hypre_amg'
nparams['newton_solver']['relaxation_parameter']=0.7
nparams['newton_solver']['maximum_iterations']=20
nparams['newton_solver']['error_on_nonconvergence']=False

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
                'newton_params' : nparams,
                'viscosity_mode' : 'isothermal',
                'b_linear' : None,
                'use_T0': False,
                'T0' : None,
                'A0' : 1e-16,
                'beta2' : Beta2(),
                'r' : 0.0,
                'E' : 1,
                'approximation' : 'fo',
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
            { 'alpha' : [1e4],
                'beta' : 0.0,
                'max_fun' : 50,
                'objective_function' : 'linear',
                'animate' : False,
                'bounds' : None,
                'control_variable' : None,
                'regularization_type' : 'Tikhonov'
            },
            'output_path' : './results/',
            'wall_markers' : [],
            'periodic_boundary_conditions' : True,
            'log': True }

model = src.model.Model()
model.set_geometry(Surface(), Bed())

nx = ny = 40
nz = 7

model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, ymin=0, ymax=L,
                            generate_pbcs=True)
model.set_parameters(src.physical_constants.IceParameters())
model.initialize_variables()

F = src.solvers.SteadySolver(model,config)
F.solve()

model.eps_reg = 1e-5
config['adjoint']['control_variable'] = [model.beta2]
config['adjoint']['bounds'] = [(0.0,5000.0)]
dolfin.File('results/beta2_obs.xml') << model.beta2

A = src.solvers.AdjointSolver(model,config)
model.beta2.vector()[:] = 1000.

U_e = 10.0
from scipy import random
u_error = U_e*random.randn(len(model.u_o.vector().get_local()))
v_error = U_e*random.randn(len(model.u_o.vector().get_local()))
model.u_o.vector().set_local(model.u.vector().get_local()+u_error)
model.v_o.vector().set_local(model.v.vector().get_local()+v_error)
dolfin.File('results/U_obs.xml') << dolfin.project(dolfin.as_vector([model.u_o,model.v_o]))

A.solve()
