import sys
src_directory = '../../../'
sys.path.append(src_directory)
import src.model
import src.helper
import src.solvers
import src.physical_constants
import pylab
import dolfin

dolfin.set_log_active(True)

L = 750000.0
S_0 = 10.0
S_b = 1e-5
R_el = 450000.0
M_max = 0.5

T_min = 238.15
S_T = 1.67e-5

class Surface(dolfin.Expression):
    def eval(self,values,x):
        values[0] = S_0

class Bed(dolfin.Expression):
    def eval(self,values,x):
        values[0] = 0.0

class MassBalance(dolfin.Expression):
    def eval(self,values,x):
        values[0] = min(M_max,S_b*(R_el-pylab.sqrt(x[0]**2 + x[1]**2))) 

class SurfaceTemperature(dolfin.Expression):
    def eval(self,values,x):
        values[0] =  T_min + S_T*pylab.sqrt(x[0]**2 + x[1]**2)

nonlin_solver_params = src.helper.default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 1.0
nonlin_solver_params['newton_solver']['absolute_tolerance'] = 1.0
nonlin_solver_params['linear_solver'] = 'gmres'
nonlin_solver_params['preconditioner'] = 'hypre_amg'


config = { 'mode' : 'steady',
           'coupled' : 
               { 'on' : False,
                 'inner_tol': 0.0,
                 'max_iter' : 1
               },
           't_start' : 0.0,
           't_end' : 50000.0,
           'time_step' : 10.0,
           'velocity' : 
               { 'on' : True,
                 'newton_params' : nonlin_solver_params,
                 'viscosity_mode' : 'full',
                 'b_linear' : None,
                 'use_T0': True,
                 'T0' : 268.0,
                 'A0' : 1e-16,
                 'beta2' : 1e5,
                 'r' : 1.0,
                 'E' : 1.0,
                 'approximation' : 'fo',
                 'boundaries' : None
               },
           'enthalpy' : 
               { 'on': True,
                 'use_surface_climate': False,
                 'T_surface' : SurfaceTemperature(),
                 'q_geo' : 0.042*60**2*24*365,
                 'lateral_boundaries':None
                 
               },
           'free_surface' :
               { 'on': True,
                 'lump_mass_matrix': False,
                 'use_shock_capturing':True,
                 'thklim': 10.0,
                 'use_pdd': False,
                 'observed_smb': MassBalance(),
                 'static_boundary_conditions':False
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
               { 'alpha' : None,
                 'beta' : None,
                 'max_fun' : None,
                 'objective_function' : 'logarithmic',
                 'animate' : False
               },
            'output_path' : './results/',
            'wall_markers' : [],
            'periodic_boundary_conditions' : False,
            'log': True }

model = src.model.Model()
model.set_geometry(Surface(), Bed())

mesh      = dolfin.Mesh('../meshes/circle.xml')
flat_mesh = dolfin.Mesh('../meshes/circle.xml')
model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.mesh.coordinates()[:,2] = model.mesh.coordinates()[:,2]/1000.0
model.set_parameters(src.physical_constants.IceParameters())
model.initialize_variables()

F = src.solvers.SteadySolver(model,config)
F.solve()

T = src.solvers.TransientSolver(model,config)
T.solve()

dolfin.File('./results/u.xml') << model.u
dolfin.File('./results/v.xml') << model.v
dolfin.File('./results/w.xml') << model.w
dolfin.File('./results/S.xml') << model.S
dolfin.File('./results/T.xml') << model.T
