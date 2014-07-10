from varglas.model              import Model
from varglas.solvers            import SteadySolver, TransientSolver
from varglas.physical_constants import IceParameters
from varglas.helper             import default_nonlin_solver_params
from scipy                      import random
from varglas.mesh.mesh_factory  import MeshFactory
from fenics                     import Expression, sqrt, set_log_active, \
                                       parameters, Constant, File

set_log_active(True)

L     = 750000.0
S_0   = 10.0
S_b   = 1e-5
R_el  = 450000.0
M_max = 0.5
T_min = 238.15
S_T   = 1.67e-5

mesh  = MeshFactory.get_circle()

model = Model()
model.set_mesh(mesh)

class MassBalance(Expression):
  def eval(self,values,x):
    values[0] = min(M_max, S_b*(R_el - sqrt(x[0]**2 + x[1]**2)))

Surface = Expression('S_0', S_0=S_0, element=model.Q.ufl_element())
Bed     = Expression('0.0', element=model.Q.ufl_element())
T_s     = Expression('T_min + S_T*sqrt(pow(x[0],2) + pow(x[1],2))',
                      T_min=T_min, S_T=S_T, element=model.Q.ufl_element())
SMB     = MassBalance(element=model.Q.ufl_element())

model.set_geometry(Surface, Bed, deform=True)

model.set_parameters(IceParameters())
model.calculate_boundaries()
model.initialize_variables()

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 1.0
nonlin_solver_params['newton_solver']['absolute_tolerance']   = 1.0
nonlin_solver_params['newton_solver']['linear_solver']        = 'mumps'
nonlin_solver_params['newton_solver']['preconditioner']       = 'default'
parameters['form_compiler']['quadrature_degree']              = 2


config = { 'mode'                         : 'steady',
           't_start'                      : 0.0,
           't_end'                        : 50000.0,
           'time_step'                    : 10.0,
           'output_path'                  : './results/',
           'wall_markers'                 : [],
           'periodic_boundary_conditions' : False,
           'log'                          : True,
           'velocity' :                   
           {                              
             'on'                         : True,
             'newton_params'              : nonlin_solver_params,
             'viscosity_mode'             : 'full',
             'b_linear'                   : None,
             'use_T0'                     : True,
             'T0'                         : 268.0,
             'A0'                         : 1e-16,
             'beta2'                      : 1e5,
             'r'                          : 1.0,
             'E'                          : 1.0,
             'approximation'              : 'fo',
             'boundaries'                 : None
           },
           'coupled' : 
           { 
             'on'                         : True,
             'inner_tol'                  : 0.0,
             'max_iter'                   : 5
           },                             
           'enthalpy' : 
           { 
             'on'                         : True,
             'use_surface_climate'        : False,
             'T_surface'                  : T_s,
             'q_geo'                      : 0.042*60**2*24*365,
             'lateral_boundaries'         : None
           },
           'free_surface' :
           { 
             'on'                         : True,
             'lump_mass_matrix'           : False,
             'use_shock_capturing'        : True,
             'thklim'                     : 10.0,
             'use_pdd'                    : False,
             'observed_smb'               : SMB,
             'static_boundary_conditions' : False
           },  
           'age' : 
           { 
             'on'                         : False,
             'use_smb_for_ela'            : False,
             'ela'                        : None,
           },                             
           'surface_climate' :            
           {                              
             'on'                         : False,
             'T_ma'                       : None,
             'T_ju'                       : None,
             'beta_w'                     : None,
             'sigma'                      : None,
             'precip'                     : None
           },                             
           'adjoint' :                    
           {                              
             'alpha'                      : None,
             'beta'                       : None,
             'max_fun'                    : None,
             'objective_function'         : 'logarithmic',
             'animate'                    : False
           }}

F = SteadySolver(model, config)
F.solve()

T = TransientSolver(model, config)
T.solve()



