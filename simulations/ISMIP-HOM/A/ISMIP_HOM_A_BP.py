from varglas.model   import Model
from varglas.solvers import SteadySolver
from varglas.helper  import default_nonlin_solver_params, default_config
from fenics          import File, Expression, pi, BoxMesh, sqrt

alpha = 0.5 * pi / 180 
L     = 40000

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['linear_solver']  = 'mumps'
#nonlin_solver_params['newton_solver']['preconditioner'] = 'hypre_amg'

config = default_config()
config['output_path']                  = './results_BP/'
config['periodic_boundary_conditions'] = True
config['velocity']['newton_params']    = nonlin_solver_params
config['model_order']                  = 'BP'
config['use_dukowicz']                 = False
config['velocity']['full_BP']          = True

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 50, 50, 10)

model = Model(config)
model.set_mesh(mesh)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha,
                     element=model.Q.ufl_element())
bed     = Expression(  '- x[0] * tan(alpha) - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.set_geometry(surface, bed, deform=True)
model.initialize_variables()

model.init_beta(sqrt(1000))

F = SteadySolver(model, config)
F.solve()



