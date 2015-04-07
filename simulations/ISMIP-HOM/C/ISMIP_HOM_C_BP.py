from varglas.model   import Model
from varglas.solvers import SteadySolver
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_text
from fenics          import File, Expression, BoxMesh, pi
from time            import time

t0 = time()

alpha = 0.1 * pi / 180
L     = 40000

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['linear_solver']  = 'cg'
nonlin_solver_params['newton_solver']['preconditioner'] = 'hypre_amg'

config = default_config()
config['output_path']                  = './results_BP_L'+str(L)+'/'
config['periodic_boundary_conditions'] = True
config['velocity']['newton_params']    = nonlin_solver_params
config['model_order']                  = 'BP'
config['use_dukowicz']                 = False

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 50, 50, 10)

model = Model(config)
model.set_mesh(mesh)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('sqrt(1000 + 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L))',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.set_geometry(surface, bed, deform=True)
model.initialize_variables()

model.init_beta(beta)

F = SteadySolver(model, config)
F.solve()

tf = time()

# calculate total time to compute
s   = tf - t0
m   = s / 60.0
h   = m / 60.0
s   = s % 60
m   = m % 60
txt = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
print_text(txt, 'red', 1)






