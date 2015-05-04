from varglas.model   import Model
from varglas.solvers import SteadySolver
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_min_max, print_text
from fenics          import File, Expression, pi, BoxMesh, sqrt, parameters
from time            import time

t0 = time()

alpha = 0.5 * pi / 180 
L     = 5000

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['linear_solver']  = 'mumps'

config = default_config()
config['output_path']                  = './results_stokes/'
config['periodic_boundary_conditions'] = True
config['velocity']['newton_params']    = nonlin_solver_params
config['model_order']                  = 'stokes'
config['use_dukowicz']                 = False
parameters['form_compiler']['quadrature_degree'] = 2

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 25, 25, 10)

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
model.init_viscosity_mode('isothermal')

F = SteadySolver(model, config)
F.solve()

# calculate total time to compute
s   = time() - t0
m   = s / 60.0
h   = m / 60.0
s   = s % 60
m   = m % 60
txt = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
print_text(txt, 'red', 1)



