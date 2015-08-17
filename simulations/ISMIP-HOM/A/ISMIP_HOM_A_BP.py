from varglas.BPModel import BPModel
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_min_max, print_text
from fenics          import File, Expression, pi, BoxMesh, sqrt

alpha = 0.5 * pi / 180 
L     = 40000

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['linear_solver']  = 'mumps'
#nonlin_solver_params['newton_solver']['linear_solver']  = 'cg'
#nonlin_solver_params['newton_solver']['preconditioner'] = 'hypre_amg'

config = default_config()
config['output_path']                  = './results_BP/'
config['periodic_boundary_conditions'] = True
config['velocity']['newton_params']    = nonlin_solver_params

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 25, 25, 10)

model = BPModel(config)
model.set_mesh(mesh)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha,
                     element=model.Q.ufl_element())
bed     = Expression(  '- x[0] * tan(alpha) - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)
model.initialize_variables()

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(sqrt(1000))

model.init_viscosity_mode('isothermal')
model.init_momentum()
model.solve_momentum()
model.solve_vert_velocity()
model.solve_pressure()

model.save_pvd(model.p,  'p')
model.save_pvd(model.U3, 'U')



