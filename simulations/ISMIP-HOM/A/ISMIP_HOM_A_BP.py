from varglas.D3Model import D3Model
from varglas.momentum import MomentumBP
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_min_max, print_text
from fenics          import File, Expression, pi, BoxMesh, sqrt

alpha = 0.5 * pi / 180 
L     = 40000

nparams = {'linear_solver'            : 'cg',
           'preconditioner'           : 'hypre_amg',
           'relative_tolerance'       : 1e-8,
           'relaxation_parameter'     : 1.0,
           'maximum_iterations'       : 25,
           'error_on_nonconvergence'  : False}
nparams = {'newton_solver' : nparams}

config = default_config()
config['output_path']                  = './results_BP/'

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 25, 25, 10)

model = D3Model(config)
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

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
model.init_b(model.A0**(-1/model.n))

mom = MomentumBP(model)
mom.solve(annotate=False, params=nparams,
          solve_vert_velocity=True, solve_pressure=True)

model.save_pvd(model.p,  'p')
model.save_pvd(model.U3, 'U')



