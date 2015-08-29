from varglas import MomentumDukowiczReduced
from fenics           import parameters, Point, BoxMesh, Expression, sqrt, pi

parameters['form_compiler']['quadrature_degree'] = 2

alpha = 0.5 * pi / 180 
L     = 5000

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 25, 25, 10)

model = D3Model(out_dir = './results_stokes_reduced/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha,
                     element=model.Q.ufl_element())
bed     = Expression(  '- x[0] * tan(alpha) - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(sqrt(1000))
model.init_b(model.A0(0)**(-1/model.n(0)))

mom = MomentumDukowiczStokesReduced(model)
mom.solve()

#model.save_pvd(model.p,  'p')
model.save_pvd(model.U3, 'U')

u = model.u
v = model.v
B = model.B
z = model.x[2]



