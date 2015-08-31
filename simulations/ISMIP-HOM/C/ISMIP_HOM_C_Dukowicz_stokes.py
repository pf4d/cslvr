from varglas  import D3Model, MomentumDukowiczStokes
from fenics   import Point, BoxMesh, Expression, sqrt, pi

alpha = 0.1 * pi / 180 
L     = 40000

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 25, 25, 10)

model = D3Model(out_dir = './results_stokes/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('1000 + 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(beta)
model.init_b(model.A0(0)**(-1/model.n(0)))

mom = MomentumDukowiczStokes(model, isothermal=True)
mom.solve()

model.save_pvd(model.p,  'p')
model.save_pvd(model.U3, 'U')





