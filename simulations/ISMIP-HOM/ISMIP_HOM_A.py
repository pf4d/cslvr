from cslvr    import *
from fenics   import Point, BoxMesh, Expression, sqrt, pi

alpha = 0.5 * pi / 180 
L     = 10000

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 15, 15, 10)

model = D3Model(mesh, out_dir = './ISMIP_HOM_A_results/', use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha,
                     element=model.Q.ufl_element())
bed     = Expression(  '- x[0] * tan(alpha) - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_mask(1.0)  # all grounded
model.init_beta(1000)
model.init_A(1e-16)
model.init_E(1.0)

#mom = MomentumBP(model)
#mom = MomentumDukowiczBP(model)
mom = MomentumDukowiczStokesReduced(model)
#mom = MomentumDukowiczBrinkerhoffStokes(model)
mom.solve()

divU = project(div(model.U3))

model.save_xdmf(model.p,  'p')
model.save_xdmf(model.U3, 'U')
model.save_xdmf(divU,     'divU')



