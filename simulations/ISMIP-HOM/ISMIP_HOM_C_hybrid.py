from cslvr    import HybridModel, MomentumHybrid
from fenics   import Point, RectangleMesh, Expression, sqrt, pi

alpha = 0.1 * pi / 180 
L     = 10000

p1    = Point(0.0, 0.0)
p2    = Point(L,   L)
mesh  = RectangleMesh(p1, p2, 25, 25)

model = HybridModel(mesh, out_dir = './ISMIP_HOM_C_hybrid_results/',
                    use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1000.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('1000 + 1000 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.init_S(surface)
model.init_B(bed)
model.init_mask(1.0)  # all grounded
model.init_beta(beta)
model.init_A(1e-16)

mom = MomentumHybrid(model)
mom.solve()

model.save_xdmf(model.U3_s, 'U_S')



