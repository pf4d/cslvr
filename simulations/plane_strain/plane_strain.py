from cslvr    import *
from fenics   import RectangleMesh, Point, Expression

l       = 200000.0
d       = 0.0
Hmax    = 4000
S0      = 100
B0      = -200
Bmin    = -400
Tmin    = 228.15
betaMax = 8000.0
betaMin = 1.0
sig     = l/4.0
St      = 6.5 / 1000.0
p1      = Point(-l+d, 0.0)
p2      = Point( l-d, 1.0)
nx      = 1000
nz      = 20
mesh    = RectangleMesh(p1, p2, nx, nz)
out_dir = 'ps_results/'

model = LatModel(mesh, out_dir = out_dir, use_periodic = False)

S = Expression('(Hmax+B0-S0)/2*cos(pi*x[0]/l) + (Hmax+B0+S0)/2',
               Hmax=Hmax, B0=B0, S0=S0, l=l,
               element = model.Q.ufl_element())
B = Expression('10*cos(200*pi*x[0]/(2*l)) + B0', l=l, B0=B0,
               element = model.Q.ufl_element())
#b = Expression('betaMax - sqrt(pow(x[0],2)) * (betaMax - betaMin)/l',
#               betaMax=betaMax, betaMin=betaMin, l=l,
#               element = model.Q.ufl_element())
b = Expression('(bMax - bMin)/2.0*cos(pi*x[0]/l) + (bMax + bMin)/2.0',
               bMax=betaMax, bMin=betaMin, l=l,
               element = model.Q.ufl_element())

model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=None)

model.init_mask(1.0)  # all grounded
model.init_beta(b)
model.init_A(1e-16)
model.init_E(1.0)

mom = MomentumDukowiczPlaneStrain(model)

mom.solve()

model.save_xdmf(model.U3, 'U3')



