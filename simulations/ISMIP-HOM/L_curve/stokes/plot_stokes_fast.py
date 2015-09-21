from varglas          import D3Model, print_text, print_min_max
from fenics           import *
from numpy            import loadtxt, array
import os

out_dir = 'dump/stokes/fast_bfgs/'

#hs = [1000, 2000, 4000, 8000, 16000, 32000]
#Hs = [250,  500,  750,  1000, 2000,  3000]
#Gs = [0.1,  0.25, 0.5,  1,    2,     4]
#
#for h in hs:
#  for H in Hs:
#    for g in Gs:
#      pass

n     = 25
h     = 1000.0
g     = 0.5

H     = 1000.0
L     = n*h
alpha = g * pi / 180

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 25, 25, 10)

model = D3Model(out_dir = out_dir + 'initial/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - H', alpha=alpha, H=H, 
                     element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded
model.init_beta(beta)
model.init_U_ob(u, v)

lg      = LagrangeInterpolator()
submesh = model.get_bed_mesh()
Q_b     = FunctionSpace(submesh, 'CG', 1)
Q3_b    = MixedFunctionSpace([Q_b]*3)

beta_true_s = Function(Q_b)
beta_opt_s  = Function(Q_b)
U_true_s    = Function(Q3_b)
U_opt_s     = Function(Q3_b)
U_ob_s      = Function(Q3_b)

d = out_dir + 'plot/'
if not os.path.exists(d):
  os.makedirs(d)
savetxt(d + 'Rs.txt', array(Rs))
savetxt(d + 'Js.txt', array(Js))
savetxt(d + 'as.txt', array(alphas))



