from gmsh_meshgenerator import MeshGenerator
from pylab              import *

msh_name = 'cylinder_mesh'
out_dir  = 'meshes/'

x = linspace(-1.0, 1.0, 100)
y = linspace(-1.0, 1.0, 100)

X,Y = meshgrid(x,y)

S = 1 - sqrt(X**2 + Y**2)

m = MeshGenerator(x, y, msh_name, out_dir)

m.create_contour(S, zero_cntr=1e-16, skip_pts=1)
#m.plot_contour()

m.write_gmsh_contour(lc=0.1, boundary_extend=True)
m.extrude(h=1, n_layers=5)

#m.add_edge_attractor(field=0, contour_index=0, NNodesByEdge=10)

##field, ifield, lcMin, lcMax, distMin, distMax
#m.add_threshold(2, 0, 0.001, 0.1, 0, 0.25)

m.finish()

m.create_mesh()
m.convert_msh_to_xml()



