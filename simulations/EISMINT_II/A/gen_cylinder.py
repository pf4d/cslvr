from gmsh_meshgenerator import MeshGenerator, MeshRefiner
from pylab              import *

msh_name = 'cylinder_mesh'
out_dir  = 'meshes/'

x = linspace(-1.0, 1.0, 500)
y = linspace(-1.0, 1.0, 500)

X,Y = meshgrid(x,y)

S = 1 - sqrt(X**2 + Y**2)

m = MeshGenerator(x, y, msh_name, out_dir)

m.create_contour(S, zero_cntr=0, skip_pts=20)
m.check_dist(0.01)
#m.plot_contour()
m.write_gmsh_contour(lc=1, boundary_extend=True)
m.extrude(h=1, n_layers=5)
m.finish()
#m.close_file()
m.create_mesh()
m.convert_msh_to_xml()

#S = 1 - sqrt(X**2 + Y**2)
#S[S < 0] = 0
#S        = 1 - S / S.max()
#
#S_min    = 0.001
#S_max    = 0.5
#S        = S_min + S * (S_max - S_min) / (S.max() - S.min())
#
##m     = MeshRefiner(x, y, S, out_dir + msh_name)
##a,aid = m.add_static_attractor(c=1, inv=False)
##m.set_background_field(aid)
#
#m.finish(gui=False, dim=3, out_file_name = out_dir + msh_name)
#m.convert_msh_to_xml()



