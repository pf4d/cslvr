import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput, MeshGenerator
from data.data_factory import DataFactory

# create meshgrid for contour :
vara = DataFactory.get_V2()

# process the data :
dd = DataInput(None, vara, flip=False, gen_space=False)
dd.set_data_max('mask', 2, 0)

m = MeshGenerator(dd, 'mesh', '../meshes/')

m.create_contour('H', 100.0, 2)
#m.plot_contour()

m.eliminate_intersections(dist=10)
#m.plot_contour()

m.write_gmsh_contour(100000, boundary_extend=False)
m.add_edge_attractor(1)
#field, ifield, lcMin, lcMax, distMin, distMax
m.add_threshold(2, 1, 5000, 10000, 1, 400000)
m.finish(4)

#m.create_2D_mesh('mesh') #FIXME: fails
#m.convert_msh_to_xml('mesh', 'mesh')

