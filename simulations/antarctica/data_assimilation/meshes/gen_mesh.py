import sys
import os
src_directory = '../../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput, MeshGenerator
from data.data_factory import DataFactory

# create meshgrid for contour :
vara = DataFactory.get_bedmap2()

# process the data :
dd = DataInput(None, vara, gen_space=False)

m = MeshGenerator(dd, '3dmesh', '')

m.create_contour('mask', 1.0, 10)
m.eliminate_intersections(dist=10)
m.plot_contour()

m.write_gmsh_contour(100000, boundary_extend=False)
m.extrude(100000, 10)

#m.add_edge_attractor(1)
#field, ifield, lcMin, lcMax, distMin, distMax
#m.add_threshold(2, 1, 10000, 50000, 1, 2000000)
#m.finish(4)

#m.create_2D_mesh('mesh') #FIXME: fails
#m.convert_msh_to_xml('mesh', 'mesh')

