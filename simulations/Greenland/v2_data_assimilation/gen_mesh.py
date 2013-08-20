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

m.create_contour('H', 100.0, 20)
m.plot_contour()

#m.eliminate_intersections()
#m.plot_contour()

m.write_gmsh_contour(10000)
m.finish()

m.create_2D_mesh('mesh') #FIXME: fails
m.convert_msh_to_xml('mesh', 'mesh')

