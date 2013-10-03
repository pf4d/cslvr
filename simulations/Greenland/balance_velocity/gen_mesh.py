import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput, MeshGenerator
from data.data_factory import DataFactory

# create meshgrid for contour :
vara = DataFactory.get_bamber()
H = vara['H']['map_data']
S = vara['h']['map_data']
B = vara['b']['map_data']
H[S<=0] = 0.0
vara['H']['map_data'] = H

# process the data :
dd = DataInput(None, vara, flip=False, gen_space=False)
#dd.set_data_max('mask', 2, 0)

m = MeshGenerator(dd, 'mesh', './')

m.create_contour('H', 100.0,25)
m.plot_contour()

m.eliminate_intersections()
m.plot_contour()

m.write_gmsh_contour(15000)
m.finish(None)

m.create_2D_mesh('mesh') #FIXME: fails
m.convert_msh_to_xml('mesh', 'mesh')
