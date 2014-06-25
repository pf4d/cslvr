from varglas.utilities         import DataInput, MeshGenerator
from varglas.data.data_factory import DataFactory

# create meshgrid for contour :
vara = DataFactory.get_bamber()
H = vara['H']['map_data']
S = vara['S']['map_data']
B = vara['B']['map_data']
H[S<=0] = 0.0
vara['H']['map_data'] = H

# process the data :
dd = DataInput(None, vara, gen_space=False)
#dd.set_data_max('mask', 2, 0)

m = MeshGenerator(dd, 'mesh', './')

# 20 works well for the last arg. below
m.create_contour('H', 200.0,3)
m.plot_contour()

m.eliminate_intersections()
m.plot_contour()

# 10000 works well on the following line
m.write_gmsh_contour(1000)
m.finish(None)

m.create_2D_mesh('mesh') #FIXME: fails
m.convert_msh_to_xml('mesh', 'mesh')
