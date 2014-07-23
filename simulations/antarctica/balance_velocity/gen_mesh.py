from varglas.utilities         import DataInput, MeshGenerator
from varglas.data.data_factory import DataFactory

# create meshgrid for contour :
vara = DataFactory.get_bedmap2()

# process the data :
dd = DataInput(vara, gen_space=False)
#dd.set_data_max('mask', 2, 0)

m = MeshGenerator(dd, 'mesh', '../meshes/') 

m.create_contour('mask', 0.999, 100)
m.plot_contour()

m.eliminate_intersections()

m.write_gmsh_contour(10000.)
m.finish()


