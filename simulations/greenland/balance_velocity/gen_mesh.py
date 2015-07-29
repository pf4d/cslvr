from varglas.meshing           import MeshGenerator, MeshRefiner
from varglas.io                import DataInput
from varglas.data.data_factory import DataFactory
from pylab                     import *
from scipy.interpolate         import interp2d


#===============================================================================
# data preparation :
out_dir = 'meshes/'

# collect the raw data :
bamber   = DataFactory.get_bamber()
rignot   = DataFactory.get_rignot()
searise  = DataFactory.get_searise()
gimp     = DataFactory.get_gimp()

# create data objects to use with varglas :
dbm      = DataInput(bamber,  gen_space=False)
drg      = DataInput(rignot,  gen_space=False)
dsr      = DataInput(searise, gen_space=False)
dgm      = DataInput(gimp,    gen_space=False)

drg.set_data_max('U_ob', boundary=1000.0, val=1000.0)
drg.set_data_min('U_ob', boundary=0.0,    val=0.0)

#===============================================================================
# form field from which to refine :
## antarctica :: Info : 4449632 vertices 25902918 elements
#drg.data['ref'] = (0.02 + 1/(1 + drg.data['U_ob'])) * 50000
dbm.data['ref'] = dbm.data['H'].copy()
dbm.data['ref'][dbm.data['ref'] < 1000.0] = 1000.0

## plot to check :
#imshow(drg.data['ref'][::-1,:])
#colorbar()
#tight_layout()
#show()


#===============================================================================
# generate the contour :

m = MeshGenerator(dbm, 'mesh', out_dir)

m.create_contour('H', zero_cntr=1.0, skip_pts=4)
m.eliminate_intersections(dist=40)
m.eliminate_intersections(dist=40)
m.transform_contour(drg)
#m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
#m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
ref = MeshRefiner(dbm, 'ref', gmsh_file_name = out_dir + 'mesh')

a,aid = ref.add_static_attractor()
ref.set_background_field(aid)

# finish stuff up :
ref.finish(gui=False, out_file_name = out_dir + 'greenland_2D_1H_mesh')




