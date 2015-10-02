from varglas           import MeshGenerator, MeshRefiner, DataFactory, \
                              DataInput, print_min_max
from pylab             import *
from scipy.interpolate import interp2d


#===============================================================================
# data preparation :
out_dir = 'dump/meshes/'

# get the data :
bamber  = DataFactory.get_bamber()
rignot  = DataFactory.get_rignot()

# process the data :
dbm  = DataInput(bamber,  gen_space=False)
drg  = DataInput(rignot,  gen_space=False)

#dbm.change_projection(rignot)

# get surface velocity magnitude :
U_ob = sqrt(drg.data['vx']**2 + drg.data['vy']**2 + 1e-16)
drg.data['U_ob'] = U_ob

#===============================================================================
# form field from which to refine :
drg.rescale_field('U_ob', 'ref', umin=1000.0, umax=40000.0, inverse=True)

print_min_max(drg.data['ref'], 'ref')

## plot to check :
#imshow(dms.data['ref'][::-1,:])
#colorbar()
#tight_layout()
#show()


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, 'mesh_high', out_dir)
m.create_contour('mask', zero_cntr=0.999, skip_pts=0)
m.eliminate_intersections(dist=2000)
m.transform_contour(drg)
#m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
ref = MeshRefiner(drg, 'ref', gmsh_file_name= out_dir + 'mesh_high')

a,aid = ref.add_static_attractor()
ref.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref.finish(gui=False, out_file_name=out_dir + 'gre_mesh_high')
ref.convert_msh_to_xml()



