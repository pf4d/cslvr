from varglas           import *
from pylab             import *
from scipy.interpolate import interp2d


#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'gre_mesh_low'

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

# eliminate just the edge of the mask so that we can properly interpolate
# the geometry to the terminus :
L = dbm.data['lat_mask']
dbm.data['mask'][L > 0.0] = 0

msk = drg.data['mask'] < 0.1

#===============================================================================
# form field from which to refine :
drg.rescale_field('U_ob', 'ref', umin=2000.0, umax=200000.0, inverse=True)

print_min_max(drg.data['ref'], 'ref')

drg.data['ref'][msk] = 25000.0

## plot to check :
#imshow(dms.data['ref'][::-1,:])
#colorbar()
#tight_layout()
#show()


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)
m.create_contour('mask', zero_cntr=0.01, skip_pts=1)
m.eliminate_intersections(dist=100)
m.check_dist(dist=100)
m.transform_contour(drg)
m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
ref = MeshRefiner(drg, 'ref', gmsh_file_name= out_dir + mesh_name)

a,aid = ref.add_static_attractor()
ref.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref.finish(gui=False, out_file_name=out_dir + mesh_name)
ref.convert_msh_to_xml()



