from cslvr             import *
from pylab             import *
from scipy.interpolate import RectBivariateSpline


#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'jakobshavn_3D_crude'

# get the data :
bamber   = DataFactory.get_bamber()
rignot   = DataFactory.get_rignot()

# process the data :
dbm      = DataInput(bamber,  gen_space=False)
drg      = DataInput(rignot,  gen_space=False)

drg.change_projection(dbm)

# get surface velocity magnitude :
U_ob = sqrt(drg.data['vx']**2 + drg.data['vy']**2 + 1e-16)
drg.data['U_ob'] = U_ob


#===============================================================================
# form field from which to refine :
drg.rescale_field('U_ob', 'ref', umin=500.0, umax=500000.0, inverse=True)

# eliminate just the edge of the mask so that we can properly interpolate
# the geometry to the terminus :
L = dbm.data['lat_mask']
dbm.data['mask'][L > 0.0] = 0


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)

m.create_contour('mask', zero_cntr=0.0001, skip_pts=0)

x1 = -500000; y1 = -2190000
x2 = -150000; y2 = -2320000

new_cont = array([[x1, y1],
                  [x2, y1],
                  [x2, y2],
                  [x1, y2],
                  [x1, y1]])

m.intersection(new_cont)
m.eliminate_intersections(dist=20)
m.transform_contour(rignot)
m.check_dist()
m.write_gmsh_contour(boundary_extend=False)
#m.plot_contour()
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
ref_bm = MeshRefiner(drg, 'ref', gmsh_file_name = out_dir + mesh_name)

a,aid = ref_bm.add_static_attractor()
ref_bm.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bm.finish(gui = False, out_file_name = out_dir + mesh_name)
ref_bm.convert_msh_to_xml()



