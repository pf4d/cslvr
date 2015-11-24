from varglas           import MeshGenerator, MeshRefiner, GetBasin, \
                              DataFactory, DataInput, print_min_max
from pylab             import *
from scipy.interpolate import RectBivariateSpline


#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'greenland_2D_mesh'

# get the data :
bamber   = DataFactory.get_bamber()

# process the data :
dbm  = DataInput(bamber,  gen_space=False)

M    = bamber['mask_orig']
m1   = M == 1
m2   = M == 2
mask = logical_or(m1,m2)

dbm.data['M'] = mask


#===============================================================================
# form field from which to refine :
dbm.data['ref'] = mask.astype('i')
dbm.data['ref'][mask == 1] = 10000.0
dbm.data['ref'][mask == 0] = 10000.0


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)

#x1 = dbm.x_min
#x2 = dbm.x_max
#y1 = dbm.y_min
#y2 = dbm.y_max
#
#cont = array([[x1, y1],
#              [x2, y1],
#              [x2, y2],
#              [x1, y2]])
#
#m.set_contour(cont)

m.create_contour('M', zero_cntr=0.999, skip_pts=10)
#m.extend_edge(200000)
#m.remove_skip_points(10)
m.plot_contour()
m.eliminate_intersections(dist=200)
#m.transform_contour(rignot)
#m.check_dist()
#import sys
#sys.exit(0)
m.write_gmsh_contour(boundary_extend=False)
m.close_file()


#===============================================================================
# refine :
ref_bm = MeshRefiner(dbm, 'ref', gmsh_file_name = out_dir + mesh_name)

a,aid = ref_bm.add_static_attractor()
ref_bm.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bm.finish(gui = False, out_file_name = out_dir + mesh_name)
ref_bm.convert_msh_to_xml()



