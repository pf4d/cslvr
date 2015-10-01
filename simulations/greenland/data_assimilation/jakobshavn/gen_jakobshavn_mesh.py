from varglas           import MeshGenerator, MeshRefiner, GetBasin, \
                              DataFactory, DataInput, print_min_max
from pylab             import *
from scipy.interpolate import RectBivariateSpline


kappa = 2.0  # ice thickness to refine

#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'jakobshavn_3D_%iH_mesh_block' % int(kappa)

# get the data :
bamber   = DataFactory.get_bamber()
rignot   = DataFactory.get_rignot()
#searise  = DataFactory.get_searise()

# process the data :
dbm      = DataInput(bamber,  gen_space=False)
#drg      = DataInput(rignot,  gen_space=False)
#dsr      = DataInput(searise, gen_space=False)

#drg.change_projection(dbm)
#dbm.change_projection(drg)


#===============================================================================
# form field from which to refine :
dbm.data['ref'] = kappa*dbm.data['H'].copy()
dbm.data['ref'][dbm.data['ref'] < kappa*1000.0] = kappa*1000.0

x1 = -500000; y1 = -2190000
x2 = -270000; y2 = -2320000

x = dbm.x
y = dbm.y

x_valid  = where(x > x1)[0]
x_valid  = intersect1d(x_valid, where(x < x2)[0])

y_valid  = where(y < y1)[0]
y_valid  = intersect1d(y_valid, where(y > y2)[0])

for i in y_valid:
  for j in x_valid:
    dbm.data['ref'][i,j] = 1000.0


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)

m.create_contour('H', zero_cntr=1, skip_pts=0)

#gb = GetBasin(dbm, basin='7.1', edge_resolution=500)
#gb.extend_edge(1200)
#gb.intersection(m.longest_cont)
#gb.plot_xycoords_buf(Show=True, other=m.longest_cont)
#m.set_contour(gb.get_xy_contour())

x1 = -600000; y1 = -2100000
x2 = -100000; y2 = -2400000

new_cont = array([[x1, y1],
                  [x2, y1],
                  [x2, y2],
                  [x1, y2],
                  [x1, y1]])

m.intersection(new_cont)
m.eliminate_intersections(dist=200)
#m.transform_contour(rignot)
#m.check_dist()
#import sys
#sys.exit(0)
m.write_gmsh_contour(boundary_extend=False)
m.plot_contour()
m.extrude(h=100000, n_layers=10)
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



