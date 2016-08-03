from cslvr             import *
from pylab             import *
from scipy.interpolate import RectBivariateSpline


#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'thwaites_3D_U_mesh_basin_crude'
#mesh_name = 'thwaites_3D_U_mesh_block_crude'

# get the data :
measure = DataFactory.get_ant_measures(res=900)
bedmap2 = DataFactory.get_bedmap2()

# process the data :
dbm = DataInput(measure, gen_space=False)
db2 = DataInput(bedmap2, gen_space=False)

# get surface velocity magnitude :
U_ob = sqrt(dbm.data['vx']**2 + dbm.data['vy']**2 + 1e-16)
dbm.data['U_ob'] = U_ob

#dbm.set_data_min('U_ob', boundary=0.0, val=0.0)
#db2.set_data_val("H",    32767,        0.0)
#db2.set_data_val('S',    32767,        0.0)

# calculate surface gradient :
gradS = gradient(db2.data['S'])
gS_n  = sqrt(gradS[0]**2 + gradS[1]**2 + 1e-16)

# create interpolation object to convert measures to bedmap2 coordinates :
interp = interp2d(db2.x, db2.y, gS_n)
gS_n   = interp(dbm.x, dbm.y)
interp = interp2d(db2.x, db2.y, db2.data['mask'])
mask   = interp(dbm.x, dbm.y)

# get dofs for shelves where we restrict the element size :
slp = gS_n >  20
shf = mask >= 1.1
nan = mask <  0.1
msk = dbm.data['mask'] < 0.1

# eliminate just the edge of the mask so that we can properly interpolate
# the geometry to the terminus :
L = db2.data['lat_mask']
db2.data['mask'][L > 0.0] = 0


#===============================================================================
# form field from which to refine :
dbm.rescale_field('U_ob', 'ref', umin=1000.0, umax=500000.0, inverse=True)

# restrict element size on the shelves and outside the domain of the data :
dbm.data['ref'][slp] = 1000.0
dbm.data['ref'][shf] = 5000.0
#dbm.data['ref'][nan] = 50000.0
#dbm.data['ref'][msk] = 50000.0

print_min_max(dbm.data['ref'], 'ref')

## plot to check :
#imshow(dbm.data['ref'][::-1,:])
#colorbar()
#tight_layout()
#show()


#===============================================================================
# generate the contour :
m = MeshGenerator(db2, mesh_name, out_dir)

m.create_contour('mask', zero_cntr=1e-9, skip_pts=0)
m.plot_contour()

## create a box around some area :
#x1 = -2.20497e6; y1 = 34742
#x2 = -295154;    y2 = -1.56258e6
#
#new_cont = array([[x1, y1],
#                  [x2, y1],
#                  [x2, y2],
#                  [x1, y2],
#                  [x1, y1]])
#m.intersection(new_cont)

# or get the basin of thwaites :
gb = GetBasin(db2, basin='21')
gb.extend_boundary(1000)
gb.check_dist(5000)
gb.intersection(m.longest_cont)
gb.plot_xycoords_buf(other=m.longest_cont)
m.set_contour(gb.get_xy_contour())

m.eliminate_intersections(dist=100)
m.check_dist()
m.write_gmsh_contour(boundary_extend=False)
#m.plot_contour()
m.extrude(h=100000, n_layers=5)
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



