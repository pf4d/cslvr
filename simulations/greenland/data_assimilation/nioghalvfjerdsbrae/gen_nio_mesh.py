from cslvr import *

#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'nioghalvfjerdsbrae_3D'

# get the data :
bedmachine = DataFactory.get_bedmachine()
rignot     = DataFactory.get_rignot()

# process the data :
dbm        = DataInput(bedmachine)
drg        = DataInput(rignot)

drg.change_projection(dbm)

# get surface velocity magnitude :
U_ob = np.sqrt(drg.data['vx']**2 + drg.data['vy']**2 + 1e-16)
drg.data['U_ob'] = U_ob


#===============================================================================
# form field from which to refine :
drg.rescale_field('U_ob', 'ref', umin=100.0, umax=300000.0, inverse=True)

# eliminate just the edge of the mask so that we can properly interpolate
# the geometry to the terminus :
L = dbm.data['lat_mask']
dbm.data['mask'][L > 0.0] = 0


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)

#m.create_contour('mask', zero_cntr=0.0001, skip_pts=0)
m.create_contour('H', zero_cntr=50, skip_pts=20) # 50 meter thickness contour
m.eliminate_intersections(dist=200)              # eliminate interscting lines

#===============================================================================
# a box region :
#x1 = -500000; y1 = -2190000
#x2 = -150000; y2 = -2320000
#
#new_cont = array([[x1, y1],
#                  [x2, y1],
#                  [x2, y2],
#                  [x1, y2],
#                  [x1, y1]])
#
#m.intersection(new_cont)

#===============================================================================
# or a basin :
gb = GetBasin(dbm, basin='2.1')
gb.remove_skip_points(400)
gb.extend_edge(10000)
gb.intersection(m.longest_cont)
gb.plot_xycoords(other=m.longest_cont)
m.set_contour(gb.get_xy_contour())

#===============================================================================
# process the file and extrude :

m.eliminate_intersections(dist=200)              # eliminate interscting lines
#m.transform_contour(rignot) # convert to rignot projection if needed
m.check_dist()                                   # remove points too close
m.write_gmsh_contour(boundary_extend=False)      # create a .geo contour file
m.write_argus_contour()                          # create a .exp contour file
m.plot_contour()                                 # plot the contour
m.extrude(h=100000, n_layers=10)                 # vertically extrude
m.close_file()                                   # close the files


#===============================================================================
# refine :
ref_bm = MeshRefiner(drg, 'ref', gmsh_file_name = out_dir + mesh_name)

a,aid = ref_bm.add_static_attractor()
ref_bm.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bm.finish(gui = False, out_file_name = out_dir + mesh_name)
ref_bm.convert_msh_to_xml()



