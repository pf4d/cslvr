from future import standard_library
standard_library.install_aliases()
import cslvr as cs
import numpy as np

out_dir   = './dump/meshes/'
mesh_name = 'nioghalvfjerdsbrae'

#===============================================================================
# data preparation :

# get the togography data :
bedmachine = cs.DataFactory.get_bedmachine()
dbm        = cs.DataInput(bedmachine)

# get the velocity data :
#rignot     = cs.DataFactory.get_rignot()
#drg        = cs.DataInput(rignot)
mouginot   = cs.DataFactory.get_mouginot()
dmg        = cs.DataInput(mouginot)

#drg.change_projection(dbm)

# calculate surface velocity magnitude :
U_ob             = np.sqrt(dmg.data['vx']**2 + dmg.data['vy']**2 + 1e-16)
dmg.data['U_ob'] = U_ob

# put the velocity values into bedmachine's DataInput object :
dbm.interpolate_from_di(dmg, 'U_ob', 'U_ob', order=1)

# form the lateral boundary mask :
lat_mask                 = dbm.data['mask'].copy()
lat_mask[lat_mask < 2.0] = 0.0

def grow_lat_mask(mask, n):
  gradm                  = np.gradient(mask)
  lat_mask               = gradm[0]**2 + gradm[1]**2
  lat_mask[lat_mask > 0] = 1
  if n > 0: grow_lat_mask(lat_mask, n-1)
  else:     return lat_mask

lat_mask = grow_lat_mask(lat_mask, 5)

# form field from which to refine :
dbm.rescale_field('U_ob', 'ref', umin=1000.0, umax=300000.0, inverse=True)
dbm.data['ref'][lat_mask == 1] = 500.0

# generate the contour :
m = cs.MeshGenerator(dbm, mesh_name, out_dir)

m.create_contour('H', zero_cntr=15, skip_pts=10)  # 10 meter thick. contour
m.eliminate_intersections(dist=10)                # eliminate interscting lines
m.save_contour('contour.txt')                     # save the contour for later

# get the basin :
gb = cs.GetBasin(dbm, basin='2.1')
gb.remove_skip_points(400)
gb.extend_edge(10000)
gb.intersection(m.longest_cont)
#gb.plot_xycoords(other=m.longest_cont)
m.set_contour(gb.get_xy_contour())


#===============================================================================
# process the file and extrude :

m.eliminate_intersections(dist=200)              # eliminate interscting lines
#m.transform_contour(rignot) # convert to rignot projection if needed
m.check_dist()                                   # remove points too close
m.write_gmsh_contour(boundary_extend=False)      # create a .geo contour file
#m.plot_contour()                                 # plot the contour
m.extrude(h=100000, n_layers=10)                 # vertically extrude
m.close_file()                                   # close the files


#===============================================================================
# refine :
ref_bm = cs.MeshRefiner(dbm, 'ref', gmsh_file_name = out_dir + mesh_name)

a,aid = ref_bm.add_static_attractor()
ref_bm.set_background_field(aid)

# finish stuff up :
ref_bm.finish(gui = False, out_file_name = out_dir + mesh_name)
ref_bm.convert_msh_to_xml()



