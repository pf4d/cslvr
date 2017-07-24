from cslvr import *

kappa = 1.0  # ice thickness to refine

#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'antarctica_2D_%iH_mesh' % int(kappa)

# get the data :
bedmap2 = DataFactory.get_bedmap2()

# process the data :
db2 = DataInput(bedmap2)

db2.set_data_val("H", 32767, 0.0)
db2.set_data_val('S', 32767, 0.0)


#===============================================================================
# form field from which to refine :
db2.data['ref'] = kappa*db2.data['H'].copy()
db2.data['ref'][db2.data['ref'] < kappa*1000.0] = kappa*1000.0

## nice to plot the refinement field to check that you're doing what you want :
#plotIce(db2, 'ref', name='ref', direc=out_dir,
#       title='ref', cmap='viridis',
#       show=False, scale='lin', tp=False, cb_format='%.1e')

#===============================================================================
# generate the contour :
m = MeshGenerator(db2, mesh_name, out_dir)

m.create_contour('mask', zero_cntr=0.99, skip_pts=10)
#m.create_contour('H', zero_cntr=200, skip_pts=5)

m.eliminate_intersections(dist=200)
m.write_gmsh_contour(boundary_extend=False)
#m.plot_contour()
m.close_file()


#===============================================================================
# refine :
ref_b2 = MeshRefiner(db2, 'ref', gmsh_file_name = out_dir + mesh_name)

a,aid = ref_b2.add_static_attractor()
ref_b2.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_b2.finish(gui = False, out_file_name = out_dir + mesh_name)
ref_b2.convert_msh_to_xml()




