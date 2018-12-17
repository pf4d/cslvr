from cslvr import *

kappa = 5.0  # ice thickness to refine

#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'greenland_2D_%iH_mesh' % int(kappa)

# get the data :
bamber   = DataFactory.get_bamber()
#rignot   = DataFactory.get_rignot()

# process the data :
dbm      = DataInput(bamber,  gen_space=False)
#drg      = DataInput(rignot,  gen_space=False)

#drg.change_projection(dbm)


#===============================================================================
# form field from which to refine :
dbm.data['ref'] = kappa*dbm.data['H'].copy()
dbm.data['ref'][dbm.data['ref'] < kappa*1000.0] = kappa*1000.0

## nice to plot the refinement field to check that you're doing what you want :
#plotIce(dbm, 'ref', name='ref', direc=out_dir,
#       title='ref', cmap='viridis',
#       show=False, scale='lin', tp=False, cb_format='%.1e')

#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)

m.create_contour('mask', zero_cntr=0.99, skip_pts=10)
#m.create_contour('H', zero_cntr=200, skip_pts=5)

m.eliminate_intersections(dist=200)
#m.transform_contour(rignot)
#m.check_dist()
m.write_gmsh_contour(boundary_extend=False)
#m.plot_contour()
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



