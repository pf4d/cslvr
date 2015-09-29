from varglas  import MeshGenerator, MeshRefiner, DataFactory, DataInput, \
                     print_min_max
from pylab             import *
from scipy.interpolate import interp2d


#===============================================================================
# data preparation :
out_dir = 'dump/meshes/'

# get the data :
bamber  = DataFactory.get_bamber()
measure = DataFactory.get_gre_measures()

# process the data :
dbm  = DataInput(bamber,  gen_space=False)
dms  = DataInput(measure, gen_space=False)

dms.set_data_val('vx', -2e9, 0.0)
dms.set_data_val('vy', -2e9, 0.0)

# get surface velocity magnitude :
U_ob = sqrt(dms.data['vx']**2 + dms.data['vy']**2 + 1e-16)
dms.data['U_ob'] = U_ob

#dms.set_data_min('U_ob', boundary=0.0, val=0.0)

#===============================================================================
# form field from which to refine :
dms.data['ref'] = (0.05 + 1/(1 + dms.data['U_ob'])) * 50000

print_min_max(dms.data['ref'], 'ref')

## plot to check :
#imshow(dms.data['ref'][::-1,:])
#colorbar()
#tight_layout()
#show()


#===============================================================================
# generate the contour :
m = MeshGenerator(dms, 'mesh_ob', out_dir)

m.create_contour('U_ob', zero_cntr=1e-7, skip_pts=1)
m.eliminate_intersections(dist=200)
#m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
ref_bs = MeshRefiner(dms, 'ref', gmsh_file_name= out_dir + 'mesh_ob')

a,aid = ref_bs.add_static_attractor()
ref_bs.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bs.finish(gui=False, out_file_name=out_dir + 'gre_mesh_ant_spacing_ob')
ref_bs.convert_msh_to_xml()



