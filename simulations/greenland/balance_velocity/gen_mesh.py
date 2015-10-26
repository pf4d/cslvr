from varglas           import MeshGenerator, MeshRefiner, GetBasin, \
                              DataFactory, DataInput, print_min_max
from pylab             import *
from scipy.interpolate import RectBivariateSpline


kappa = 1.0  # ice thickness to refine

#===============================================================================
# data preparation :
out_dir   = 'dump/meshes/'
mesh_name = 'greenland_2D_%iH_mesh' % int(kappa)

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


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, mesh_name, out_dir)

m.create_contour('mask', zero_cntr=0.99, skip_pts=0)

m.eliminate_intersections(dist=200)
#m.transform_contour(rignot)
#m.check_dist()
#import sys
#sys.exit(0)
m.write_gmsh_contour(boundary_extend=False)
m.plot_contour()
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



