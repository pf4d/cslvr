from varglas.utilities         import DataInput, MeshGenerator, MeshRefiner
from varglas.data.data_factory import DataFactory
from pylab                     import *


#===============================================================================
# data preparation :

thklim = 0.0

# create meshgrid for contour :
bedmap2 = DataFactory.get_bedmap2()

# process the data :
dbm = DataInput(bedmap2, gen_space=False)
dbm.set_data_val("H", 32767, thklim)


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, 'mesh', '')

m.create_contour('H',    zero_cntr=200.0, skip_pts=2)
m.eliminate_intersections(dist=40)
#m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
dbm.set_data_min('H', boundary=thklim, val=2000.0)

## plot to check :
#imshow(dbm.data['H'][::-1,:])
#colorbar()
#show()

ref_bm = MeshRefiner(dbm, 'H', gmsh_file_name='mesh')   # thickness


#===============================================================================
## refine on velocity and divide using inverse option :
#lmax = 70000
#lmin = 2000
#
#a1,a1id = ref_sr.add_linear_attractor(log(1.0), lmin, lmax, inv=True, 
#                                      hard_cut=False)
#a2,a2id = ref_sr.add_linear_attractor(log(1.0), lmin, lmax, inv=False, 
#                                      hard_cut=False)
#
#m1  = ref_sr.add_min_field([a1.op, a2.op])
#ref_sr.set_background_field(mid)


#===============================================================================
# refine on thickness :
a,aid = ref_bm.add_static_attractor(100)
#H     = dbm.data['H']
#a,aid = ref_bm.add_linear_attractor(0, H.min(), H.max(), inv=False, 
#                                    hard_cut=False)
ref_bm.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bm.finish(gui=False)



