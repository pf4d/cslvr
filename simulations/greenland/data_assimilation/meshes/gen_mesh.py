from varglas.utilities         import DataInput, MeshGenerator, MeshRefiner
from varglas.data.data_factory import DataFactory
from pylab                     import *


#===============================================================================
# data preparation :

thklim = 0.0

# collect the raw data :
bamber   = DataFactory.get_bamber(thklim = thklim)
searise  = DataFactory.get_searise(thklim = thklim)

dsr  = DataInput(None, searise, gen_space=False)
dbm  = DataInput(None, bamber,  gen_space=False)


#===============================================================================
# generate the contour :
m = MeshGenerator(dbm, 'mesh', '')

m.create_contour('H', zero_cntr=200.0, skip_pts=2)
m.eliminate_intersections(dist=40)
#m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
thklim = 1250.0
#dsr.set_data_min('U_ob', boundary=0.0,    val=0.0)
#dsr.set_data_max('U_ob', boundary=400.0,  val=400.0)
dbm.set_data_min('H',    boundary=thklim, val=thklim)
#dbm.set_data_max('mask', 2, 0)

# might want to refine off of thickness :
H    = dbm.data['H']

# ensure that there are no values less than 1 for taking log :
vel  = dsr.data['U_ob']
vel += 1
vel  = log(vel)

# plot to check :
#imshow(H[::-1,:])
#colorbar()
#show()

ref_bm = MeshRefiner(dbm, 'H',    gmsh_file_name='mesh')   # thickness
#ref_sr = MeshRefiner(dsr, 'U_ob', gmsh_file_name='mesh')   # velocity


#===============================================================================
## refine in steps :
#lmax      = 70000
#lmin      = 2000
#num_cuts  = 8
#lmin_max  = 10000
#lmin_min  = 1000
#fmax_min  = log(50)
#fmax_max  = log(500)
#
#a_list    = []
#fmax_list = linspace(fmax_min, fmax_max, num_cuts)
#lc_list   = linspace(lmin_max, lmin_min, num_cuts)
#for fmax, l_min in zip(fmax_list, lc_list):
#  a,aid = ref_sr.add_linear_attractor(fmax, l_min, lmax, hard_cut=True, 
#                                      inv=True)
#  a_list.append(a.op)
#m1  = ref_sr.add_min_field(a_list)
#ref_sr.set_background_field(m1)


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
a,aid = ref_bm.add_static_attractor()
#a,aid = ref_bm.add_linear_attractor(0, H.min(), H.max(), inv=False, 
#                                    hard_cut=False)
ref_bm.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bm.finish(gui=False)



