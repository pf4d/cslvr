from varglas           import *
from pylab             import *
from scipy.interpolate import interp2d


#===============================================================================
# data preparation :
out_dir = 'dump/meshes/'

# get the data :
measure = DataFactory.get_ant_measures()
bedmap2 = DataFactory.get_bedmap2()

# process the data :
dbm = DataInput(measure, gen_space=False)
db2 = DataInput(bedmap2, gen_space=False)

# get surface velocity magnitude :
U_ob = sqrt(dbm.data['vx']**2 + dbm.data['vy']**2 + 1e-16)
dbm.data['U_ob'] = U_ob

dbm.set_data_min('U_ob', boundary=0.0, val=0.0)
db2.set_data_val("H",    32767,        0.0)
db2.set_data_val('S',    32767,        0.0)

# calculate surface gradient :
gradS = gradient(db2.data['S'])
gS_n  = sqrt(gradS[0]**2 + gradS[1]**2 + 1e-16)

# create interpolation object to convert bedmap2 to measures coordinates :
interp = interp2d(db2.x, db2.y, gS_n)
gS_n   = interp(dbm.x, dbm.y)
interp = interp2d(db2.x, db2.y, db2.data['mask'])
mask   = interp(dbm.x, dbm.y)

# get dofs for shelves where we restrict the element size :
slp = gS_n >  30
shf = mask >= 0.9
gnd = mask <  0.9
slw = U_ob < 1e-7


#===============================================================================
# form field from which to refine :
dbm.rescale_field('U_ob', 'ref', umin=2500.0, umax=50000.0, inverse=True)

# restrict element size on the shelves and outside the domain of the data :
#dbm.data['ref'][slp] = 2000.0
dbm.data['ref'][shf] = 10000.0
dbm.data['ref'][slw] = 10000.0

print_min_max(dbm.data['ref'], 'ref')

## plot to check :
#imshow(dbm.data['ref'][::-1,:])
#colorbar()
#tight_layout()
#show()


#===============================================================================
# generate the contour :
m = MeshGenerator(db2, 'mesh', out_dir)

m.create_contour('mask', zero_cntr=0.999, skip_pts=4)
m.eliminate_intersections(dist=200)
#m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.extrude(h=100000, n_layers=10)
m.close_file()


#===============================================================================
# refine :
ref_bm = MeshRefiner(dbm, 'ref', gmsh_file_name= out_dir + 'mesh')

a,aid = ref_bm.add_static_attractor()
ref_bm.set_background_field(aid)


#===============================================================================
# finish stuff up :
ref_bm.finish(gui=False, out_file_name=out_dir + 'ant_mesh_high')
ref_bm.convert_msh_to_xml()



