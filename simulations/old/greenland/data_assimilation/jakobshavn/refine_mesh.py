from cslvr    import *
from fenics   import *

# set the relavent directories :
base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'
var_dir  = 'dump/vars_jakobshavn_small/'

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

#===============================================================================
# open the hdf5 file :
f     = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5', 'r')

# initialize the water content :
d3model.init_W(f)

zmax = mesh.coordinates()[:,2].max()

cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)
for cell in cells(mesh):
  p  = cell.midpoint()
  if p.z() < (zmax/10)/2 and d3model.W(p) > 1e-4:
    cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

File('dump/meshes/jakobshavn_3D_small_block_refined.xml') << mesh

d3model.set_mesh(mesh)
