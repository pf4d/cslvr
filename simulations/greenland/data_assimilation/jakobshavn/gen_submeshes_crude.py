from varglas import *

# set the output directory :
in_dir  = 'dump/vars_jakobshavn_crude/'
out_dir = in_dir

f  = HDF5File(mpi_comm_world(), in_dir + 'state.h5',     'r')
fn = HDF5File(mpi_comm_world(), in_dir + 'submeshes.h5', 'w')

model = D3Model(mesh=f, out_dir=out_dir, save_state=True, state=fn)

model.init_lat_mask(f)

# automatically saved with save_state=True to f
bedmesh = model.get_bed_mesh()
srfmesh = model.get_surface_mesh()
dvdmesh = model.get_divide_mesh()

fn.close()
