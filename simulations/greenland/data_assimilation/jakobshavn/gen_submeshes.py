from cslvr import *

# set the output directory :
in_dir  = 'dump/vars_jakobshavn_small_refined/'
out_dir = in_dir

f  = HDF5File(mpi_comm_world(), in_dir + 'state.h5',     'r')
fn = HDF5File(mpi_comm_world(), in_dir + 'submeshes.h5', 'w')

model = D3Model(mesh=f, out_dir=out_dir)

model.init_lat_mask(f)

model.form_bed_mesh()
model.form_srf_mesh()
model.form_dvd_mesh()

model.save_bed_mesh(fn)
model.save_srf_mesh(fn)
model.save_dvd_mesh(fn)

fn.close()
