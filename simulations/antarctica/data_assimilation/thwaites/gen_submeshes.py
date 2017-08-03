from cslvr import *

# set the output directory :
msh_lvl = 'crude'
in_dir  = 'dump/vars_thwaites_basin_%s/' % msh_lvl
out_dir = in_dir

f  = HDF5File(mpi_comm_world(), in_dir + 'state_%s.h5' % msh_lvl,     'r')
fn = HDF5File(mpi_comm_world(), in_dir + 'submeshes_%s.h5' % msh_lvl, 'w')

model = D3Model(mesh=f, out_dir=out_dir)

model.init_lat_mask(f)

model.form_bed_mesh()
model.form_srf_mesh()
model.form_lat_mesh()
model.form_dvd_mesh()

model.save_bed_mesh(fn)
model.save_srf_mesh(fn)
model.save_lat_mesh(fn)
model.save_dvd_mesh(fn)

fn.close()
