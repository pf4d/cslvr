from cslvr import *

# set the output directory :
in_dir  = './dump/vars/'
msh_dir = './dump/meshes/'
out_dir = in_dir

# retrieve the domain contour 
contour  = np.loadtxt(msh_dir + 'contour.txt')

f  = HDF5File(mpi_comm_world(), in_dir + 'state.h5',     'r')
fn = HDF5File(mpi_comm_world(), in_dir + 'submeshes.h5', 'w')

model = D3Model(mesh=f, out_dir=out_dir)

model.form_bed_mesh()
model.form_srf_mesh()
model.form_lat_mesh()
model.form_dvd_mesh(contour)

model.save_bed_mesh(fn)
model.save_srf_mesh(fn)
model.save_lat_mesh(fn)
model.save_dvd_mesh(fn)

fn.close()



