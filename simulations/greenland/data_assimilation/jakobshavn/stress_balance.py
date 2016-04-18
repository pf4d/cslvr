from cslvr   import *
from fenics  import *

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.01/11_tik_1e-1/'
in_dir   = base_dir
out_dir  = base_dir + 'stress_balance/'
var_dir  = 'dump/vars_jakobshavn_small/'

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',  'r')
fin     = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5',    'r')
fout    = HDF5File(mpi_comm_world(), in_dir  + 'stress.h5', 'w')

## not deformed mesh :
#mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_S(fdata)
model.init_B(fdata)
model.init_mask(fdata)
model.init_adot(fdata)
model.init_beta(fin)
model.init_U(fin)
model.init_p(fin)
model.init_T(fin)
model.init_W(fin)
model.init_theta(fin)
model.init_E(1.0)

mom = MomentumDukowiczBrinkerhoffStokes(model)
F   = FS_Balance(model, momentum=mom)
F.solve()


model.save_hdf5(model.N_ii, f=fout)
model.save_hdf5(model.N_ij, f=fout)
model.save_hdf5(model.N_ik, f=fout)
model.save_hdf5(model.N_ji, f=fout)
model.save_hdf5(model.N_jj, f=fout)
model.save_hdf5(model.N_jk, f=fout)
model.save_hdf5(model.N_ki, f=fout)
model.save_hdf5(model.N_kj, f=fout)
model.save_hdf5(model.N_kk, f=fout)

model.save_hdf5(model.tau_ii, f=fout)
model.save_hdf5(model.tau_ij, f=fout)
model.save_hdf5(model.tau_ik, f=fout)
model.save_hdf5(model.tau_ji, f=fout)
model.save_hdf5(model.tau_jj, f=fout)
model.save_hdf5(model.tau_jk, f=fout)
model.save_hdf5(model.tau_ki, f=fout)
model.save_hdf5(model.tau_kj, f=fout)
model.save_hdf5(model.tau_kk, f=fout)

fout.close()



