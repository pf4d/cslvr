from cslvr   import *
from fenics  import *

# set the relavent directories :
#base_dir = 'dump/jakob_small/inversion_k_1e-3_FSTMC/10/'
base_dir = 'dump/jakob_small/tmc_cond_disc_vars/'
in_dir   = base_dir
out_dir  = base_dir + 'stress_balance/'
var_dir  = 'dump/vars_jakobshavn_small/'

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')
fin     = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5',       'r')
fout    = HDF5File(mpi_comm_world(), in_dir  + 'Q_int.h5',     'w')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_U(fin)
model.init_T(fin)
model.init_W(fin)
model.init_E(1.0)

nrg = Enthalpy(model)
nrg.calc_integrated_strain_heat()

model.save_hdf5(model.Q_int, f=fout)

fout.close()



