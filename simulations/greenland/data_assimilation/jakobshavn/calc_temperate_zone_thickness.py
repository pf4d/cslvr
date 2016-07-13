from cslvr   import *
from fenics  import *

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.01/10/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'
var_dir  = 'dump/vars_jakobshavn_small/'

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',       'r')
fin     = HDF5File(mpi_comm_world(), in_dir  + 'inverted_10.h5', 'r')
fout    = HDF5File(mpi_comm_world(), in_dir  + 'alpha_int.h5',   'w')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_S(fdata)
model.init_B(fdata)
model.init_alpha(fin)
model.init_W(fin)

momTMC = MomentumDukowiczBrinkerhoffStokes(model)
nrg    = Enthalpy(model, momTMC)
nrg.calc_vert_avg_W()
nrg.calc_temp_rat()

model.save_hdf5(model.alpha_int, f=fout)
model.save_hdf5(model.Wbar,      f=fout)
model.save_hdf5(model.temp_rat,  f=fout)

fout.close()



