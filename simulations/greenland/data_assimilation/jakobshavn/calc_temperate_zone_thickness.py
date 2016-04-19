from cslvr   import *
from fenics  import *

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.03/01/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'
var_dir  = 'dump/vars_jakobshavn_small/'

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',       'r')
fin     = HDF5File(mpi_comm_world(), in_dir  + 'inverted_01.h5', 'r')
fout    = HDF5File(mpi_comm_world(), in_dir  + 'alpha_int.h5',   'w')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_alpha(fin)

momTMC = MomentumDukowiczBrinkerhoffStokes(model)
nrg    = Enthalpy(model, momTMC)
nrg.calc_temperate_thickness()

model.save_hdf5(model.alpha_int, f=fout)

fout.close()



