from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# set the relavent directories :
#in_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_1/'
#in_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'
in_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100' + \
         '_disc_kappa/tmc/10/'
var_dir  = 'dump/vars_jakobshavn_small/'
out_dir  = in_dir

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')
fini    = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5',   'r')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_mask(fdata)
model.init_q_geo(model.ghf)
model.init_T(fini)
model.init_W(fini)
model.init_alpha(fini)
model.init_PE(fini)
model.init_W_int(fini)
model.init_U(fini)
model.init_p(fini)
model.init_beta(fini)
model.init_theta(fini)

# create enthalpy instance :
nrg = Enthalpy(model, transient=False, use_lat_bc=True)

nrg.solve_basal_melt_rate()
nrg.solve_basal_water_flux()

# after every completed adjoining, save the state of these functions :
tmc_save_vars = [model.T,
                 model.W,
                 model.Wb_flux,
                 model.Mb,
                 model.alpha,
                 model.PE,
                 model.W_int,
                 model.U3,
                 model.p,
                 model.beta,
                 model.theta]

# save state to unique hdf5 file :
out_file = out_dir + 'tmc_new.h5'
foutput  = HDF5File(mpi_comm_world(), out_file, 'w')
for var in tmc_save_vars:
  model.save_hdf5(var, f=foutput)
foutput.close()


 
