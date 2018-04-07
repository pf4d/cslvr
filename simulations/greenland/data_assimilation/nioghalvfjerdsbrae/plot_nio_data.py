from cslvr import *

# set the relavent directories :
var_dir = 'dump/vars/'  # directory from gen_vars.py
out_dir = 'dump/nio_small/inversion/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
d3model.set_srf_mesh(fmeshes)
d3model.set_bed_mesh(fmeshes)
#d3model.set_dvd_mesh(fmeshes)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_q_geo(d3model.ghf)
d3model.init_T_surface(fdata)
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)

fmeshes.close()
fdata.close()

d3model.save_xdmf(d3model.U_ob, 'U_ob')

import sys
sys.exit(0)

d3model.init_time_step(1e-6)
d3model.init_E(1.0)
d3model.init_W(0.0)
d3model.init_Wc(0.03)
d3model.init_T(d3model.T_surface)
d3model.init_k_0(1e-3)
d3model.solve_hydrostatic_pressure()
d3model.form_energy_dependent_rate_factor()



