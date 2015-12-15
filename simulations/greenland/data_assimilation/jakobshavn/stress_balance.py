from varglas import *
from fenics  import *

# set the relavent directories :
dir_b   = 'dump/jakob_small/'
var_dir = 'dump/vars_jakobshavn_small/'       # directory from gen_vars.py
in_dir  = dir_b + '01/hdf5/'                  # input dir
out_dir = dir_b + '01/'                       # base directory to save

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',       'r')
fthermo = HDF5File(mpi_comm_world(), in_dir  + 'thermo_01.h5',   'r')
finv    = HDF5File(mpi_comm_world(), in_dir  + 'inverted_01.h5', 'r')
fout    = HDF5File(mpi_comm_world(), in_dir  + 'stress_01.h5',   'w')

# not deformed mesh :
#mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir, state=fout)

# init subdomains :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_S(fdata)
model.init_B(fdata)
model.init_mask(fdata)
model.init_adot(fdata)
model.init_beta(finv)
model.init_T(fthermo)
model.init_U(finv)
model.init_W(fthermo)
model.init_theta(fthermo)
model.init_E(1.0)

mom = MomentumDukowiczBP(model)
F   = BP_Balance(model, momentum=mom)
F.solve()

F.solve_component_stress()

model.save_hdf5(model.tau_id)
model.save_hdf5(model.tau_jd)
model.save_hdf5(model.tau_ii)
model.save_hdf5(model.tau_ij)
model.save_hdf5(model.tau_iz)
model.save_hdf5(model.tau_ji)
model.save_hdf5(model.tau_jj)
model.save_hdf5(model.tau_jz)

model.save_xdmf(model.tau_id, 'tau_id')
model.save_xdmf(model.tau_jd, 'tau_jd')
model.save_xdmf(model.tau_ib, 'tau_ib')
model.save_xdmf(model.tau_jb, 'tau_jb')
model.save_xdmf(model.tau_ip, 'tau_ip')
model.save_xdmf(model.tau_jp, 'tau_jp')
model.save_xdmf(model.tau_ii, 'tau_ii')
model.save_xdmf(model.tau_ij, 'tau_ij')
model.save_xdmf(model.tau_iz, 'tau_iz')
model.save_xdmf(model.tau_ji, 'tau_ji')
model.save_xdmf(model.tau_jj, 'tau_jj')
model.save_xdmf(model.tau_jz, 'tau_jz')

model.state.close()


