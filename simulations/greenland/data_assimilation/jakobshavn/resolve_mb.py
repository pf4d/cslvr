from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# set the relavent directories :
in_dir   = 'dump/jakob_small/rstrt_FS_a_0_1_cont/'
var_dir  = 'dump/vars_jakobshavn_small/'
out_dir  = in_dir

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')
fini    = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5',   'r')
finv    = HDF5File(mpi_comm_world(), var_dir + 'inv.h5',   'r')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
model.set_subdomains(fdata)

# initialize the 3D model vars :
model.init_S(fdata)
model.init_B(fdata)
model.init_mask(fdata)
model.init_q_geo(model.ghf)
model.init_T(fini)
model.init_W(fini)
model.init_alpha(fini)
model.init_PE(fini)
model.init_W_int(fini)
model.init_U(fini)
model.init_beta(fini)
model.init_theta(fini)
model.init_k_0(1.0)
model.init_p(fini)

p = model.p.copy()

model.solve_hydrostatic_pressure()

p_v  = p.vector().array()
ph_v = model.p.vector().array()

dp = Function(model.Q, name='dp')
model.assign_variable(dp, p_v - ph_v)
    
# Mb is only valid on basal surface, needs extra matrix care :
phi  = TestFunction(model.Q)
du   = TrialFunction(model.Q)
a_n  = du * phi * model.dBed_g
L_n  = div(model.U3) * phi * model.dBed_g

A_n  = assemble(a_n, keep_diagonal=True, annotate=False)
B_n  = assemble(L_n, annotate=False)
A_n.ident_zeros()

divU = Function(model.Q, name='divU')
solve(A_n, divU.vector(), B_n, 'cg', 'amg', annotate=False)

# create enthalpy instance :
nrg = Enthalpy(model, transient=False, use_lat_bc=True)

nrg.solve_basal_melt_rate()
nrg.solve_basal_water_flux()

# after every completed adjoining, save the state of these functions :
tmc_save_vars = [model.Fb,
                 model.Mb,
                 model.PE,
                 model.p,
                 dp,
                 divU]

# save state to unique hdf5 file :
out_file = out_dir + 'Mb.h5'
foutput  = HDF5File(mpi_comm_world(), out_file, 'w')
for var in tmc_save_vars:
  model.save_hdf5(var, f=foutput)
foutput.close()


 
