from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# set the relavent directories :
var_dir  = 'dump/vars_jakobshavn_small/'
out_dir  = 'dump/jakob_small/tmc_test/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5',   'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',       'r')
fini    = HDF5File(mpi_comm_world(), var_dir + 'inv_Wc_0.01.h5', 'r')

# create 3D model for stokes solves :
model   = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
model.set_subdomains(fdata)
#model.set_srf_mesh(fmeshes)
#model.set_bed_mesh(fmeshes)
#model.set_dvd_mesh(fmeshes)

# initialize the 3D model vars :
model.init_S(fdata)
model.init_B(fdata)
model.init_mask(fdata)
model.init_q_geo(model.ghf)
model.init_T_surface(fdata)
model.init_adot(fdata)
model.init_U_ob(fdata, fdata)
model.init_U_mask(fdata)
model.init_time_step(1e-6)
model.init_E(1.0)
model.init_beta(fini)
#model.init_T(model.T_surface)
#model.init_W(0.0)
model.init_theta(fini)
model.init_T(fini)
model.init_W(fini)
model.init_U(fini)
model.init_p(fini)
model.init_alpha(fini)
model.init_Fb(fini)
model.init_k_0(1e-3)
#model.solve_hydrostatic_pressure()
model.form_energy_dependent_rate_factor()

#frstrt = HDF5File(mpi_comm_world(), out_dir + 'tmc/08/tmc.h5', 'r')
#model.init_alpha(frstrt)
#model.init_U(frstrt)
#model.init_T(frstrt)
#model.init_W(frstrt)
#model.init_theta(frstrt)
#model.init_p(frstrt)

# initialize the physics :
mom = MomentumDukowiczBrinkerhoffStokes(model)
#mom = MomentumDukowiczBP(model)
nrg = Enthalpy(model, mom, transient=False, use_lat_bc=True)

mom.calc_q_fric()
nrg.solve()
fout    = HDF5File(mpi_comm_world(), out_dir + 'theta.h5',   'w')
model.save_hdf5(model.theta, f=fout)
model.save_xdmf(model.theta, 'theta')
sys.exit(0)


# thermo-solve callback function :
def tmc_cb_ftn():
  nrg.calc_PE()#avg=True)
  nrg.calc_internal_water()
  nrg.calc_integrated_strain_heat()
  nrg.solve_basal_melt_rate()


# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
  mom.solve_vert_velocity(annotate=False)

# after every completed adjoining, save the state of these functions :
tmc_save_vars = [model.T,
                 model.W,
                 model.Fb,
                 model.Mb,
                 model.q_fric,
                 model.alpha,
                 model.PE,
                 model.W_int,
                 model.Q_int,
                 model.U3,
                 model.p,
                 model.beta,
                 model.theta]

# form the objective functional for water-flux optimization :
nrg.form_cost_ftn(kind='L2')

## form regularization for water-flux :
#nrg.form_reg_ftn(model.Fb, integral=model.GAMMA_B_GND,
#                 kind='TV', alpha=2.5e7)

wop_kwargs = {'max_iter'            : 350,
              'bounds'              : (0.0, 100.0),
              'method'              : 'ipopt',
              'adj_callback'        : None}
                                    
tmc_kwargs = {'momentum'            : mom,
              'energy'              : nrg,
              'wop_kwargs'          : wop_kwargs,
              'callback'            : tmc_cb_ftn, 
              'atol'                : 1e2,
              'rtol'                : 1e0,
              'max_iter'            : 10,
              'itr_tmc_save_vars'   : tmc_save_vars,
              'post_tmc_save_vars'  : tmc_save_vars,
              'starting_i'          : 1}
                                    
# or restart and thermo_solve :
model.thermo_solve(**tmc_kwargs)



