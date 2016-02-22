from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# set the relavent directories :
old_dir     = 'dump/jakob_small/' + \
              'tmc_inversion_cont_kappa_TV_beta_reg_10/04/hdf5/'
old_var_dir = 'dump/vars_jakobshavn_small/'
new_var_dir = 'dump/vars_jakobshavn_small_refined/'
out_dir     = 'dump/jakob_small_refined/'
new_dir     = out_dir + 'rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'

# create HDF5 files for saving and loading data :
old_fmeshes = HDF5File(mpi_comm_world(), old_var_dir + 'submeshes.h5', 'r')
old_fdata   = HDF5File(mpi_comm_world(), old_var_dir + 'state.h5',     'r')

new_fmeshes = HDF5File(mpi_comm_world(), new_var_dir + 'submeshes.h5', 'r')
new_fdata   = HDF5File(mpi_comm_world(), new_var_dir + 'state.h5',     'r')

# create 3D model for stokes solves :
oldModel = D3Model(old_fdata, old_dir)
newModel = D3Model(new_fdata, new_dir)

# init subdomains and boundary meshes :
newModel.set_subdomains(new_fdata)
#newModel.set_srf_mesh(new_fmeshes)
#newModel.set_bed_mesh(new_fmeshes)
#newModel.set_dvd_mesh(new_fmeshes)

# initialize the 3D model vars :
newModel.init_S(new_fdata)
newModel.init_B(new_fdata)
newModel.init_mask(new_fdata)
newModel.init_q_geo(newModel.ghf)
newModel.init_T_surface(new_fdata)
newModel.init_adot(new_fdata)
newModel.init_U_ob(new_fdata, new_fdata)
newModel.init_U_mask(new_fdata)
newModel.init_time_step(1e-6)
newModel.init_E(1.0)

#===============================================================================
# generate initial traction field :
newModel.init_T(newModel.T_surface)

mom = MomentumDukowiczBrinkerhoffStokes(newModel, isothermal=False)
nrg = Enthalpy(newModel, transient=False, use_lat_bc=True, 
               epsdot_ftn=mom.strain_rate_tensor)

fini = HDF5File(mpi_comm_world(), old_dir + 'inverted_04.h5', 'r')

oldModel.init_beta(fini)
newModel.assign_submesh_variable(newModel.beta, oldModel.beta)

# post-thermo-solve callback function :
def tmc_cb_ftn():
  nrg.solve_basal_melt_rate()
  nrg.solve_basal_water_flux()
  nrg.calc_PE()#avg=True)
  nrg.calc_internal_water()

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
  mom.solve_vert_velocity(annotate=False)

# after every completed adjoining, save the state of these functions :
tmc_save_vars = [newModel.T,
                 newModel.W,
                 newModel.Wb_flux,
                 newModel.Mb,
                 newModel.alpha,
                 newModel.PE,
                 newModel.W_int,
                 newModel.U3,
                 newModel.p,
                 newModel.beta,
                 newModel.theta]

# form the objective functional for water-flux optimization :
nrg.form_obj_ftn(kind='L2')

# form regularization for water-flux :
nrg.form_reg_ftn(newModel.alpha, integral=newModel.GAMMA_B_GND,
                 kind='TV', alpha=1e8)

wop_kwargs = {'max_iter'            : 500, 
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
              'post_tmc_save_vars'  : tmc_save_vars}
                                    
# or restart and thermo_solve :
newModel.thermo_solve(**tmc_kwargs)



 
