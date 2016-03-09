from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys


# set the relavent directories :
var_dir = 'dump/vars_jakobshavn_small/'  # directory from gen_vars.py
out_dir = 'dump/jakob_small/inversion/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
d3model.set_srf_mesh(fmeshes)
d3model.set_bed_mesh(fmeshes)
d3model.set_dvd_mesh(fmeshes)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_q_geo(d3model.ghf)
d3model.init_T_surface(fdata)
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
d3model.init_time_step(1e-6)
d3model.init_E(1.0)
d3model.init_k_0(1.0)
d3model.solve_hydrostatic_pressure()

# NOTE: un-comment this for initializing beta to beta_SIA :
## create a 2D model for balance-velocity :
#bedmodel = D2Model(d3model.bedmesh, out_dir)
#
#bedmodel.assign_submesh_variable(bedmodel.S,      d3model.S)
#bedmodel.assign_submesh_variable(bedmodel.B,      d3model.B)
#bedmodel.assign_submesh_variable(bedmodel.adot,   d3model.adot)
#
## solve the balance velocity :
#bv = BalanceVelocity(bedmodel, kappa=5.0)
#bv.solve(annotate=False)
#
## assign the balance velocity to the 3D model's bed :
#d3model.assign_submesh_variable(d3model.d_x,  bedmodel.d_x)
#d3model.assign_submesh_variable(d3model.d_y,  bedmodel.d_y)
#d3model.assign_submesh_variable(d3model.Ubar, bedmodel.Ubar)
#
## extrude the bed values up the column : 
#d_x_e  = d3model.vert_extrude(d3model.d_x,  d='up')
#d_y_e  = d3model.vert_extrude(d3model.d_y,  d='up')
#Ubar_e = d3model.vert_extrude(d3model.Ubar, d='up')
#
## set the appropriate variable to be the function extruded :
#d3model.init_d_x(d_x_e)
#d3model.init_d_y(d_y_e)
#d3model.init_Ubar(Ubar_e)
#
## generate initial traction field :
#d3model.init_T(d3model.T_surface)
#d3model.init_beta_SIA()

mom = MomentumDukowiczBP(d3model, linear=False, isothermal=False)
nrg = Enthalpy(d3model, transient=False, use_lat_bc=True)
#               epsdot_ftn=mom.strain_rate_tensor)

frstrt = HDF5File(mpi_comm_world(), out_dir + '03/tmc.h5', 'r')
d3model.init_alpha(frstrt)
d3model.init_U(frstrt)
d3model.init_T(frstrt)
d3model.init_W(frstrt)
d3model.init_theta(frstrt)
d3model.init_beta(frstrt)
d3model.init_p(frstrt)

# thermo-solve callback function :
def tmc_cb_ftn():
  nrg.solve_basal_water_flux()
  nrg.calc_PE()#avg=True)
  nrg.calc_internal_water()

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
  # solve for optimal vertical velocity :
  mom.solve_vert_velocity(annotate=False)

# after every completed adjoining, save the state of these functions :
tmc_save_vars = [d3model.T,
                 d3model.W,
                 d3model.Fb,
                 d3model.Mb,
                 d3model.alpha,
                 d3model.PE,
                 d3model.W_int,
                 d3model.U3,
                 d3model.p,
                 d3model.beta,
                 d3model.theta]

# form the cost functional :
mom.form_obj_ftn(integral=d3model.GAMMA_U_GND, kind='log_L2_hybrid', 
                 g1=0.01, g2=5000)

# form the regularization functional :
mom.form_reg_ftn(d3model.beta, integral=d3model.GAMMA_B_GND,
                 kind='TV', alpha=10.0)
#mom.form_reg_ftn(d3model.beta, integral=d3model.GAMMA_B_GND,
#                  kind='Tikhonov', alpha=1e-6)

# form the objective functional for water-flux optimization :
nrg.form_cost_ftn(kind='L2')

## form regularization for water-flux :
#nrg.form_reg_ftn(d3model.alpha, integral=d3model.GAMMA_B_GND,
#                 kind='TV', alpha=1e7)

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
              'max_iter'            : 5,
              'itr_tmc_save_vars'   : tmc_save_vars,
              'post_tmc_save_vars'  : tmc_save_vars,
              'starting_i'          : 1}

uop_kwargs = {'control'             : d3model.beta,
              'bounds'              : (1e-5, 1e7),
              'method'              : 'ipopt',
              'max_iter'            : 250,
              'adj_save_vars'       : None,
              'adj_callback'        : None,
              'post_adj_callback'   : adj_post_cb_ftn}
                                    
ass_kwargs = {'iterations'          : 10,
              'tmc_kwargs'          : tmc_kwargs,
              'uop_kwargs'          : uop_kwargs,
              'initialize'          : False,
              'incomplete'          : True,
              'ini_save_vars'       : tmc_save_vars,
              'post_iter_save_vars' : tmc_save_vars,
              'post_ini_callback'   : None,
              'starting_i'          : 4}

# assimilate ! :
d3model.assimilate_U_ob(**ass_kwargs) 
                                    
## or restart and thermo_solve :
#d3model.set_out_dir(out_dir + '03/')
#d3model.thermo_solve(**tmc_kwargs)

 

