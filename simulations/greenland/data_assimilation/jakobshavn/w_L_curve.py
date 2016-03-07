from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# set the relavent directories :
var_dir  = 'dump/vars_jakobshavn_small/'
out_dir  = 'dump/jakob_small/w_L_curve/'

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')
fini    = HDF5File(mpi_comm_world(), var_dir + 'inv.h5',       'r')

# create 3D model for stokes solves :
model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
model.set_subdomains(fdata)

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
model.init_theta(fini)
model.init_T(fini)
model.init_W(fini)
model.init_U(fini)
model.init_k_0(1.0)
model.solve_hydrostatic_pressure()

#mom = MomentumDukowiczBrinkerhoffStokes(model, isothermal=False)
nrg = Enthalpy(model, transient=False, use_lat_bc=True)

#===============================================================================
# objective gradient callback function :
def post_cb():
  nrg.partition_energy()
  nrg.solve_basal_melt_rate()
  nrg.calc_PE()#avg=True)
  nrg.calc_internal_water()

# after every completed adjoining, save the state of these functions :
itr_save_vars = [model.T,
                 model.W,
                 model.Fb,
                 model.Mb,
                 model.alpha,
                 model.W_int,
                 model.p,
                 model.theta]

#alphas = [1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6]  # TV obj
alphas = [1e5, 1e6, 1e7, 2.5e7, 5e7, 7.5e7, 1e8]

# form the objective functional for water-flux optimization :
nrg.form_cost_ftn(kind='L2')

wop_kwargs = {'max_iter'            : 1000,
              'bounds'              : (0.0, 100.0),
              'method'              : 'ipopt',
              'adj_callback'        : None}

Lc_kwargs = {'alphas'        : alphas,
             'physics'       : nrg,
             'control'       : model.alpha,
             'int_domain'    : model.GAMMA_B_GND,
             'adj_ftn'       : nrg.optimize_water_flux,
             'adj_kwargs'    : wop_kwargs,
             'reg_kind'      : 'TV',
             'pre_callback'  : None,
             'post_callback' : post_cb,
             'itr_save_vars' : itr_save_vars}

model.L_curve(**Lc_kwargs)


 
