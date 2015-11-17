from varglas          import *
from varglas.energy   import Enthalpy 
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys


# get the input args :
i       = 0
dir_b   = 'dump/jakob_small/0'     # directory to save

# set the relavent directories (complicated, right?!) :
var_dir = 'dump/vars_jakobshavn_small/'
out_dir = dir_b + str(i)   + '/thermo_solve/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5',         'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',             'r')
foutput = HDF5File(mpi_comm_world(), out_dir + 'hdf5/thermo_solve.h5', 'w')

# get the bed and surface meshes :
bedmesh = Mesh()
srfmesh = Mesh()
fmeshes.read(bedmesh, 'bedmesh', False)
fmeshes.read(srfmesh, 'srfmesh', False)

# create boundary function spaces for saving variables :
Qb  = FunctionSpace(bedmesh, 'CG', 1)
Qs  = FunctionSpace(srfmesh, 'CG', 1)
Q3s = MixedFunctionSpace([Qs]*3)

# create 3D model for stokes solves, and 2D model for balance velocity :
d3model = D3Model(fdata,   out_dir, state=foutput)
d2model = D2Model(bedmesh, out_dir, state=foutput)

## setup full-stokes functionspaces with 'mini' enriched elements :
#d3model.generate_stokes_function_spaces(kind='mini')

# initialize the 3D model vars :
d3model.set_subdomains(fdata)
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_q_geo(d3model.ghf)
d3model.init_T_surface(fdata)
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
d3model.init_E(1.0)

d3model.save_xdmf(d3model.ff, 'ff')

fUin = HDF5File(mpi_comm_world(), out_dir + 'hdf5/U3.h5', 'r')
d3model.init_U(fUin)

# 2D model gets balance-velocity appropriate variables initialized :
d2model.assign_submesh_variable(d2model.S,    d3model.S)
d2model.assign_submesh_variable(d2model.B,    d3model.B)
d2model.assign_submesh_variable(d2model.adot, d3model.adot)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.save_xdmf(d2model.Ubar, 'Ubar')
d2model.save_hdf5(d2model.Ubar)

# assign the balance velocity to the 3D model's bed :
d3model.assign_submesh_variable(d3model.d_x,  d2model.d_x)
d3model.assign_submesh_variable(d3model.d_y,  d2model.d_y)
d3model.assign_submesh_variable(d3model.Ubar, d2model.Ubar)

# extrude the bed values up the column : 
d_x_e  = d3model.vert_extrude(d3model.d_x,  d='up')
d_y_e  = d3model.vert_extrude(d3model.d_y,  d='up')
Ubar_e = d3model.vert_extrude(d3model.Ubar, d='up')

# set the appropriate variable to be the function extruded :
d3model.init_d_x(d_x_e)
d3model.init_d_y(d_y_e)
d3model.init_Ubar(Ubar_e)

# use T0 and beta0 from the previous run :
if i > 0:
  inv_dir = dir_b + str(i-1) + '/inverted/hdf5/'
  tmc_dir = dir_b + str(i-1) + '/thermo_solve/hdf5/'
  fin_inv = HDF5File(mpi_comm_world(), inv_dir + 'inverted.h5',     'r')
  fin_tmc = HDF5File(mpi_comm_world(), tmc_dir + 'thermo_solve.h5', 'r')
  d3model.init_T(fin_tmc)      # temp
  d3model.init_W(fin_tmc)      # water
  d3model.init_beta(fin_inv)   # friction
  d3model.init_E_shf(fin_inv)  # enhancement
else:
  d3model.init_T(d3model.T_surface)
  #d3model.init_beta(1e4)
  d3model.init_beta_SIA()

nparams = {'newton_solver' : {'linear_solver'            : 'cg',
                              'preconditioner'           : 'hypre_amg',
                              'relative_tolerance'       : 1e-9,
                              'relaxation_parameter'     : 0.7,
                              'maximum_iterations'       : 30,
                              'error_on_nonconvergence'  : False}}
#nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
#                              'relative_tolerance'       : 1e-9,
#                              'relaxation_parameter'     : 0.7,
#                              'maximum_iterations'       : 30,
#                              'error_on_nonconvergence'  : False}}
m_params  = {'solver'               : nparams,
             'solve_vert_velocity'  : True,
             'solve_pressure'       : True,
             'vert_solve_method'    : 'mumps'}

e_params  = {'solver'               : 'mumps',
             'use_surface_climate'  : False}

#mom = MomentumDukowiczStokes(d3model, m_params, isothermal=False)
#mom = MomentumDukowiczBrinkerhoffStokes(d3model, m_params, isothermal=False)
mom = MomentumDukowiczStokesReduced(d3model, m_params, isothermal=False)
#mom = MomentumBP(d3model, m_params, isothermal=False)
nrg = Enthalpy(d3model, e_params, use_lat_bc=True, 
               epsdot_ftn=mom.strain_rate_tensor)

# functions over appropriate surfaces for saving :
beta = Function(Qb,  name='beta_SIA')
U_ob = Function(Qs,  name='U_ob')
Tb   = Function(Qb,  name='Tb')
Us   = Function(Q3s, name='Us')
Wb   = Function(Qb,  name='Wb')
Mb   = Function(Qb,  name='Mb')
rhob = Function(Qb,  name='rhob')

d3model.assign_submesh_variable(beta, d3model.beta)
d3model.assign_submesh_variable(U_ob, d3model.U_ob)
d3model.save_xdmf(beta, 'beta_SIA')
d3model.save_xdmf(U_ob, 'U_ob')

nrg.generate_approx_theta(init=False, annotate=False)
d3model.save_xdmf(d3model.theta_app, 'theta_ini')
d3model.save_xdmf(d3model.T,         'T_ini')
d3model.save_xdmf(d3model.W,         'W_ini')
sys.exit(0)

def cb_ftn():
  nrg.calc_bulk_density()
  nrg.solve_basal_melt_rate()
  #nrg.generate_approx_theta(annotate=False)
  d3model.assign_submesh_variable(Tb,   d3model.T)
  d3model.assign_submesh_variable(Us,   d3model.U3)
  d3model.assign_submesh_variable(Wb,   d3model.W)
  d3model.assign_submesh_variable(Mb,   d3model.Mb)
  d3model.assign_submesh_variable(rhob, d3model.rho_b)
  d3model.save_xdmf(d3model.T, 'T')
  d3model.save_xdmf(d3model.theta_app, 'theta_app')
  d3model.save_xdmf(d3model.theta,     'theta')
  d3model.save_xdmf(Tb,   'Tb')
  d3model.save_xdmf(Us,   'Us')
  d3model.save_xdmf(Wb,   'Wb')
  d3model.save_xdmf(Mb,   'Mb')
  d3model.save_xdmf(rhob, 'rhob')
  d3model.save_xdmf(d3model.p, 'p')

d3model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=pi*1e3, max_iter=15)

d3model.save_hdf5(d3model.theta)
d3model.save_hdf5(d3model.T)
d3model.save_hdf5(d3model.W)
d3model.save_hdf5(d3model.U3)
d3model.save_hdf5(d3model.beta)
d3model.save_hdf5(d3model.Mb)



