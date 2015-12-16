from varglas          import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys


# set the relavent directories :
#var_dir = 'dump/vars_jakobshavn_basin/'  # directory from gen_vars.py
#out_dir = 'dump/jakob_basin/'            # base directory to save
var_dir = 'dump/vars_jakobshavn_small/'  # directory from gen_vars.py
out_dir = 'dump/jakob_small_rstrt/'      # base directory to save
old_dir = 'dump/jakob_small/'            # base directory to save

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5',           'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',               'r')
frstrt  = HDF5File(mpi_comm_world(), old_dir + '01/hdf5/inverted_01.h5', 'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
d3model.set_srf_mesh(fmeshes)
d3model.set_bed_mesh(fmeshes)
d3model.set_dvd_mesh(fmeshes)

## setup full-stokes functionspaces with 'mini' enriched elements :
#d3model.generate_stokes_function_spaces(kind='mini')

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

#fUin = HDF5File(mpi_comm_world(), out_dir + 'hdf5/U3.h5', 'r')
#d3model.init_U(fUin)

#===============================================================================
# create 2D model for balance velocity :
#d2model = D2Model(d3model.dvdmesh, out_dir, state=foutput)
#
## 2D model gets balance-velocity appropriate variables initialized :
#d2model.assign_submesh_variable(d2model.S,         d3model.S)
#d2model.assign_submesh_variable(d2model.B,         d3model.B)
#d2model.assign_submesh_variable(d2model.mask,      d3model.mask)
#d2model.assign_submesh_variable(d2model.q_geo,     d3model.q_geo)
#d2model.assign_submesh_variable(d2model.T_surface, d3model.T_surface)
#d2model.assign_submesh_variable(d2model.adot,      d3model.adot)
#d2model.assign_submesh_variable(d2model.u_ob,      d3model.u_ob)
#d2model.assign_submesh_variable(d2model.v_ob,      d3model.v_ob)
#d2model.assign_submesh_variable(d2model.U_mask,    d3model.U_mask)
#d2model.assign_submesh_variable(d2model.lat_mask,  d3model.lat_mask)
#
#d2model.calculate_boundaries(mask=d2model.mask,
#                             lat_mask=d2model.lat_mask,
#                             U_mask=d2model.U_mask,
#                             adot=d2model.adot,
#                             latmesh=True, mark_divide=True)
#
#d2model.save_xdmf(d2model.ff, 'd2ff')
#d2model.save_xdmf(d2model.cf, 'd2cf')
#
#d2model.state.close()
#sys.exit(0)

d2model = D2Model(d3model.bedmesh, out_dir)

d2model.assign_submesh_variable(d2model.S,      d3model.S)
d2model.assign_submesh_variable(d2model.B,      d3model.B)
d2model.assign_submesh_variable(d2model.adot,   d3model.adot)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

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

#===============================================================================
# create boundary function spaces for saving variables :
Qb  = FunctionSpace(d3model.bedmesh, 'CG', 1)
Qs  = FunctionSpace(d3model.srfmesh, 'CG', 1)
Q3s = MixedFunctionSpace([Qs]*3)

# functions over appropriate surfaces for saving :
beta   = Function(Qb,  name='beta_SIA')
U_ob   = Function(Qs,  name='U_ob')
Tb     = Function(Qb,  name='Tb')
Us     = Function(Q3s, name='Us')
Wb     = Function(Qb,  name='Wb')
Mb     = Function(Qb,  name='Mb')
beta_b = Function(Qb,  name="beta_control")

# saving the regularization and cost functional values for convergence :
Rs  = []
Js  = []
J1s = []
J2s = []
Ms  = []

#===============================================================================
# generate initial traction field :
d3model.init_T(d3model.T_surface)
#d3model.init_beta_SIA()
d3model.init_beta(frstrt)
#d3model.init_beta(1e4)
#d2model.init_beta_SIA()
#d2model.init_T(d2model.T_surface)

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
             'solve_pressure'       : False,
             'vert_solve_method'    : 'mumps'}

e_params  = {'solver'               : 'mumps',
             'use_surface_climate'  : False}

#mom = MomentumDukowiczStokes(d3model, m_params, isothermal=False)
#mom = MomentumDukowiczBrinkerhoffStokes(d3model, m_params, isothermal=False)
#mom = MomentumDukowiczStokesReduced(d3model, m_params, isothermal=False)
mom = MomentumDukowiczBP(d3model, m_params, isothermal=False)
#mom = MomentumBP(d3model, m_params, isothermal=False)
nrg = Enthalpy(d3model, e_params, transient=False, use_lat_bc=True, 
               epsdot_ftn=mom.strain_rate_tensor)

#nrg.generate_approx_theta(init=True, annotate=False)
#d3model.save_xdmf(d3model.theta_app, 'theta_ini')
#d3model.save_xdmf(d3model.T,         'T_ini')
#d3model.save_xdmf(d3model.W,         'W_ini')
  
nrg.solve_basal_melt_rate()

# post-thermo-solve callback function :
def tmc_post_cb_ftn():
  nrg.solve_basal_melt_rate()
  d3model.assign_submesh_variable(Tb,   d3model.T)
  d3model.assign_submesh_variable(Us,   d3model.U3)
  d3model.assign_submesh_variable(Wb,   d3model.W)
  d3model.assign_submesh_variable(Mb,   d3model.Mb)
  d3model.save_xdmf(Tb,   'Tb')
  d3model.save_xdmf(Us,   'Us')
  d3model.save_xdmf(Wb,   'Wb')
  d3model.save_xdmf(Mb,   'Mb')

# derivative of objective function callback function : 
def deriv_cb(I, dI, beta):
  d3model.assign_submesh_variable(beta_b, beta)
  d3model.save_xdmf(beta_b, 'beta_control')

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
  d3model.init_beta(d3model.control_opt)

  # re-solve the momentum equations with vertical velocity and optimal beta :
  m_params['solve_vert_velocity'] = True
  mom.solve(annotate=False)

  # calculate the L_inf norm of misfit :
  d3model.calc_misfit(d3model.GAMMA_U_GND)
  
  # print the regularization and cost function values :  
  mom.print_eval_ftns()

  # collect all the functionals :
  Rs.append(assemble(mom.Rp))
  Js.append(assemble(mom.J))
  J1s.append(assemble(mom.J1))
  J2s.append(assemble(mom.J2))
  Ms.append(d3model.misfit)
  
  # save the optimal velocity and beta fields for viewing with paraview :
  d3model.assign_submesh_variable(Us,     d3model.U3)
  d3model.assign_submesh_variable(beta_b, d3model.beta)
  d3model.save_xdmf(Us,     'U_opt')
  d3model.save_xdmf(beta_b, 'beta_opt')

# after every completed coupling, save the state of these functions :
tmc_save_vars = [d3model.T,
                 d3model.W,
                 d3model.theta,
                 d3model.U3,
                 d3model.Mb]

# after every completed adjoining, save the state of these functions :
adj_save_vars = [d3model.beta,
                 d3model.U3]

# the initial step saves everything :
ini_save_vars = tmc_save_vars + [d3model.Ubar, d3model.U_ob, d3model.beta]

# form the cost functional :
J = mom.form_obj_ftn(integral=d3model.dSrf_gu, kind='log_L2_hybrid', 
                     g1=0.01, g2=5000)
#J = mom.form_obj_ftn(integral=d3model.dSrf_gu, kind='ratio')

# form the regularization functional :
R = mom.form_reg_ftn(d3model.beta, integral=d3model.dBed_g, kind='TV', 
                     alpha=1.0)
#R = mom.form_reg_ftn(d3model.beta, integral=d3model.dBed_g, kind='Tikhonov', 
#                     alpha=1e-6)

# define the objective functional to minimize :
I = J + R

# reset the output directory to the base : 
d3model.set_out_dir(out_dir)

# assimilate ! :
d3model.assimilate_U_ob(mom, nrg, 
                        control           = d3model.beta,
                        obj_ftn           = I,
                        bounds            = (1e-5, 1e7),
                        method            = 'ipopt',
                        adj_iter          = 1000,
                        iterations        = 10,
                        save_state        = True,
                        ini_save_vars     = ini_save_vars,
                        tmc_save_vars     = tmc_save_vars,
                        adj_save_vars     = adj_save_vars,
                        tmc_callback      = tmc_post_cb_ftn,
                        post_ini_callback = tmc_post_cb_ftn,
                        post_tmc_callback = tmc_post_cb_ftn,
                        post_adj_callback = adj_post_cb_ftn,
                        adj_callback      = deriv_cb,
                        tmc_rtol          = 1e-6,
                        tmc_atol          = 1e2,
                        tmc_max_iter      = 50)
 
# save all the objective function values : 
from numpy import savetxt, array
import os

if d3model.MPI_rank==0:
  d = out_dir + 'objective_ftns/'
  if not os.path.exists(d):
    os.makedirs(d)
  savetxt(d + 'Rs.txt',   array(Rs))
  savetxt(d + 'Js.txt',   array(Js))
  savetxt(d + 'J1s.txt',  array(J1s))
  savetxt(d + 'J2s.txt',  array(J2s))
  savetxt(d + 'Ms.txt',   array(Ms))



