from cslvr            import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# painAmplifier IPOpt : Total time to compute: 47:35:04
# painAmplifier bfgs  : Total time to compute: 41:21:00


# set the relavent directories :
var_dir = 'dump/vars_jakobshavn_small/'  # directory from gen_vars.py
out_dir = 'dump/jakob_small/'            # base directory to save

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5',     'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',         'r')

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

#fUin = HDF5File(mpi_comm_world(), out_dir + 'U3.h5', 'r')
#d3model.init_U(fUin)

#===============================================================================
## create 2D model for lateral energy solution :
#latmodel = D2Model(d3model.dvdmesh, out_dir)
#
## 2D model gets balance-velocity appropriate variables initialized :
#latmodel.assign_submesh_variable(latmodel.S,         d3model.S)
#latmodel.assign_submesh_variable(latmodel.B,         d3model.B)
#latmodel.assign_submesh_variable(latmodel.mask,      d3model.mask)
#latmodel.assign_submesh_variable(latmodel.T_surface, d3model.T_surface)
#latmodel.assign_submesh_variable(latmodel.adot,      d3model.adot)
#latmodel.assign_submesh_variable(latmodel.u_ob,      d3model.u_ob)
#latmodel.assign_submesh_variable(latmodel.v_ob,      d3model.v_ob)
#latmodel.assign_submesh_variable(latmodel.U_mask,    d3model.U_mask)
#latmodel.assign_submesh_variable(latmodel.lat_mask,  d3model.lat_mask)
#latmodel.assign_submesh_variable(latmodel.U3,        d3model.U3)
#latmodel.init_q_geo(latmodel.ghf)
#latmodel.init_T_surface(240)
#latmodel.init_T(240)
#latmodel.init_E(1.0)
#
#latmodel.save_xdmf(latmodel.T_surface, 'T_surface')
#
#latmodel.calculate_boundaries(mask=latmodel.mask,
#                             lat_mask=latmodel.lat_mask,
#                             U_mask=latmodel.U_mask,
#                             adot=latmodel.adot,
#                             latmesh=True, mark_divide=True)
#
##latmodel.save_xdmf(latmodel.ff, 'lat_ff')
#phi  = TestFunction(d3model.V)
#du   = TrialFunction(d3model.V)
#dudt = Function(d3model.V)
#n    = d3model.N
#u    = d3model.U3
#dLat = d3model.dLat_d
#u_t  = u - dot(u, n)*n
#a_n  = inner(du,  phi) * dLat
#L_n  = inner(u_t, phi) * dLat
#A_n  = assemble(a_n, keep_diagonal=True)
#B_n  = assemble(L_n, keep_diagonal=True)
#A_n.ident_zeros()
#solve(A_n, dudt.vector(), B_n)
#
#latmodel.assign_submesh_variable(latmodel.U3, dudt)
#d3model.save_xdmf(dudt, 'u_t')
#latmodel.save_xdmf(latmodel.U3, 'u_t_lat')
#
#nrg = Enthalpy(latmodel)
#nrg.solve()
#latmodel.save_xdmf(latmodel.T, 'T_lat')
#latmodel.save_xdmf(latmodel.W, 'W_lat')
#sys.exit(0)

bedmodel = D2Model(d3model.bedmesh, out_dir)

bedmodel.assign_submesh_variable(bedmodel.S,      d3model.S)
bedmodel.assign_submesh_variable(bedmodel.B,      d3model.B)
bedmodel.assign_submesh_variable(bedmodel.adot,   d3model.adot)

# solve the balance velocity :
bv = BalanceVelocity(bedmodel, kappa=5.0)
bv.solve(annotate=False)

# assign the balance velocity to the 3D model's bed :
d3model.assign_submesh_variable(d3model.d_x,  bedmodel.d_x)
d3model.assign_submesh_variable(d3model.d_y,  bedmodel.d_y)
d3model.assign_submesh_variable(d3model.Ubar, bedmodel.Ubar)

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
Wb_fl  = Function(Qb,  name="Wb_flux")
Mb     = Function(Qb,  name='Mb')
betab  = Function(Qb,  name="beta_control")
alphab = Function(Qb,  name="alpha_control")

# saving the regularization and cost functional values for convergence :
Rs  = []
Js  = []
J1s = []
J2s = []
Ms  = []

#===============================================================================
# generate initial traction field :
d3model.init_T(d3model.T_surface)
d3model.init_beta_SIA()
#d3model.init_beta(1e4)
#bedmodel.init_beta_SIA()
#bedmodel.init_T(bedmodel.T_surface)

#mom = MomentumDukowiczStokes(d3model, isothermal=False)
#mom = MomentumDukowiczBrinkerhoffStokes(d3model, isothermal=False)
#mom = MomentumDukowiczStokesReduced(d3model, isothermal=False)
mom = MomentumDukowiczBP(d3model, linear=False, isothermal=False)
#mom = MomentumBP(d3model, isothermal=False)
nrg = Enthalpy(d3model, transient=False, use_lat_bc=True, 
               epsdot_ftn=mom.strain_rate_tensor)

#===============================================================================
## derivative of objective function callback function : 
#d3model.set_out_dir(out_dir + 'u_inversion/')
#
#def deriv_cb(I, dI, beta):
#  d3model.assign_submesh_variable(beta_b, beta)
#  d3model.save_xdmf(beta_b, 'beta_control')
#
## post-adjoint-iteration callback function :
#def adj_post_cb_ftn():
#  mom.solve_params['solve_vert_velocity'] = True
#  mom.solve(annotate=False)
#
#  # save the optimal velocity and beta fields for viewing with paraview :
#  d3model.assign_submesh_variable(Us,     d3model.U3)
#  d3model.assign_submesh_variable(beta_b, d3model.beta)
#  d3model.save_xdmf(Us,     'U_opt')
#  d3model.save_xdmf(beta_b, 'beta_opt')
#
## after every completed adjoining, save the state of these functions :
#adj_save_vars = [d3model.beta, d3model.U3]
#
## form the cost functional :
#mom.form_obj_ftn(integral=d3model.GAMMA_U_GND, kind='log_L2_hybrid', 
#                 g1=0.01, g2=5000)
#
## form the regularization functional :
#mom.form_reg_ftn(d3model.beta, integral=d3model.GAMMA_B_GND, kind='TV', 
#                 alpha=1.0)
#
### post-thermo-solve callback function :
##def tmc_cb_ftn():
##  nrg.solve_basal_melt_rate()
##  d3model.assign_submesh_variable(Tb,   d3model.T)
##  d3model.assign_submesh_variable(Us,   d3model.U3)
##  d3model.assign_submesh_variable(Wb,   d3model.W)
##  d3model.assign_submesh_variable(Mb,   d3model.Mb)
##  d3model.save_xdmf(Tb,   'Tb')
##  d3model.save_xdmf(Us,   'Us')
##  d3model.save_xdmf(Wb,   'Wb')
##  d3model.save_xdmf(Mb,   'Mb')
##  d3model.save_xdmf(d3model.T, 'T')
##  d3model.save_xdmf(d3model.W, 'W')
##
##d3model.thermo_solve(mom, nrg, callback=tmc_cb_ftn)
#mom.linearize_viscosity()
#
## optimize for beta :
#mom.optimize_u_ob(control           = d3model.beta,
#                  bounds            = (1e-5, 1e7),
#                  method            = 'ipopt',
#                  adj_iter          = 20,
#                  adj_save_vars     = adj_save_vars,
#                  adj_callback      = deriv_cb,
#                  post_adj_callback = adj_post_cb_ftn)
#
#sys.exit(0)

#===============================================================================
#mom.solve(annotate=False)
#fU   = HDF5File(mpi_comm_world(), out_dir + 'U3.h5', 'w')
#d3model.save_hdf5(d3model.U3, fU)
#fU.close()
#sys.exit(0)
#nrg.solve_divide(annotate=False)
#d3model.save_xdmf(d3model.theta_app, 'theta_app')

#===============================================================================
#d3model.set_out_dir(out_dir + 'W_L_curve_no_reg_linear/')
#
## save these for later analysis :  
#q_fric = project(nrg.q_fric, annotate=False)
#d3model.save_xdmf(q_fric, 'q_fric')
#
## number of digits for saving variables :
#iterations = 150
#gamma      = 2.5e7
#n_i        = len(str(iterations))
#
## derivative of objective function callback function : 
#def deriv_cb(I, dI, alpha):
#  d3model.assign_submesh_variable(a_b, alpha)
#  d3model.save_xdmf(a_b, 'alpha_control')
#      
## objective gradient callback function :
#def post_cb():
#  nrg.partition_energy()
#  nrg.solve_basal_melt_rate()
#  d3model.save_xdmf(d3model.alpha, 'alpha_opt')
#  d3model.save_xdmf(d3model.theta, 'theta_opt')
#  d3model.save_xdmf(d3model.T,     'T_opt')
#  d3model.save_xdmf(d3model.W,     'W_opt')
#  d3model.save_xdmf(d3model.Mb,    'Mb_opt')
#
#adj_kwargs = {'iterations'   : iterations,
#              'gamma'        : gamma,
#              'reg_kind'     : 'TV',
#              'method'       : 'ipopt',
#              'adj_callback' : deriv_cb}
#
##alphas = [1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6]  # TV obj
#alphas = [1e5, 1e6, 1e7, 2.5e7, 5e7, 7.5e7]
#
#Lc_kwargs = {'alphas'        : alphas,
#             'physics'       : nrg,
#             'control'       : d3model.alpha,
#             'int_domain'    : d3model.GAMMA_B_GND,
#             'adj_ftn'       : nrg.optimize_water_flux,
#             'adj_kwargs'    : adj_kwargs,
#             'reg_kind'      : 'TV',
#             'pre_callback'  : None,
#             'post_callback' : post_cb}
#
#nrg.form_obj_ftn(kind='L2')
#
##d3model.L_curve(**Lc_kwargs)
#
#nrg.optimize_water_flux(**adj_kwargs)
#  
#nrg.partition_energy()
#nrg.solve_basal_melt_rate()
#d3model.save_xdmf(d3model.alpha, 'alpha_opt')
#d3model.save_xdmf(d3model.theta, 'theta_opt')
#d3model.save_xdmf(d3model.T,     'T_opt')
#d3model.save_xdmf(d3model.W,     'W_opt')
#d3model.save_xdmf(d3model.Mb,    'Mb_opt')
#
#sys.exit(0)

#===============================================================================
#d3model.set_out_dir(out_dir + 'u_L_curve/')
#
## derivative of objective function callback function : 
#def deriv_cb(I, dI, beta):
#  # calculate the L_inf norm of misfit :
#  mom.calc_misfit(d3model.GAMMA_U_GND)
#  d3model.assign_submesh_variable(beta_b, beta)
#  d3model.save_xdmf(beta_b, 'beta_control')
#
## post-adjoint-iteration callback function :
#def post_cb():
#  # re-solve the momentum equations with vertical velocity and optimal beta :
#  m_params['solve_vert_velocity'] = True
#  mom.solve(annotate=False)
#
#  # save the optimal velocity and beta fields for viewing with paraview :
#  d3model.save_xdmf(d3model.U3,   'U_opt')
#  d3model.save_xdmf(d3model.beta, 'beta_opt')
#
## form the cost functional :
#mom.form_obj_ftn(integral=d3model.GAMMA_U_GND, kind='log_L2_hybrid', 
#                 g1=0.01, g2=5000)
#
## number of digits for saving variables :
#iterations = 2
#gamma      = 1e10
#n_i        = len(str(iterations))
#
## after every completed adjoining, save the state of these functions :
#adj_save_vars = [d3model.beta, d3model.U3]
#
#uop_kwargs = {'control'           : d3model.beta,
#              'bounds'            : (1e-5, 1e7),
#              'method'            : 'ipopt',
#              'adj_iter'          : 10,
#              'adj_save_vars'     : adj_save_vars,
#              'adj_callback'      : deriv_cb,
#              'post_adj_callback' : post_cb}
#      
#alphas = [0.5, 1.0, 1.5]
#
#Lc_kwargs = {'alphas'        : alphas,
#             'physics'       : mom,
#             'control'       : d3model.beta,
#             'int_domain'    : d3model.GAMMA_B_GND,
#             'adj_ftn'       : mom.optimize_u_ob,
#             'adj_kwargs'    : uop_kwargs,
#             'reg_kind'      : 'TV',
#             'pre_callback'  : None,
#             'post_callback' : None}
#
#d3model.L_curve(**Lc_kwargs)
#
#sys.exit(0)

#===============================================================================
#nrg.generate_approx_theta(init=True, annotate=False)
#d3model.save_xdmf(d3model.theta_app, 'theta_ini')
#d3model.save_xdmf(d3model.T,         'T_ini')
#d3model.save_xdmf(d3model.W,         'W_ini')

new_dir = 'tmc_inversion_cont_kappa_TV_beta_reg_10/04/hdf5/'
#new_dir = 'tmc_inversion_cont_kappa_TV_beta_reg_10_a_var/'

#d3model.set_out_dir(out_dir + new_dir)
#d3model.set_out_dir(out_dir + 'rstrt_disc_kappa/')
d3model.set_out_dir(out_dir + 'rstrt_alpha_1e8_regularized/')

fini = HDF5File(mpi_comm_world(),
                out_dir + new_dir + 'inverted_04.h5', 'r')
#fini = HDF5File(mpi_comm_world(), 
#                out_dir + new_dir + '01/hdf5/inverted_01.h5', 'r')

d3model.init_T(fini)
d3model.init_W(fini)
d3model.init_Wb_flux(fini)
d3model.init_Mb(fini)
d3model.init_alpha(fini)
#d3model.init_PE(fini)
#d3model.init_W_int(fini)
#d3model.init_U(fini)
d3model.init_beta(fini)
#d3model.init_theta(fini)

# post-thermo-solve callback function :
def tmc_cb_ftn():
  nrg.solve_basal_melt_rate()
  nrg.solve_basal_water_flux()
  nrg.calc_PE()#avg=True)
  nrg.calc_internal_water()
  #d3model.save_xdmf(d3model.T,          'T')
  #d3model.save_xdmf(d3model.W,          'W')
  #d3model.save_xdmf(d3model.Wb_flux,    'Wb_flux')
  #d3model.save_xdmf(d3model.Mb,         'Mb')
  #d3model.save_xdmf(d3model.alpha,      'alpha')
  #d3model.save_xdmf(d3model.PE,         'PE')
  #d3model.save_xdmf(d3model.W_int,      'W_int')
  #d3model.save_xdmf(d3model.U3,         'U_opt')
  #d3model.save_xdmf(d3model.beta,       'beta_opt')

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
  # solve for optimal vertical velocity :
  mom.solve_vert_velocity(annotate=False)

# after every completed adjoining, save the state of these functions :
adj_save_vars = [d3model.T,
                 d3model.W,
                 d3model.Wb_flux,
                 d3model.Mb,
                 d3model.alpha,
                 d3model.PE,
                 d3model.W_int,
                 d3model.U3,
                 d3model.beta,
                 d3model.theta]

# the initial step saves everything :
ini_save_vars = adj_save_vars + [d3model.Ubar, d3model.U_ob]

# form the objective functional for water-flux optimization :
nrg.form_obj_ftn(kind='L2')

# form regularization for water-flux :
nrg.form_reg_ftn(d3model.alpha, integral=d3model.GAMMA_B_GND,
                 kind='TV', alpha=1e8)

# form the cost functional :
mom.form_obj_ftn(integral=d3model.GAMMA_U_GND, kind='log_L2_hybrid', 
                 g1=0.01, g2=5000)

# form the regularization functional :
mom.form_reg_ftn(d3model.beta, integral=d3model.GAMMA_B_GND,
                 kind='TV', alpha=10.0)
#mom.form_reg_ftn(d3model.beta, integral=d3model.GAMMA_B_GND,
#                  kind='Tikhonov', alpha=1e-6)

wop_kwargs = {'max_iter'            : 500, 
              'bounds'              : (0.0, 100),
              'method'              : 'ipopt',
              'adj_callback'        : None}
                                    
tmc_kwargs = {'momentum'            : mom,
              'energy'              : nrg,
              'wop_kwargs'          : wop_kwargs,
              'callback'            : tmc_cb_ftn, 
              'atol'                : 1e2,
              'rtol'                : 1e0,
              'max_iter'            : 10,
              'post_tmc_save_vars'  : ini_save_vars}
                                    
uop_kwargs = {'control'             : d3model.beta,
              'bounds'              : (1e-5, 1e7),
              'method'              : 'ipopt',
              'max_iter'            : 250,
              'adj_save_vars'       : None,
              'adj_callback'        : None,
              'post_adj_callback'   : adj_post_cb_ftn}
                                    
ass_kwargs = {'iterations'          : 4,
              'tmc_kwargs'          : tmc_kwargs,
              'uop_kwargs'          : uop_kwargs,
              'initialize'          : False,
              'incomplete'          : True,
              'ini_save_vars'       : ini_save_vars,
              'post_iter_save_vars' : adj_save_vars,
              'post_ini_callback'   : None,
              'starting_i'          : 2}

# assimilate ! :
#d3model.assimilate_U_ob(**ass_kwargs) 

# or restart and thermo_solve :
d3model.thermo_solve(**tmc_kwargs)



 
