from cslvr import *
from dolfin_adjoint import *

# use inexact integration :
parameters['form_compiler']['quadrature_degree']  = 4

# directories for saving data :
mdl_odr = 'BP'
reg_typ = None#'tv_tik_hybrid'#'tv'#'tikhonov'#
opt_met = 'ipopt'#'l_bfgs_b'#

var_dir = './dump/vars/'
#out_dir = './dump/results/' + mdl_odr +'/'+ opt_met +'/'+ reg_typ +'/'
out_dir = './dump/results/' + mdl_odr + '/u_opt/'#'/tmc_opt/'

# create the output directory if it does not exist :
d       = os.path.dirname(out_dir)
if not os.path.exists(d):
	os.makedirs(d)

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
d3model.set_srf_mesh(fmeshes)
d3model.set_bed_mesh(fmeshes)
d3model.set_lat_mesh(fmeshes)
d3model.set_dvd_mesh(fmeshes)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_sigma(fdata)
d3model.init_mask(fdata)
d3model.init_q_geo(d3model.ghf)
d3model.init_T_surface(fdata)
d3model.init_S_ring(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
#d3model.init_time_step(1e-6)
d3model.init_E(1.0)
d3model.init_W(0.0)
d3model.init_Wc(0.03)
d3model.init_T(d3model.T_surface)
d3model.init_k_0(1e-3)
d3model.solve_hydrostatic_pressure()
d3model.form_energy_dependent_rate_factor()
#d3model.init_A(2e-17)               # isothermal flow-rate factor

d3model.save_xdmf(d3model.U_ob,   'U_ob')
d3model.save_xdmf(d3model.ff,     'ff')

#frstrt = HDF5File(mpi_comm_world(), out_dir + '01/inverted.h5', 'r')
#d3model.init_T(frstrt)
#d3model.init_W(frstrt)
#d3model.init_B_ring(frstrt)
#d3model.init_alpha(frstrt)
#d3model.init_U(frstrt)
#d3model.init_p(frstrt)
#d3model.init_theta(frstrt)

# create a 2D model for balance-velocity :
bedmodel = D2Model(d3model.bedmesh, out_dir, kind='balance')

bedmodel.assign_submesh_variable(bedmodel.S,      d3model.S)
bedmodel.assign_submesh_variable(bedmodel.B,      d3model.B)
bedmodel.assign_submesh_variable(bedmodel.S_ring,   d3model.S_ring)

# solve the balance velocity :
bv = BalanceVelocity(bedmodel, kappa=5.0, stabilization_method='GLS')
d  = (-bedmodel.S.dx(0), -bedmodel.S.dx(1))  # direction of imposed flow
bv.solve_direction_of_flow(d)
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

# generate initial traction field :
#d3model.init_beta_SIA()
d3model.init_beta(1e3)

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
	mom = MomentumBP(d3model, linear=False)
elif mdl_odr == 'BP_duk':
	mom = MomentumDukowiczBP(d3model, linear=False)
elif mdl_odr == 'RS':
	mom = MomentumDukowiczStokesReduced(d3model, linear=False)
elif mdl_odr == 'FS_duk':
	mom = MomentumDukowiczStokes(d3model, linear=False)
elif mdl_odr == 'FS_stab':
	mom = MomentumNitscheStokes(d3model, linear=False, stabilized=True)
elif mdl_odr == 'FS_th':
	mom = MomentumNitscheStokes(d3model, linear=False, stabilized=False)

#momTMC = MomentumDukowiczStokes(d3model, linear=False)
momTMC = mom
nrg    = Enthalpy(d3model, momTMC, transient=False, use_lat_bc=True,
                  energy_flux_mode='B_ring')

#frstrt = HDF5File(mpi_comm_world(), out_dir + '02/u_opt.h5', 'r')
#d3model.set_out_dir(out_dir + '02/')
#d3model.init_U(frstrt)
#d3model.init_beta(frstrt)

# thermo-solve callback function :
def tmc_cb_ftn(counter):
	nrg.calc_PE()#avg=True)
	nrg.calc_vert_avg_strain_heat()
	nrg.calc_vert_avg_W()
	nrg.calc_temp_rat()
	nrg.solve_basal_melt_rate()

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
	# solve for optimal vertical velocity :
	mom.solve_vert_velocity(annotate=False)

# after every completed adjoining, save the state of these functions :
adj_save_vars = [d3model.T,
                 d3model.W,
                 d3model.B_ring,
                 d3model.Mb,
                 d3model.alpha,
                 d3model.alpha_int,
                 d3model.PE,
                 d3model.Wbar,
                 d3model.Qbar,
                 d3model.temp_rat,
                 d3model.u,
                 d3model.p,
                 d3model.beta,
                 d3model.theta]

u_opt_save_vars = [d3model.beta, d3model.u]
w_opt_save_vars = [d3model.B_ring,   d3model.theta]

tmc_save_vars   = [d3model.T,
                   d3model.W,
                   d3model.B_ring,
                   d3model.Mb,
                   d3model.alpha,
                   d3model.alpha_int,
                   d3model.PE,
                   d3model.Wbar,
                   d3model.Qbar,
                   d3model.temp_rat,
                   d3model.u,
                   d3model.p,
                   d3model.beta,
                   d3model.theta]

# define the integral measure for functionals :
J_measure = d3model.dGamma_s     # cost measure
R_measure = d3model.dGamma_b     # regularization measure

# form the cost functional :
J_log = mom.form_cost_ftn(u        = mom.get_unknown(),
                          u_ob     = [d3model.u_ob,  d3model.v_ob],
                          integral = J_measure,
                          kind     = 'log')

# form the cost functional :
J_l2  = mom.form_cost_ftn(u        = mom.get_unknown(),
                          u_ob     = [d3model.u_ob,  d3model.v_ob],
                          integral = J_measure,
                          kind     = 'l2')

# form the cost functional :
J_l1  = mom.form_cost_ftn(u        = mom.get_unknown(),
                          u_ob     = [d3model.u_ob,  d3model.v_ob],
                          integral = J_measure,
                          kind     = 'l1')

# form the cost functional :
J_rat = mom.form_cost_ftn(u        = mom.get_unknown(),
                          u_ob     = [d3model.u_ob,  d3model.v_ob],
                          integral = J_measure,
                          kind     = 'ratio')

# form the total cost functional :
J = 1e10*J_log

# form the regularization functional :
if reg_typ == 'tv':
	R = mom.form_reg_ftn(d3model.beta, integral=R_measure, kind='tv')

elif reg_typ == 'tikhonov':
	R = mom.form_reg_ftn(d3model.beta, integral=R_measure, kind='tikhonov')

elif reg_typ == 'tv_tik_hybrid':
	R_tv  = mom.form_reg_ftn(d3model.beta, integral=R_measure, kind='tv')
	R_tik = mom.form_reg_ftn(d3model.beta, integral=R_measure, kind='tikhonov')
	R     = R_tv + 20*R_tik

elif reg_typ is None:
	R = 0.0

# upper and lower friction bounds :
beta_max = 1e7
beta_min = DOLFIN_EPS

# form bounds for friction :
upper             = Function(d3model.Q)
lower             = Function(d3model.Q)
upper.vector()[:] = beta_max#1e-2
lower.vector()[:] = beta_min

#upper_v = upper.vector().get_local()
#lower_v = lower.vector().get_local()
#
#m = vertex_to_dof_map(d3model.Q)
#
#for i in np.where(d3model.ff.array() == d3model.GAMMA_B_GND)[0]:
#  v_i_dofs = m[Facet(d3model.mesh, i).entities(0)]
#  if np.all(v_i_dofs < len(upper_v)):
#    upper_v[v_i_dofs] = beta_max
#    lower_v[v_i_dofs] = beta_min
#
# FIXME: calling ``vector().set_local()`` breaks dolfin_adjoint :
#upper.vector().set_local(upper_v)
#upper.vector().apply('insert')
#lower.vector().set_local(upper_v)
#lower.vector().apply('insert')

# form the objective functional for water-flux optimization :
nrg.form_cost_ftn(kind='L2')

wop_kwargs = {'max_iter'            : 350,
              'bounds'              : (0.0, 100.0),
              'method'              : 'ipopt',
              'adj_save_vars'       : w_opt_save_vars,
              'adj_callback'        : None}

tmc_kwargs = {'momentum'            : momTMC,
              'energy'              : nrg,
              'wop_kwargs'          : wop_kwargs,
              'callback'            : tmc_cb_ftn,
              'atol'                : 1e2,
              'rtol'                : 1e0,
              'max_iter'            : 10,
              'iter_save_vars'      : None,
              'post_tmc_save_vars'  : tmc_save_vars,
              'starting_i'          : 1}

uop_kwargs = {'u'                   : mom.get_unknown(),
              'u_ob'                : [d3model.u_ob, d3model.v_ob],
              'I'                   : J + R,
              'control'             : d3model.beta,
              'J_measure'           : J_measure,
              'R_measure'           : R_measure,
              'bounds'              : (lower, upper),
              'method'              : opt_met,
              'max_iter'            : 1000,#3000
              'adj_save_vars'       : u_opt_save_vars,
              'adj_callback'        : None,
              'post_adj_callback'   : adj_post_cb_ftn}

ass_kwargs = {'momentum'            : mom,
              'beta_i'              : d3model.beta.copy(True),
              'max_iter'            : 10,
              'tmc_kwargs'          : tmc_kwargs,
              'uop_kwargs'          : uop_kwargs,
              'atol'                : 1.0,
              'rtol'                : 1e-4,
              'initialize'          : True,
              'incomplete'          : True,
              'post_iter_save_vars' : adj_save_vars,
              'post_ini_callback'   : None,
              'starting_i'          : 1}

# FIXME: the pressure goes wacky over very thin regions :
mom.solve_params['solve_pressure'] = False

# solving the incomplete adjoint is more efficient, so first derive the
# velocity-dependant viscosity :
mom.solve()                 # first, initialize the velocity for viscosity
mom.linearize_viscosity()   # remove velocity dependence from the viscosity

# by not solving these variables, dolfin_adjoint will also not solve them
# each interation of the DA algorithm :
# NOTE: you might want to solve these, if you were going to perform an
#       adjoint sensitivity analysis of the energy balance, or something like
#       that where the pressure would also be an important factor.
mom.solve_params['solve_vert_velocity'] = False
mom.solve_params['solve_pressure']      = False

# optimize for beta :
mom.optimize(**uop_kwargs)

## regularization parameters :
#alphas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
#
#d3model.L_curve(alphas        = alphas,
#                control       = d3model.beta,
#                physics       = mom,
#                J             = J,
#                R             = R,
#                adj_kwargs    = uop_kwargs,
#                pre_callback  = None,
#                post_callback = None,
#                itr_save_vars = None)
#
## this function iterates through the directories and creates a plot :
#ftnl_a = plot_l_curve(out_dir=out_dir, control='beta')
#
## now re-solve for saving :
#mom.solve_params['solve_vert_velocity'] = True
#mom.solve_params['solve_pressure']      = True
#mom.solve()
#d3model.save_xdmf(d3model.u,   'U3_opt')
#d3model.save_xdmf(d3model.beta, 'beta_opt')
#
#mom.solve()
#
#fres = HDF5File(mpi_comm_world(), out_dir + 'momentum.h5', 'w')
#d3model.save_list_to_hdf5([d3model.u, d3model.p], fres)
#fres.close()

# or only thermo-mechanically couple :
#d3model.thermo_solve(**tmc_kwargs)

# thermo-mechanical data-assimilation (Cummings et al., 2016) !
#d3model.assimilate_U_ob(**ass_kwargs)



