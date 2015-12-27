from varglas          import *
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

# painAmplifier IPOpt : Total time to compute: 47:35:04
# painAmplifier bfgs  : Total time to compute: 41:21:00


# set the relavent directories :
var_dir = 'dump/vars_jakobshavn_crude/'  # directory from gen_vars.py
out_dir = 'dump/jakob_crude/'            # base directory to save

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

fUin = HDF5File(mpi_comm_world(), out_dir + 'U3.h5', 'r')
d3model.init_U(fUin)

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
d3model.init_beta_SIA()
#d3model.init_beta(1e4)
#bedmodel.init_beta_SIA()
#bedmodel.init_T(bedmodel.T_surface)

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

#d3model.thermo_solve(mom, nrg, callback=None, max_iter=1)
#fU   = HDF5File(mpi_comm_world(), out_dir + 'U3.h5', 'w')
#d3model.save_hdf5(d3model.U3, fU)
#fU.close()
#nrg.solve_divide(annotate=False)
#d3model.save_xdmf(d3model.theta_app, 'theta_app')
      
def eval_cb(I, alpha):
  s    = '::: adjoint objective eval post callback function :::'
  print_text(s)
  print_min_max(I,    'I')
  print_min_max(alpha, 'alpha')

# objective gradient callback function :
def deriv_cb(I, dI, alpha):
  s    = '::: adjoint obj. gradient post callback function :::'
  print_text(s)
  print_min_max(dI,    'dI/dalpha')

d3model.init_alpha(0.3)

nrg.solve(annotate=True)

theta   = d3model.theta
theta_m = d3model.theta_melt
L       = d3model.L

W_c  = conditional( le(theta, theta_m), 0.0, 1.0)
#X    = W_c * (W_w + DOLFIN_EPS) / (Wmax + DOLFIN_EPS) * Mb * rho * L
J    = W_c * ( (theta - theta_m)/L - 0.03 ) * dx

m = Control(d3model.alpha, value=d3model.alpha)
      
F = ReducedFunctional(Functional(J), m, eval_cb_post=eval_cb,
                      derivative_cb_post=deriv_cb)
        
out = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=(0,1),
               options={"disp"    : True,
                        "maxiter" : 1000,
                        "gtol"    : 1e-5})
a_opt = out[0]

d3model.init_alpha(a_opt)

nrg.solve()
d3model.save_xdmf(d3model.T, 'T')
d3model.save_xdmf(d3model.W, 'W')
sys.exit(0)

#nrg.generate_approx_theta(init=True, annotate=False)
#d3model.save_xdmf(d3model.theta_app, 'theta_ini')
#d3model.save_xdmf(d3model.T,         'T_ini')
#d3model.save_xdmf(d3model.W,         'W_ini')
  
# post-thermo-solve callback function :
def tmc_cb_ftn():
  nrg.solve_basal_melt_rate()
  d3model.assign_submesh_variable(Tb,   d3model.T)
  d3model.assign_submesh_variable(Us,   d3model.U3)
  d3model.assign_submesh_variable(Wb,   d3model.W)
  d3model.assign_submesh_variable(Mb,   d3model.Mb)
  d3model.save_xdmf(Tb,   'Tb')
  d3model.save_xdmf(Us,   'Us')
  d3model.save_xdmf(Wb,   'Wb')
  d3model.save_xdmf(Mb,   'Mb')
  d3model.save_xdmf(d3model.T, 'T')
  d3model.save_xdmf(d3model.W, 'W')

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

# after every completed adjoining, save the state of these functions :
adj_save_vars = [d3model.beta,
                 d3model.U3,
                 d3model.T,
                 d3model.W,
                 d3model.theta,
                 d3model.Mb]

# the initial step saves everything :
ini_save_vars = adj_save_vars + [d3model.Ubar, d3model.U_ob]

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
                        adj_save_vars     = adj_save_vars,
                        tmc_callback      = tmc_cb_ftn,
                        post_ini_callback = None,
                        post_adj_callback = adj_post_cb_ftn,
                        adj_callback      = deriv_cb,
                        tmc_rtol          = 1e0,
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



