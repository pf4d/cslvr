from varglas          import *
from varglas.energy   import Enthalpy 
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys


# get the input args :
i       = 0
dir_b   = 'dump/low/0'     # directory to save

# set the relavent directories (complicated, right?!) :
var_dir = 'dump/vars_low/'
in_dir  = dir_b + str(i-1) + '/inverted/xml/'
out_dir = dir_b + str(i) + '/thermo_solve/'
bv_dir  = dir_b + str(i) + '/balance_velocity/'

# load the meshes :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')

# get the bed mesh :
mesh = Mesh()
fmeshes.read(mesh, 'bedmesh', False)

# create 3D model for stokes solves, and 2D model for balance velocity :
d3model = D3Model(fdata, out_dir)
d2model = D2Model(mesh,  bv_dir)

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
d3model.init_u_lat(0.0)
d3model.init_v_lat(0.0)

# 2D model gets balance-velocity appropriate variables initialized :
d2model.assign_submesh_variable(d2model.S,    d3model.S)
d2model.assign_submesh_variable(d2model.B,    d3model.B)
d2model.assign_submesh_variable(d2model.adot, d3model.adot)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.save_pvd(d2model.Ubar, 'Ubar')
d2model.save_xml(d2model.Ubar, 'Ubar')

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
  d3model.init_T(in_dir + 'T.xml')          # temp
  d3model.init_W(in_dir + 'W.xml')          # water
  d3model.init_beta(in_dir + 'beta.xml')    # friction
  d3model.init_E_shf(in_dir + 'E_shf.xml')  # enhancement
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
m_params  = {'solver'               : nparams,
             'solve_vert_velocity'  : True,
             'solve_pressure'       : True,
             'vert_solve_method'    : 'mumps'}

e_params  = {'solver'               : 'mumps',
             'use_surface_climate'  : False}

mom = MomentumDukowiczStokesReduced(d3model, m_params, isothermal=False)
#mom = MomentumBP(d3model, m_params, isothermal=False)
nrg = Enthalpy(d3model, e_params)

d3model.save_pvd(d3model.beta, 'beta0')
d3model.save_pvd(d3model.U_ob, 'U_ob')

def cb_ftn():
  nrg.solve_basal_melt_rate()
  #nrg.calc_bulk_density()
  d3model.save_pvd(d3model.U3,    'U3')
  #d3model.save_pvd(d3model.p,     'p')
  d3model.save_pvd(d3model.theta, 'theta')
  d3model.save_pvd(d3model.T,     'T')
  d3model.save_pvd(d3model.W,     'W')
  d3model.save_pvd(d3model.Mb,    'Mb')
  #d3model.save_pvd(d3model.rho_b, 'rho_b')

d3model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=1e-6, max_iter=15)

d3model.save_xml(d3model.T,                       'T')
d3model.save_xml(d3model.W,                       'W')
d3model.save_xml(interpolate(d3model.u, d3model.Q), 'u')
d3model.save_xml(interpolate(d3model.v, d3model.Q), 'v')
d3model.save_xml(interpolate(d3model.w, d3model.Q), 'w')
d3model.save_xml(d3model.beta,                    'beta')
d3model.save_xml(d3model.Mb,                      'Mb')



