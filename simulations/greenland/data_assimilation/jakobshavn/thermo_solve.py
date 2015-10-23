from varglas          import *
from varglas.energy   import Enthalpy 
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import sys

#set_log_active(False)
#set_log_level(PROGRESS)

# get the input args :
i       = 0
dir_b   = 'dump/jakob_da_ipopt_SIA0_SR/0'     # directory to save

# set the output directory :
out_dir = dir_b + str(i)
var_dir = 'dump/vars_jakobshavn/'
ts_dir  = dir_b + str(i-1) + '/thermo_solve/xml/'
in_dir  = dir_b + str(i-1) + '/inverted/xml/'

f = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')

model = D3Model(f, out_dir + '/thermo_solve_flux_variable_c_consistent_L/pvd/')
model.set_subdomains(f)

model.init_S(f)
model.init_B(f)
model.init_mask(f)
model.init_q_geo(model.ghf)
model.init_T_surface(f)
model.init_adot(f)
model.init_U_ob(f, f)
model.init_E(1.0)
model.init_u_lat(0.0)
model.init_v_lat(0.0)

# use T0 and beta0 from the previous run :
if i > 0:
  model.init_T(ts_dir + 'T.xml')          # temp
  model.init_W(ts_dir + 'W.xml')          # water
  model.init_beta(in_dir + 'beta.xml')    # friction
else:
  model.init_T(model.T_surface)
  #model.init_beta(1e4)
  model.init_beta_SIA()

nparams = {'newton_solver' : {'linear_solver'            : 'cg',
                              'preconditioner'           : 'hypre_amg',
                              'relative_tolerance'       : 1e-9,
                              'relaxation_parameter'     : 0.7,
                              'maximum_iterations'       : 30,
                              'error_on_nonconvergence'  : False}}
m_params  = {'solver'               : nparams,
             'solve_vert_velocity'  : True,
             'solve_pressure'       : False,
             'vert_solve_method'    : 'mumps'}

e_params  = {'solver'               : 'mumps',
             'use_surface_climate'  : False}

mom = MomentumDukowiczStokesReduced(model, m_params, isothermal=False)
#mom = MomentumBP(model, m_params, isothermal=False)
nrg = Enthalpy(model, e_params)

model.save_pvd(model.beta, 'beta0')
model.save_pvd(model.U_ob, 'U_ob')

def cb_ftn():
  nrg.solve_basal_melt_rate()
  #nrg.calc_bulk_density()
  model.save_pvd(model.U3,    'U3')
  #model.save_pvd(model.p,     'p')
  model.save_pvd(model.theta, 'theta')
  model.save_pvd(model.T,     'T')
  model.save_pvd(model.W,     'W')
  model.save_pvd(model.Mb,    'Mb')
  #model.save_pvd(model.rho_b, 'rho_b')
    
  gradB = as_vector([model.B.dx(0), model.B.dx(1), -1])
  dTdn  = project(dot(grad(model.T), gradB))
  model.save_pvd(dTdn, 'dTdn')

model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=1e-6, max_iter=15)

model.set_out_dir(out_dir = out_dir + '/thermo_solve_flux_variable_c_consistent_L/xml/')
model.save_xml(model.T,                       'T')
model.save_xml(model.W,                       'W')
model.save_xml(interpolate(model.u, model.Q), 'u')
model.save_xml(interpolate(model.v, model.Q), 'v')
model.save_xml(interpolate(model.w, model.Q), 'w')
model.save_xml(model.beta,                    'beta')
model.save_xml(model.Mb,                      'Mb')



