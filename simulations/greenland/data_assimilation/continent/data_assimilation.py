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
dir_b   = 'dump/high_da_bfgs_SIA0_SR/0'     # directory to save

# set the output directory :
out_dir = dir_b + str(i)
in_dir  = 'dump/vars_high/'

f = HDF5File(mpi_comm_world(), in_dir + 'state.h5', 'r')

mesh   = Mesh()
ff     = MeshFunction('size_t', mesh)
cf     = MeshFunction('size_t', mesh)
ff_acc = MeshFunction('size_t', mesh)
f.read(mesh,    'mesh', False)
f.read(ff,      'ff')
f.read(cf,      'cf')
f.read(ff_acc,  'ff_acc')

model = D3Model(mesh, out_dir + '/thermo_solve/pvd/')
model.set_subdomains(ff, cf, ff_acc)

model.init_S(f)
model.init_B(f)
model.init_mask(f)
model.init_q_geo(model.ghf)
model.init_T_surface(f)
model.init_adot(f)
model.init_U_ob(f, f)
model.init_E(1.0)

# use T0 and beta0 from the previous run :
if i > 0:
  model.init_T(dir_b + str(i-1) + '/inverted/xml/T.xml')          # temp
  model.init_W(dir_b + str(i-1) + '/inverted/xml/W.xml')          # water
  model.init_beta(dir_b + str(i-1) + '/inverted/xml/beta.xml')    # friction
  model.init_E_shf(dir_b + str(i-1) + '/inverted/xml/E_shf.xml')  # enhancement
else:
  model.init_T(model.T_surface)
  model.init_beta_SIA()

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

mom = MomentumDukowiczStokesReduced(model, m_params, isothermal=False)
nrg = Enthalpy(model, e_params)

model.save_pvd(model.beta, 'beta0')
model.save_pvd(model.U_ob, 'U_ob')

def cb_ftn():
  #nrg.solve_basal_melt_rate()
  #nrg.calc_bulk_density()
  model.save_pvd(model.U3,    'U3')
  #model.save_pvd(model.p,     'p')
  model.save_pvd(model.theta, 'theta')
  model.save_pvd(model.T,     'T')
  model.save_pvd(model.W,     'W')
  #model.save_pvd(model.Mb,    'Mb')
  #model.save_pvd(model.rho_b, 'rho_b')

model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=1e-6, max_iter=15)

model.set_out_dir(out_dir = out_dir + '/thermo_solve/xml/')
model.save_xml(model.T,                       'T')
model.save_xml(model.W,                       'W')
model.save_xml(interpolate(model.u, model.Q), 'u')
model.save_xml(interpolate(model.v, model.Q), 'v')
model.save_xml(interpolate(model.w, model.Q), 'w')
model.save_xml(model.beta,                    'beta')
model.save_xml(model.Mb,                      'Mb')
model.save_xml(model.E_shf,                   'E_shf')

# invert for basal friction over grounded ice :
nparams['newton_solver']['relaxation_parameter'] = 1.0
nparams['newton_solver']['relative_tolerance']   = 1e-8
nparams['newton_solver']['maximum_iterations']   = 3

mom = MomentumDukowiczStokesReduced(model, m_params, isothermal=False, 
                                    linear=True)
mom.solve(annotate=True)

model.set_out_dir(out_dir = out_dir + '/inverted/pvd/')
  
J = mom.form_obj_ftn(integral=model.dSrf_s, kind='log_lin_hybrid', 
                     g1=0.01, g2=1000)
R = mom.form_reg_ftn(model.beta, integral=model.dGnd, kind='Tikhonov', 
                     alpha=1.0)
I = J + R

controls = File(model.out_dir + "control_viz/beta_control.pvd")
beta_viz = Function(model.Q, name="beta_control")
  
def eval_cb(I, beta):
  #mom.print_eval_ftns()
  #print_min_max(mom.U, 'U')
  print_min_max(I,    'I')
  print_min_max(beta, 'beta')

def deriv_cb(I, dI, beta):
  #print_min_max(I,     'I')
  print_min_max(dI,    'dI/dbeta')
  #print_min_max(beta,  'beta')
  beta_viz.assign(beta)
  controls << beta_viz

def hessian_cb(I, ddI, beta):
  print_min_max(ddI, 'd/db dI/db')

m = FunctionControl('beta')
F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                      derivative_cb_post = deriv_cb,
                      hessian_cb = hessian_cb)

b_opt = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=(1e-6, 1e6),
                 options={"disp"    : True,
                          "maxiter" : 1000,
                          "gtol"    : 1e-5,
                          "factr"   : 1e7})

#problem = MinimizationProblem(F, bounds=(0, 4000))
#parameters = {"tol"                : 1e8,
#              "acceptable_tol"     : 1000.0,
#              "maximum_iterations" : 1000,
#              "linear_solver"      : "ma57"}
#
#solver = IPOPTSolver(problem, parameters=parameters)
#b_opt = solver.solve()
print_min_max(b_opt, 'b_opt')

model.set_out_dir(out_dir = out_dir + '/inverted/xml/')

u,v,w = model.U3.split(True)

model.save_xml(model.T,                       'T')
model.save_xml(model.W,                       'W')
model.save_xml(interpolate(model.u, model.Q), 'u')
model.save_xml(interpolate(model.v, model.Q), 'v')
model.save_xml(interpolate(model.w, model.Q), 'w')
model.save_xml(model.beta,                    'beta')
model.save_xml(model.Mb,                      'Mb')
model.save_xml(model.E_shf,                   'E_shf')

    

