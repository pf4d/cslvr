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
dir_b   = 'dump/high_da/0'     # directory to save

# set the output directory :
out_dir = dir_b + str(i) + '/'
in_dir  = 'dump/vars_high/'

mesh   = Mesh(in_dir + 'mesh.xdmf')
Q      = FunctionSpace(mesh, 'CG', 1)
ff     = MeshFunction('size_t', mesh)
cf     = MeshFunction('size_t', mesh)
ff_acc = MeshFunction('size_t', mesh)

S        = Function(Q)
B        = Function(Q)
T_s      = Function(Q)
adot     = Function(Q)
mask     = Function(Q)
q_geo    = Function(Q)
u_ob     = Function(Q)
v_ob     = Function(Q)

f = HDF5File(mesh.mpi_comm(), in_dir + 'vars.h5', 'r')

f.read(S,        'S')
f.read(B,        'B')
f.read(T_s,      'T_s')
f.read(q_geo,    'q_geo')
f.read(adot,     'adot')
f.read(mask,     'mask')
f.read(ff,       'ff')
f.read(cf,       'cf')
f.read(ff_acc,   'ff_acc')
f.read(u_ob,     'u')
f.read(v_ob,     'v')

model = D3Model(out_dir = out_dir)
model.set_mesh(mesh)
model.set_subdomains(ff, cf, ff_acc)
model.generate_function_spaces(use_periodic = False)

model.init_S(S)
model.init_B(B)
model.init_mask(mask)
model.init_q_geo(model.ghf)
model.init_T_surface(T_s)
model.init_adot(adot)
model.init_U_ob(u_ob, v_ob)
model.init_E(1.0)

# use T0 and beta0 from the previous run :
if i > 0:
  model.init_T(dir_b + str(i-1) + '/T.xml')             # temp
  model.init_W(dir_b + str(i-1) + '/W.xml')             # water
  model.init_beta(dir_b + str(i-1) + '/beta.xml')       # friction
  model.init_E_shf(dir_b + str(i-1) + '/E_shf.xml')      # enhancement
else:
  model.init_T(model.T_w(0) - 30.0)
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
  #model.save_pvd(model.W,     'W')
  #model.save_pvd(model.Mb,    'Mb')
  #model.save_pvd(model.rho_b, 'rho_b')

model.thermo_solve(mom, nrg, callback=cb_ftn, rtol=1e-6, max_iter=15)

# invert for basal friction over grounded ice :
nparams['newton_solver']['relaxation_parameter'] = 1.0
nparams['newton_solver']['relative_tolerance']   = 1e-8
nparams['newton_solver']['maximum_iterations']   = 3

mom = MomentumDukowiczStokesReduced(model, m_params, isothermal=False, 
                                    linear=True)
mom.solve(annotate=True)

model.set_out_dir(out_dir = out_dir + 'inverted/')
  
J = mom.form_obj_ftn('log_lin_hybrid', integral=model.dSrf_s,
                     g1=0.01, g2=1000)
R = mom.form_reg_ftn(model.beta, 'Tikhonov', integral=model.dGnd,
                     alpha=1.0)
I = J + R

controls = File(model.out_dir + "control_viz/beta_control.pvd")
beta_viz = Function(model.Q, name="beta_control")
  
def eval_cb(I, beta):
  #       commented out because the model variables are not updated by 
  #       dolfin-adjoint (yet) :
  #mom.print_eval_ftns()
  #print_min_max(mom.U, 'U')
  print_min_max(I,    'I')
  print_min_max(beta, 'beta')

def deriv_cb(I, dI, beta):
  print_min_max(I,     'I')
  print_min_max(dI,    'dI/dbeta')
  print_min_max(beta,  'beta')
  beta_viz.assign(beta)
  controls << beta_viz

def hessian_cb(I, ddI, beta):
  print_min_max(ddI, 'd/db dI/db')

m = FunctionControl('beta')
F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                      derivative_cb_post = deriv_cb,
                      hessian_cb = hessian_cb)

b_opt = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=(0, 4000),
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

model.set_out_dir(out_dir = out_dir + 'xml/')

u,v,w = model.U3.split(True)

model.save_xml(model.T,     'T')
model.save_xml(model.W,     'W')
model.save_xml(u,           'u')
model.save_xml(v,           'v')
model.save_xml(w,           'w')
model.save_xml(model.beta,  'beta')
model.save_xml(model.Mb,    'Mb')
model.save_xml(model.E_shf, 'E_shf')

    
## invert for basal friction over grounded ice :
#if i % 2 == 0:
#  params['newton_solver']['relaxation_parameter'] = 1.0
#  params['newton_solver']['relative_tolerance']   = 1e-8
#  params['newton_solver']['maximum_iterations']   = 3
#  config['adjoint']['objective_function']         = 'log_lin_hybrid'
#  config['adjoint']['gamma1']                     = 0.01
#  config['adjoint']['gamma2']                     = 1000
#  config['adjoint']['surface_integral']           = 'grounded'
#  config['adjoint']['control_domain']             = 'bed'
#  config['adjoint']['alpha']                      = 1e4
#  config['adjoint']['bounds']                     = (0.0, 4000)
#  config['adjoint']['control_variable']           = model.beta
#  model.init_viscosity_mode('linear')
#
## invert for enhancement over shelves :
#else:
#  #params['newton_solver']['relaxation_parameter'] = 1.0
#  #params['newton_solver']['relative_tolerance']   = 1e-8
#  #params['newton_solver']['maximum_iterations']   = 3
#  config['adjoint']['objective_function']         = 'log_lin_hybrid'
#  config['adjoint']['gamma1']                     = 0.001
#  config['adjoint']['gamma2']                     = 10000
#  config['adjoint']['surface_integral']           = 'shelves'
#  config['adjoint']['control_domain']             = 'complete'
#  config['adjoint']['alpha']                      = 1e-12
#  config['adjoint']['bounds']                     = (1e-6, 5.0)
#  config['adjoint']['control_variable']           = model.E_shf
#  #model.init_viscosity_mode('linear')



