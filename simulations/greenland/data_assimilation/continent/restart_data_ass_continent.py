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
dir_b   = 'dump/high_da_ipopt_SIA0_SR/0'     # directory to save

# set the output directory :
out_dir = dir_b + str(i)
var_dir = 'dump/vars_high/'
in_dir  = 'dump/high_da_bfgs_SIA0_SR/00/thermo_solve/xml/'

f = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')

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
model.init_u_lat(0.0)
model.init_v_lat(0.0)

model.init_T(in_dir + 'T.xml')          # temp
model.init_W(in_dir + 'W.xml')          # water
model.init_beta_SIA()
model.init_U(in_dir + 'u.xml',
             in_dir + 'v.xml',
             in_dir + 'w.xml')


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

# invert for basal friction over grounded ice :
nparams['newton_solver']['relaxation_parameter'] = 1.0
nparams['newton_solver']['relative_tolerance']   = 1e-8
nparams['newton_solver']['maximum_iterations']   = 3
m_params['solve_vert_velocity']                  = False

mom = MomentumDukowiczStokesReduced(model, m_params, isothermal=False, 
                                    use_lat_bcs=True, linear=True)
#mom = MomentumBP(model, m_params, isothermal=False, linear=True)
mom.solve(annotate=True)

out_dir = out_dir + '/inverted_g2_10000/'

model.set_out_dir(out_dir = out_dir + 'pvd/')
  
J = mom.form_obj_ftn(integral=model.dSrf_g, kind='log_L2_hybrid', 
                     g1=0.01, g2=10000)
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
  model.save_pvd(beta_viz, 'beta_control', f_file=controls) 

def hessian_cb(I, ddI, beta):
  print_min_max(ddI, 'd/db dI/db')

m = FunctionControl('beta')
F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                      derivative_cb_post = deriv_cb,
                      hessian_cb = hessian_cb)

#b_opt = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=(1e-6, 1e7),
#                 options={"disp"    : True,
#                          "maxiter" : 1000,
#                          "gtol"    : 1e-5})

problem = MinimizationProblem(F, bounds=(1e-6, 1e7))
parameters = {"tol"                : 1e8,
              "acceptable_tol"     : 1000.0,
              "maximum_iterations" : 1000,
              "ma97_order"         : "metis",
              "linear_solver"      : "ma97"}

solver = IPOPTSolver(problem, parameters=parameters)
b_opt  = solver.solve()
print_min_max(b_opt, 'b_opt')

model.init_beta(b_opt)

m_params['solve_vert_velocity'] = True
mom.solve(annotate=False)

model.save_pvd(model.U3, 'U_opt')

model.set_out_dir(out_dir = out_dir + 'xml/')

u,v,w = model.U3.split(True)

model.save_xml(model.T,                       'T')
model.save_xml(model.W,                       'W')
model.save_xml(interpolate(model.u, model.Q), 'u')
model.save_xml(interpolate(model.v, model.Q), 'v')
model.save_xml(interpolate(model.w, model.Q), 'w')
model.save_xml(model.beta,                    'beta')
model.save_xml(model.Mb,                      'Mb')
model.save_xml(model.E_shf,                   'E_shf')



