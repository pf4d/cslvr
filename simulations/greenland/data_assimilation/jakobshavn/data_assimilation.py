from varglas          import *
from varglas.energy   import Enthalpy 
from fenics           import *
from dolfin_adjoint   import *

#set_log_active(False)
#set_log_level(PROGRESS)

# set the base directory to save :
i       = 0                                      # manually iterate
dir_b   = 'dump/jakob_da_ipopt_SIA0_SR/0'

# set the output directory :
in_dir  = dir_b + str(i) + '/thermo_solve/xml/'  # previous thermo_solve.py run
out_dir = dir_b + str(i) + '/inverted_logL2_TV/' # new output directory
var_dir = 'dump/vars_jakobshavn/'                # gen_vars.py ouput state.h5

# get the data created with gen_vars.py :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')

# get the bed mesh :
bedmesh = Mesh()
fmeshes.read(bedmesh, 'bedmesh', False)

# create bed function space for saving beta opt :
Qb = FunctionSpace(bedmesh, 'CG', 1)

# initialize the model :
model = D3Model(f, out_dir)
model.set_subdomains(f)

# initialize variables :
model.init_S(fdata)
model.init_B(fdata)
model.init_mask(fdata)
model.init_q_geo(model.ghf)
model.init_T_surface(fdata)
model.init_adot(fdata)
model.init_U_ob(fdata, fdata)
model.init_U_mask(fdata)
model.init_E(1.0)

# use T0 and beta0 from the previous thermo_solve :
model.init_T(in_dir + 'T.xml')          # temp
model.init_W(in_dir + 'W.xml')          # water
model.init_beta(in_dir + 'beta.xml')    # friction
#model.init_beta(3000)
model.init_U(in_dir + 'u.xml',
             in_dir + 'v.xml',
             in_dir + 'w.xml')

# Newton solver parameters for momentum :
nparams = {'newton_solver' : {'linear_solver'            : 'cg',
                              'preconditioner'           : 'hypre_amg',
                              'relative_tolerance'       : 1e-8,
                              'relaxation_parameter'     : 1.0,
                              'maximum_iterations'       : 3,
                              'error_on_nonconvergence'  : False}}

# momentum basic parameters :
m_params  = {'solver'               : nparams,
             'solve_vert_velocity'  : False,
             'solve_pressure'       : False,
             'vert_solve_method'    : 'mumps'}

# create a Momentum instance with linearized viscosity for incompelete-adjoint :
mom = MomentumDukowiczStokesReduced(model, m_params, isothermal=False, 
                                    linear=True)
#mom = MomentumDukowiczStokes(model, m_params, isothermal=False, linear=True)
#mom = MomentumBP(model, m_params, isothermal=False, linear=True)

# solve the momentum, with annotation for dolfin-adjoint :
mom.solve(annotate=True)

model.save_pvd(model.U3,   'U_ini')

# form the cost functional :
J = mom.form_obj_ftn(integral=model.dSrf_g, kind='log_L2_hybrid', 
                     g1=0.01, g2=10000)
#J = mom.form_obj_ftn(integral=model.dSrf_g, kind='ratio') 

# form the regularization functional :
R = mom.form_reg_ftn(model.beta, integral=model.dBed_g, kind='TV', 
                     alpha=1.0)

# define the objective functional to minimize :
I = J + R

# create a pvd file for saving current optimized traction :
beta_b = File(model.out_dir + "control_viz/beta_control.pvd")
beta_viz = Function(Qb, name="beta_control")

# create counter to save .xml and .pvd files ;
i = 0

# objective function callback function : 
def eval_cb(I, beta):
  #mom.print_eval_ftns()
  #print_min_max(mom.U, 'U')
  print_min_max(I,    'I')
  print_min_max(beta, 'beta')

# derivative of objective function callback function : 
def deriv_cb(I, dI, beta):
  global i
  #print_min_max(I,     'I')
  print_min_max(dI,    'dI/dbeta')
  #print_min_max(beta,  'beta')
  if i % 100 == 0:
    model.assign_submesh_variable(beta_viz, beta)
    model.set_out_dir(out_dir = out_dir + 'xml/')
    model.save_xml(beta_viz, 'beta_control_%s' % str(i))
    model.set_out_dir(out_dir = out_dir + 'pvd/')
    model.save_pvd(beta_viz, 'beta_control', f_file=beta_b) 
  i += 1

# hessian of objective function callback function (not used) :
def hessian_cb(I, ddI, beta):
  print_min_max(ddI, 'd/db dI/db')

# define the control parameter :
m = FunctionControl('beta')

# create the reduced functional for to minimize :
F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                      derivative_cb_post = deriv_cb,
                      hessian_cb = hessian_cb)

# optimize with scipy's fmin_l_bfgs_b :
#b_opt = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=(1e-6, 1e7),
#                 options={"disp"    : True,
#                          "maxiter" : 1000,
#                          "gtol"    : 1e-5})

# or optimize with IPOpt (preferred) :
problem = MinimizationProblem(F, bounds=(1e-6, 1e7))
parameters = {"tol"                : 1e-8,
              "acceptable_tol"     : 1e-6,
              "maximum_iterations" : 1000,
              "ma97_order"         : "metis",
              "linear_solver"      : "ma97"}
solver = IPOPTSolver(problem, parameters=parameters)
b_opt  = solver.solve()
print_min_max(b_opt, 'b_opt')

# initalize the model with the optimal traction :
model.init_beta(b_opt)

# re-solve the momentum equations with vertical velocity and optimal beta :
m_params['solve_vert_velocity'] = True
mom.solve(annotate=False)

# save the optimal velocity and beta fields :
model.save_pvd(model.U3,   'U_opt')
model.save_pvd(model.beta, 'beta_opt')

# save xml files for thermo_solve.py to update temperature :
u,v,w = model.U3.split(True)

model.save_xml(interpolate(model.u, model.Q), 'u')
model.save_xml(interpolate(model.v, model.Q), 'v')
model.save_xml(interpolate(model.w, model.Q), 'w')
model.save_xml(model.beta,                    'beta')



