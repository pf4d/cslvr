from varglas          import *
from varglas.energy   import Enthalpy 
from fenics           import *
from dolfin_adjoint   import *


# set the base directory to save :
i       = 0                                      # manually iterate
dir_b   = 'dump/low/0'

# set the relavent directories (complicated, right?!) :
var_dir = 'dump/vars_low/'
out_dir = dir_b + str(i) + '/inverted/'
tmc_dir = dir_b + str(i) + '/thermo_solve/hdf5/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5',     'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',         'r')
finput  = HDF5File(mpi_comm_world(), tmc_dir + 'thermo_solve.h5',  'r')
foutput = HDF5File(mpi_comm_world(), out_dir + 'hdf5/inverted.h5', 'w')

# get the bed and surface meshes :
bedmesh = Mesh()
srfmesh = Mesh()
fmeshes.read(bedmesh, 'bedmesh', False)
fmeshes.read(srfmesh, 'srfmesh', False)

# create boundary function spaces for saving variables :
Qb  = FunctionSpace(bedmesh, 'CG', 1)
Qs  = FunctionSpace(srfmesh, 'CG', 1)
Q3s = MixedFunctionSpace([Qs]*3)

# initialize the model :
model = D3Model(fdata, out_dir, state=foutput)
model.set_subdomains(fdata)

model.save_xdmf(model.ff, 'ff')

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

# use T0 and beta0 resulting from thermo_solve.py :
model.init_T(finput)          # temp
model.init_W(finput)          # water
model.init_beta(finput)       # friction
model.init_U(finput)          # velocity

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

# form the cost functional :
J = mom.form_obj_ftn(integral=model.dSrf_gu, kind='log_L2_hybrid', 
                     g1=0.01, g2=5000)
#J = mom.form_obj_ftn(integral=model.dSrf_gu, kind='ratio')

# form the regularization functional :
#R = mom.form_reg_ftn(model.beta, integral=model.dBed_g, kind='TV', 
#                     alpha=1.0)
R = mom.form_reg_ftn(model.beta, integral=model.dBed_f, kind='Tikhonov', 
                     alpha=1e-1)

# define the objective functional to minimize :
I = J + R

# create a pvd file for saving current optimized traction :
beta_b = Function(Qb,  name="beta_control")
beta_f = File(model.out_dir + "control_viz/beta_control.xdmf")

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
  #if i % 100 == 0:
  #  model.assign_submesh_variable(beta_b, beta)
  #  #model.save_xdmf(beta_b, 'beta_control', f_file=beta_f, t=i)
  #i += 1
  model.assign_submesh_variable(beta_b, beta)
  model.save_xdmf(beta_b, 'beta_control')

# hessian of objective function callback function (not used) :
def hessian_cb(I, ddI, beta):
  print_min_max(ddI, 'd/db dI/db')

# define the control parameter :
m = FunctionControl('beta')

# create the reduced functional to minimize :
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
model.save_xdmf(model.U3,   'U_opt')
model.save_xdmf(model.beta, 'beta_opt')

# save the state for thermo_solve.py to update temperature :
model.save_hdf5(model.U3)
model.save_hdf5(model.beta)


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



