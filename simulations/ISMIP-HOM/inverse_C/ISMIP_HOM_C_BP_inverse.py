from varglas.model   import Model
from varglas.solvers import SteadySolver, AdjointSolver
from varglas.helper  import default_nonlin_solver_params, default_config
from varglas.io      import print_text, print_min_max
from fenics          import *
from scipy           import random

set_log_active(False)

alpha = 1.0 * pi / 180
L     = 20000

nparams = default_nonlin_solver_params()
#nparams['newton_solver']['linear_solver']           = 'mumps'
nparams['newton_solver']['linear_solver']           = 'cg'
nparams['newton_solver']['preconditioner']          = 'hypre_amg'
nparams['newton_solver']['relative_tolerance']      = 1e-8
nparams['newton_solver']['relaxation_parameter']    = 0.7
nparams['newton_solver']['maximum_iterations']      = 25
nparams['newton_solver']['error_on_nonconvergence'] = False
parameters['form_compiler']['quadrature_degree']    = 2

config = default_config()
config['log_history']                     = False
config['output_path']                     = './results/initial/'
config['model_order']                     = 'BP'
config['use_dukowicz']                    = False
config['periodic_boundary_conditions']    = True
config['use_pressure_boundary']           = True
config['velocity']['newton_params']       = nparams
config['velocity']['vert_solve_method']   = 'mumps'
config['coupled']['on']                   = True
config['coupled']['max_iter']             = 4
config['enthalpy']['on']                  = True

#BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
mesh  = BoxMesh(0, 0, 0, L, L, 1, 20, 20, 10)

model = Model(config)
model.set_mesh(mesh)

surface = Expression('- x[0] * tan(alpha) + 500', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - 1100.0', alpha=alpha, 
                     element=model.Q.ufl_element())
beta    = Expression('sqrt(1000 - 800 * sin(pi*x[0]/L) * sin(pi*x[1]/L))',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.calculate_boundaries()
model.set_geometry(surface, bed, deform=True)
model.initialize_variables()

#model.init_viscosity_mode('isothermal')
model.init_viscosity_mode('full')
model.init_beta(beta)
model.init_mask(0.0)
model.init_T_surface(model.T_w - 20.0)
model.init_T(model.T_w - 20.0)
model.init_q_geo(model.ghf)
model.init_E(1.0)

model.save_pvd(model.beta, 'beta_true')

F = SteadySolver(model, config)
F.solve()


model.init_viscosity_mode('linear')

u_o = model.u.vector().array()
v_o = model.v.vector().array()
#U_e = model.get_norm(as_vector([model.u, model.v]), 'linf')[1] / 500
U_e = 2.0
print_min_max(U_e, 'U_e')
n   = len(u_o)

config['velocity']['solve_vert_velocity'] = False
config['adjoint']['objective_function']   = 'log_lin_hybrid'
config['adjoint']['gamma1']               = 0.01
config['adjoint']['gamma2']               = 1000
config['adjoint']['bounds']               = (10, 100.0)
config['adjoint']['control_variable']     = model.beta
config['adjoint']['alpha']                = 0.0
config['adjoint']['max_fun']              = 100
config['enthalpy']['on']                  = False
config['coupled']['on']                   = False
config['velocity']['solve_vert_velocity'] = False

nparams['newton_solver']['relaxation_parameter']    = 1.0
nparams['newton_solver']['maximum_iterations']      = 10

alpha_v = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 
           1e-6,  1e-5,  1e-4,  1e-3,  1e-2,  1e-1,  1.0,  10.0, 100]
D_v     = [] # array of discrepencies
R_v     = [] # array of regularization cost function integrals
J_v     = [] # array of misfit cost function integrals
for a in alpha_v:
  u_error = U_e * random.randn(n)
  v_error = U_e * random.randn(n)
  u_ob    = u_o + u_error
  v_ob    = v_o + v_error
  
  config['output_path']      = './results/alpha_%.1E/' % a
  config['adjoint']['alpha'] = a
  
  model.init_beta(30.0)
  model.init_U_ob(u_ob, v_ob)
  model.save_pvd(model.U_ob, 'U_ob')
  
  A = AdjointSolver(model, config)
  A.solve()
  
  model.calc_misfit('grounded')
  R = assemble(A.adjoint_instance.R)
  J = assemble(A.adjoint_instance.J)
  
  D_v.append(model.misfit)
  R_v.append(R)
  J_v.append(J)


if model.MPI_rank == 0:
  from pylab import *
  
  savetxt('D.txt',     D_v)
  savetxt('R.txt',     R_v)
  savetxt('J.txt',     J_v)
  savetxt('alpha.txt', alpha_v)

  fig = figure()
  ax  = fig.add_subplot(111)
  ax.plot(J_v, R_v, 'ko-', lw=2.0)
  ax.grid()
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel(r'$\mathcal{J}(\alpha)$')
  ax.set_ylabel(r'$\mathcal{R}(\alpha)$')
  tight_layout()
  show()
  
  fig = figure()
  ax  = fig.add_subplot(111)
  ax.plot(alpha_v, D_v, 'ko-', lw=2.0)
  ax.grid()
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel(r'$\alpha$')
  ax.set_ylabel(r'$\mathcal{D}(\alpha)$')
  tight_layout()
  show()



