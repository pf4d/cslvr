from varglas          import D3Model, MomentumDukowiczStokes, print_text, \
                             print_min_max
from varglas.energy   import Enthalpy 
from scipy            import random
from fenics           import *
from dolfin_adjoint   import *
import numpy              as np

def assimilate(h,H,g):
  """
  Run the full assimilation process for a cell size <h>, ice thickness <H>,
  and surface slope <g>.
  """
  out_dir = 'dump/stokes_E/h_%i/H_%i/g_%.2f/' % (h,H,g)
  n       = 25
  L       = n*h
  alpha   = g * pi / 180
  
  p1      = Point(0.0, 0.0, 0.0)
  p2      = Point(L,   L,   1)
  mesh    = BoxMesh(p1, p2, n, n, 10)
  
  model   = D3Model(mesh, out_dir = out_dir + 'initial/', use_periodic=True)
  
  surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                       element=model.Q.ufl_element())
  bed     = Expression('- x[0] * tan(alpha) - H', alpha=alpha, H=H, 
                       element=model.Q.ufl_element())
  beta    = Expression('H - H * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)', H=H,
                       alpha=alpha, L=L, element=model.Q.ufl_element())
  
  model.calculate_boundaries()
  model.deform_mesh_to_geometry(surface, bed)
  
  model.init_beta(beta)
  model.init_b(model.A0(0)**(-1/model.n(0)))
  model.init_E(1.0)
  
  nparams = {'newton_solver' : {'linear_solver'            : 'mumps',
                                'relative_tolerance'       : 1e-8,
                                'relaxation_parameter'     : 1.0,
                                'maximum_iterations'       : 28,
                                'error_on_nonconvergence'  : False}}
  m_params  = {'solver'      : nparams}
  
  mom = MomentumDukowiczStokes(model, m_params, isothermal=True)
  mom.solve(annotate=False)
  
  u,v,w = model.U3.split(True)
  u_s   = model.vert_extrude(u, d='down')
  v_s   = model.vert_extrude(v, d='down')
  sigma = 100.0
  U_mag = model.get_norm(as_vector([u_s, v_s]), 'linf')[1]
  n     = len(U_mag)
  U_avg = sum(U_mag) / n
  U_e   = U_avg / sigma
  print_min_max(U_e, 'U_error')
  
  u_o     = u.vector().array()
  v_o     = v.vector().array()
  n       = len(u_o)
  u_error = U_e * random.randn(n)
  v_error = U_e * random.randn(n)
  u_ob    = u_o + u_error
  v_ob    = v_o + v_error
  
  model.assign_variable(u, u_ob)
  model.assign_variable(v, v_ob)
  
  print_min_max(u_error, 'u_error')
  print_min_max(v_error, 'v_error')
  
  model.init_U_ob(u, v)
  # calculate and save the true enhancement for MSE calculation :
  # ( initialize viscosity in model.eta from momentum with unify_eta() )
  mom.unify_eta()
  tau_e  = mom.effective_stress(model.U3, model.eta)
  eps_e  = mom.effective_strain_rate(model.U3)
  A      = model.A0
  n      = model.n
  eta    = model.eta
  E      = A**(-1) * eps_e**(1/2.) * tau_e**(-n/2)
  #E      = (2*eta*A**(1/n)*eps_e**(-(1-n)/(2*n)))**(-n)
  E_true = project(E, model.Q)
  print_min_max(E_true, 'E_true')
  
  #model.init_beta(30**2)
  #model.init_beta(1.0)
  model.init_beta_SIA()
  model.save_pvd(model.beta, 'beta_SIA')
  
  model.save_pvd(model.U3,   'U_true')
  model.save_pvd(model.U_ob, 'U_ob')
  model.save_pvd(model.beta, 'beta_SIA')
  model.save_pvd(E_true,     'E_true')
  model.set_out_dir(model.out_dir + 'xml/')

  model.save_xml(interpolate(model.u, model.Q),  'u_true')
  model.save_xml(interpolate(model.v, model.Q),  'v_true')
  model.save_xml(interpolate(model.w, model.Q),  'w_true')
  model.save_xml(model.U_ob,                     'U_ob')
  model.save_xml(E_true,                         'E_true')
  model.save_xml(model.beta,                     'beta_SIA')
  
  # resolve momentum with uniform beta :
  #m_params['solver']['newton_solver']['relaxation_parameter'] = 0.7
  #model.assign_variable(mom.get_U(), DOLFIN_EPS)
  #mom.solve(annotate=False)
  
  # linearize the momentum equations : 
  m_params['solver']['newton_solver']['maximum_iterations']   = 3
  m_params['solver']['newton_solver']['relaxation_parameter'] = 1.0
  mom = MomentumDukowiczStokes(model, m_params, linear=True, isothermal=True)
  
  #alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 
  #          1e1,  1e2,  1e3,  1e4,  1e5,  1e6]
  #alphas = [1e-2, 5e-2, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 
  #          5e-1, 1e0,  5e0,  1e1,  5e1,  1e2]
  #alphas = [1e-2, 5e-2, 1e-1, 5e-1, 1e0,  5e0,  1e1]
  alphas = [1.0]
  Js     = []
  Rs     = []
  MSEs   = []
  REs    = []
  nits   = []
  
  for alpha in alphas:
  
    model.init_E_gnd(1.0)
    mom.solve(annotate=True)
  
    model.set_out_dir(out_dir = out_dir + 'assimilated/alpha_%.1E/' % alpha)
    
    J = mom.form_obj_ftn(integral=model.dSrf, kind='log_lin_hybrid', 
                         g1=0.01, g2=1000)
    R = mom.form_reg_ftn(model.E_gnd, integral=model.dx_g, kind='Tikhonov',
                         alpha=alpha)
    I = J + R
    
    controls = File(model.out_dir + "control_viz/E_control.pvd")
    E_viz    = Function(model.Q, name="E_control")
      
    def eval_cb(I, E):
      #mom.print_eval_ftns()
      #print_min_max(mom.U, 'U')
      print_min_max(I, 'I')
      print_min_max(E, 'E')
    
    def deriv_cb(I, dI, E):
      #print_min_max(I,  'I')
      print_min_max(dI, 'dI/dE')
      #print_min_max(E,  'E')
      E_viz.assign(E)
      controls << E_viz
    
    def hessian_cb(I, ddI, E):
      print_min_max(ddI, 'd/dE dI/dE')
    
    m = FunctionControl('E_gnd')
    F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                          derivative_cb_post = deriv_cb,
                          hessian_cb = hessian_cb)
    
    E_opt,res = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=(1e-8, 1e8),
                         options={"disp"    : True,
                                  "maxiter" : 1000,
                                  "gtol"    : 1e-5})
      
    #problem = MinimizationProblem(F, bounds=(1e-8, 1e0))
    #parameters = {"tol"                : 1e8,
    #              "acceptable_tol"     : 1000.0,
    #              "maximum_iterations" : 1000,
    #              "linear_solver"      : "ma57"}
    #
    #solver = IPOPTSolver(problem, parameters=parameters)
    #E_opt = solver.solve()
    print_min_max(E_opt, 'E_opt')
    
    model.init_E_gnd(E_opt)
    model.assign_variable(mom.get_U(), DOLFIN_EPS)
    mom.solve(annotate=False)
    mom.print_eval_ftns()

    E_true_v = E_true.vector()
    E_opt_v  = E_opt.vector()

    mse = norm(E_opt_v - E_true_v)**2 / len(E_true_v)
    re  = norm(E_opt_v - E_true_v) / norm(E_true_v)

    print_min_max(mse, 'MSE')
    print_min_max(re,  'RE')
      
    Rs.append(assemble(mom.Rp))
    Js.append(assemble(mom.J))
    MSEs.append(mse)
    REs.append(re)
    nits.append(res['nit'])
    
    model.save_pvd(model.E_gnd,  'E_opt')
    model.save_pvd(model.U3,     'U_opt')
    model.set_out_dir(model.out_dir + 'xml/')
    
    model.save_xml(model.E_gnd,                   'E_opt')
    model.save_xml(interpolate(model.u, model.Q), 'u_opt')
    model.save_xml(interpolate(model.v, model.Q), 'v_opt')
    model.save_xml(interpolate(model.w, model.Q), 'w_opt')
    
    # reset entire dolfin-adjoint state :
    adj_reset()
  
  from numpy import savetxt, array
  import os
  
  if model.MPI_rank==0:
    d = out_dir + 'plot/'
    if not os.path.exists(d):
      os.makedirs(d)
    savetxt(d + 'Rs.txt',   array(Rs))
    savetxt(d + 'Js.txt',   array(Js))
    savetxt(d + 'as.txt',   array(alphas))
    savetxt(d + 'MSEs.txt', array(MSEs))
    savetxt(d + 'REs.txt',  array(REs))
    savetxt(d + 'nits.txt', array(nits))


#===============================================================================
# perform assimilation over the following parameter space :

hs = [1000, 2000, 4000, 8000, 16000, 32000]
Hs = [250,  500,  750,  1000, 2000,  3000 ]
Gs = [0.1,  0.25, 0.5,  1,    2,     4    ]

for h in hs:
  for H in Hs:
    for g in Gs:
      pass
      #assimilate(h, H, g)

assimilate(1000, 1000, 0.25)


