from cslvr import *

# this problem has a few more constants than the last :
l       = 400000            # width of the domain
Hmax    = 3000              # thickness at the divide
S0      = 100               # terminus height above water 
B0      = -200              # terminus depth below water
nb      = 25                # number of basal bumps
b       = 50                # amplitude of basal bumps 
betaMax = 1000              # maximum traction coefficient
betaMin = 100               # minimum traction coefficient
Tmin    = 273.15 - 45       # temperature at divide
St      = 6.5 / 1000.0      # lapse rate

# create the rectangle mesh, to be deformed by the chosen geometry later :
p1      = Point(-l/2, 0.0)  # lower-left corner
p2      = Point( l/2, 1.0)  # upper-right corner
kx      = 150               # number of x-divisions
kz      = 10                # number of z-divisions
mesh    = RectangleMesh(p1, p2, kx, kz)

# these control the basal-boundary condition :
e_mode  = 'zero_energy'
#e_mode  = 'Fb'

# the output directories :
out_dir = 'ps_results_new/' + e_mode + '/'
plt_dir = '../../images/tmc/plane_strain_new/' + e_mode + '/'

# this is a lateral mesh problem, defined in the x,z plane.  Here we use
# linear-Lagrange elements corresponding with order=1 :
model = LatModel(mesh, out_dir=out_dir, order=1)

# the expressions for out data :
S = Expression('(Hmax+B0-S0)/2*cos(2*pi*x[0]/l) + (Hmax+B0+S0)/2',
               Hmax=Hmax, B0=B0, S0=S0, l=l,
               element = model.Q.ufl_element())
B = Expression('b*cos(nb*2*pi*x[0]/l) + B0',
               b=b, l=l, B0=B0, nb=nb,
               element = model.Q.ufl_element())
b = Expression('(bMax - bMin)/2.0*cos(2*pi*x[0]/l) + (bMax + bMin)/2.0',
               bMax=betaMax, bMin=betaMin, l=l,
               element = model.Q.ufl_element())
T = Expression('Tmin + St*(Hmax + B0 - S0 - x[1])',
               Tmin=Tmin, Hmax=Hmax, B0=B0, S0=S0, St=St,
               element = model.Q.ufl_element())

# deform the geometry to match the surface and bed functions :
model.deform_mesh_to_geometry(S, B)

# mark the facets and cells for proper integration :
model.calculate_boundaries(mask=None)

# initialize the variables :
model.init_beta(b)                             # traction
model.init_T(T)                                # internal temperature
model.init_T_surface(T)                        # atmospheric temperature
model.init_Wc(0.03)                            # max basal water content
model.init_k_0(5e-3)                           # non-advective flux coef.
model.init_q_geo(model.ghf)                    # geothermal flux
model.solve_hydrostatic_pressure()             # for pressure-melting 
model.form_energy_dependent_rate_factor()      # thermo-mech coupling      

# the momentum and energy physics :
mom = MomentumDukowiczPlaneStrain(model)

# the energy physics using the chosen basal energy flux mode and 
# Galerkin/least-squares stabilization :
nrg = Enthalpy(model, mom, energy_flux_mode = e_mode,
               stabilization_method = 'SUPG')

# thermo-solve callback function, at the end of each TMC iteration :
def tmc_cb_ftn():
  nrg.calc_PE()                      # calculate grid Peclet number
  nrg.calc_vert_avg_strain_heat()    # calc vert. avg. of Q
  nrg.calc_vert_avg_W()              # calv vert. avg. of W
  nrg.calc_temp_rat()                # calc ratio of H that is temperate

# at the end of the TMC procedure, save the state of these functions :
tmc_save_vars = [model.T,
                 model.W,
                 model.Fb,
                 model.Mb,
                 model.alpha,
                 model.alpha_int,
                 model.PE,
                 model.Wbar,
                 model.Qbar,
                 model.temp_rat,
                 model.U3,
                 model.p,
                 model.beta,
                 model.theta]

# form the objective functional for water-flux optimization :
nrg.form_cost_ftn(kind='L2')

# the water-content optimization problem args :
wop_kwargs = {'max_iter'            : 25,
              'bounds'              : (DOLFIN_EPS, 100.0),
              'method'              : 'ipopt',
              'adj_callback'        : None}
 
# thermo-mechanical coupling args :
tmc_kwargs = {'momentum'            : mom,
              'energy'              : nrg,
              'wop_kwargs'          : wop_kwargs,
              'callback'            : tmc_cb_ftn,
              'atol'                : 1e2,
              'rtol'                : 1e0,
              'max_iter'            : 20,
              'iter_save_vars'      : None,
              'post_tmc_save_vars'  : tmc_save_vars,
              'starting_i'          : 1}
                                    
# thermo_solve :
model.thermo_solve(**tmc_kwargs)

# save the mesh for plotting the data saved by tmc_save_vars :
f = HDF5File(mpi_comm_world(), out_dir + 'state.h5', 'w')
model.save_subdomain_data(f)
model.save_mesh(f)
f.close()

#===============================================================================
# plotting :

figsize = (10,2.2)

model.init_U_mag(model.U3)
U_min  = model.U_mag.vector().min()
U_max  = model.U_mag.vector().max()
U_lvls = array([U_min, 1e2, 1e3, 1e4, 1.5e4, U_max])
plot_variable(u = model.U_mag, name = 'U_mag', direc = plt_dir,
              figsize             = figsize,
              title               = r'$\Vert \mathbf{u} \Vert$',
              cmap                = 'viridis',
              levels              = None,#U_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%.1e')

p_min  = model.p.vector().min()
p_max  = model.p.vector().max()
p_lvls = array([p_min, 1e6, 5e6, 1e7, 1.5e7, 2e7, 2.5e7, p_max])
plot_variable(u = model.p, name = 'p', direc = plt_dir,
              figsize             = figsize,
              title               = r'$p$',
              cmap                = 'viridis',
              levels              = None,#p_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%.1e')

beta_lvls = array([0, 200, 400, 600, 800, 1000])
plot_variable(u = model.beta, name = 'beta', direc = plt_dir,
              figsize             = (6,2),
              title               = r'$\beta$',
              cmap                = 'viridis',
              levels              = None,#beta_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              cb_format           = '%g')

T_min  = model.T.vector().min()
T_max  = model.T.vector().max()
T_lvls = array([T_min, 230, 240, 250, 260, 265, 270, T_max])
plot_variable(u = model.T, name = 'T', direc = plt_dir,
              figsize             = figsize,
              title               = r'$T$',
              cmap                = 'viridis',
              levels              = None,#T_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%.1f')

W_min  = model.W.vector().min()
W_max  = model.W.vector().max()
W_lvls = array([0.0, 1e-2, 5e-2, W_max])
plot_variable(u = model.W, name = 'W', direc = plt_dir,
              figsize             = figsize,
              title               = r'$W$',
              cmap                = 'viridis',
              levels              = None,#W_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%.1e')

plot_variable(u = model.theta, name = 'theta', direc = plt_dir,
              figsize             = figsize,
              title               = r'$\theta$',
              cmap                = 'viridis',
              levels              = None,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%g')

Mb_min  = model.Mb.vector().min()
Mb_max  = model.Mb.vector().max()
Mb_lvls = array([0.0, 0.2, 0.3, 0.4, 0.5, Mb_max])
plot_variable(u = model.Mb, name = 'Mb', direc = plt_dir,
              figsize             = figsize,
              title               = r'$M_b$',
              cmap                = 'viridis',#'gist_yarg',
              levels              = None,#Mb_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%.1f')

Fb_min  = model.Fb.vector().min()
Fb_max  = model.Fb.vector().max()
Fb_lvls = array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, Fb_max])
plot_variable(u = model.Fb, name = 'Fb', direc = plt_dir,
              figsize             = figsize,
              title               = r'$F_b$',
              cmap                = 'viridis',
              levels              = None,#Fb_lvls,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              extend              = 'both',
              cb_format           = '%.1f')



