from cslvr                    import *
from fenics_viz               import plot_variable
from sympy.utilities.lambdify import lambdify, implemented_function
import sympy                      as sp
import numpy                      as np

parameters['form_compiler']['quadrature_degree'] = 2

# this should be obvious what this does :
plot_analytic_solution = False

# output directiories :
mdl_odr = 'FS'
out_dir = './results/' + mdl_odr + '/'
plt_dir = './images/'  + mdl_odr + '/'

order  = 2
linear = False
if order > 1:  stab = False
else:          stab = True

# create a genreic box mesh, we'll fit it to geometry below :
#L     = 40000.0                       # width of domain
L     = 1.0                           # width of domain
p1    = Point(0.0, 0.0, 0.0)          # origin
p2    = Point(L,   L,   1)            # x, y, z corner
nx    = 25
nz    = 25
mesh  = BoxMesh(p1, p2, nx, nx, nz)   # a box to fill the void

# we have a three-dimensional problem here, with periodic lateral boundaries :
model   = D3Model(mesh, out_dir = out_dir, use_periodic = False, order=order)

# define constants :
a       = np.float128(0.5 * pi / 180) # surface slope in radians
#L       = np.float128(L)              # width of domain
#b_bar   = np.float128(1000.0)         # average ice thickness
#amp     = np.float128(500.0)          # surface modulation amplitude
#lam     = 2                           # deformation parameter in x direction
#u_mag   = np.float128(1.0)            # avg x component of vel. at the surf.
#u_x_amp = np.float128(0.5)            # x-velocity modulation amplitude

L       = np.float128(L)              # width of domain
b_bar   = np.float128(1.0)            # average ice thickness
amp     = np.float128(1e-1)           # surface modulation amplitude
#lam     = 2                           # deformation parameter in x direction
u_mag   = np.float128(1.0)            # avg x component of vel. at the surf.
u_x_amp = np.float128(0.5)            # x-velocity modulation amplitude

lam     = 1                           # deformation parameter in x direction

# upper surface :
def s(x,y):
	#return amp * sp.sin(3*pi*x/L)
	#return -x*sp.tan(a)
	return sp.Rational(0.0)

# lower surface
def b(x,y):
	#return s(x,y) - b_bar + amp * sp.sin(2*pi*x/L)# * sp.sin(2*pi*y/L)
	return s(x,y) - b_bar

# rate of change of upper surface :
def dsdt(x,y):
	#return 3 * ((y - L/2)**2 - (x - L/2)**2)
	return sp.Rational(0.0)

# rate of change of lower surface :
def dbdt(x,y):
	#return sp.sin(pi*x/L) * sp.sin(pi*y/L)
	return sp.Rational(0.0)

# upper-surface-mass balance :
def s_ring(x,y):
	#return sp.Rational(1.0)
	#return sp.sin(4*pi*x/L) * sp.sin(4*pi*y/L)
	return sp.Rational(-1.0)

# lower-surface-mass balance :
def b_ring(x,y):
	#return sp.sin(2*pi*x/L) * sp.sin(2*pi*y/L)
	return sp.Rational(-1.0)

# x-component of velocity at the upper surface :
def u_xs(x,y):
	return u_mag - u_x_amp * sp.sin(2*pi*x/L) * sp.sin(2*pi*y/L)
	#return sp.Rational(1.0)

# x-component of velocity at the lower surface :
def u_xb(x,y):
	#return 0.5 * u_xs(x,y)
	#return 0.5*(u_mag - u_x_amp * sp.sin(3*pi*x/L) * sp.sin(3*pi*y/L))
	return sp.Rational(0.0)
	#return u_mag - u_x_amp * sp.sin(2*pi*x/L) * sp.sin(2*pi*y/L)

# friction coefficent on lower surface :
def beta(x,y):
	return sp.Rational(1e0)

# temperature :
def T(x,y,z):
	return sp.Rational(268.0)

ver = Verification(model)
ver.init_expressions(s, b, u_xs, u_xb, dsdt, dbdt, s_ring, b_ring, lam)
ver.init_beta(beta)
ver.init_T(T)
ver.verify_analytic_solution(nx=1000, ny=1000, Lx=L, Ly=L)

#===============================================================================
# solve FEM :

# deform the mesh to match our desired geometry :
model.deform_mesh_to_geometry(ver.get_S(), ver.get_B())

model.calc_normal_vector()

# save the facet markers :
model.save_xdmf(model.ff, 'ff')

# the model must be deformed first for the momentum to interpolate correctly :
u_ob = interpolate(ver.get_u(), model.V)
u_ob_x, u_ob_y, u_ob_z = u_ob.split(True)

# initialize all the pertinent variables :
model.init_beta(ver.get_beta())             # friction coefficent on l.s.
model.init_T(ver.get_T())
model.init_Tp(model.T)
model.init_u_x_ob(u_ob_x)
model.init_u_y_ob(u_ob_y)
model.init_u_z_ob(u_ob_z)
model.init_u(u_ob)
model.init_S_ring(ver.get_S_ring())
model.init_B_ring(ver.get_B_ring())
model.init_dSdt(ver.get_dSdt())
model.init_dBdt(ver.get_dBdt())
model.form_energy_dependent_rate_factor()

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
	mom = MomentumDukowiczBP(model, use_pressure_bc=False)
elif mdl_odr == 'RS':
	mom = MomentumDukowiczStokesReduced(model, use_pressure_bc=False)
elif mdl_odr == 'FS':
	#mom = MomentumNitscheStokes(model, use_pressure_bc=False,
	#                            stabilized=stab,
	#                            linear=linear)
	mom = MomentumDukowiczStokes(model, use_pressure_bc=False,
	                             stabilized=stab,
	                             linear=linear)
	#mom = MomentumStokes(model, use_pressure_bc=False, stabilized=False,
	#                     stabilized=stab,
	#                     linear=linear)

# save the facet markers :
#model.save_xdmf(mom.ff, 'mom.ff')

# add the appropriate compensatary forcing terms to the momentum balance :
mom.add_compensatory_forcing_terms(ver)

# TODO: this has to be called after ``add_compensatory_forcing_terms()``:
#       find a better way to create the velocity before momentum.
p_ob = interpolate(ver.get_p(), model.Q)
model.init_p(p_ob)

# solve the momentum :
mom.solve()

# let's investigate the velocity divergence :
if mdl_odr == 'BP':
	u_x,u_y  = mom.get_unknown()
	u_z      = mom.w
elif mdl_odr == 'FS':
	u_x, u_y, u_z, p  = mom.get_unknown()
drhodt   = project(model.rho_i*div(as_vector([u_x,u_y,u_z])), model.Q,
                   solver_type='iterative')
div_u    = project(div(model.u), model.Q, solver_type='iterative')
div_u_ob = interpolate(ver.get_div_u(), model.Q)


# the purpose for everything below this line is data visualization :
#===============================================================================

# save these files with a name that makes sense for use with paraview :
model.save_xdmf(model.p,  'p')
model.save_xdmf(model.u,  'u')
model.save_xdmf(drhodt,   'drhodt')
model.save_xdmf(div_u,    'div_u')
model.save_xdmf(u_ob,     'u_ob')
model.save_xdmf(div_u_ob, 'div_u_ob')
model.save_xdmf(p_ob,     'p_ob')

plt_kwargs = {'direc'               : plt_dir,
              'cells'               : None,
              'figsize'             : (3,3),
              'cmap'                : 'viridis',#'RdGy',
              'scale'               : 'lin',
              'numLvls'             : 10,
              'levels_2'            : None,
              'umin'                : None,
              'umax'                : None,
              'normalize_vec'       : False,
              'plot_tp'             : False,
              'tp_kwargs'           : {'linestyle'      : '-',
                                       'lw'             : 1.0,
                                       'color'          : 'k',
                                       'alpha'          : 0.5},
              'show'                : False,
              'hide_x_tick_labels'  : True,
              'hide_y_tick_labels'  : True,
              'vertical_y_labels'   : False,
              'vertical_y_label'    : True,
              'xlabel'              : '',#r'$x$',
              'ylabel'              : '',#r'$y$',
              'equal_axes'          : True,
              'title'               : None,
              'hide_axis'           : False,
              'colorbar_loc'        : 'right',
              'contour_type'        : 'lines',
              'extend'              : 'neither',
              'ext'                 : '.pdf',
              'plot_quiver'         : False,
              'quiver_kwargs'       : {'pivot'          : 'middle',
                                       'color'          : 'k',
                                       'alpha'          : 0.8,
                                       'width'          : 0.004,
                                       'headwidth'      : 4.0,
                                       'headlength'     : 4.0,
                                       'headaxislength' : 4.0},
              'res'                 : 150,
              'cb'                  : False,
              'cb_format'           : '%.1f'}


# remove the coordinates :
plt_kwargs['coords'] = None

# create the bed and surface meshes :
model.form_bed_mesh()
model.form_srf_mesh()

# create 2D models :
bedmodel = D2Model(model.bedmesh, out_dir, order=order)
srfmodel = D2Model(model.srfmesh, out_dir, order=order)

# we don't have a function for this included in the `model' instance,
# so we have to make one ourselves :
drhodt_b = Function(bedmodel.Q, name='drhodt')

# function allows Lagrange interpolation between different meshes :
bedmodel.assign_submesh_variable(drhodt_b, drhodt)
srfmodel.assign_submesh_variable(srfmodel.u, model.u)
srfmodel.init_u_mag(srfmodel.u)  # calculates the velocity magnitude
bedmodel.assign_submesh_variable(bedmodel.u, model.u)
bedmodel.init_u_mag(bedmodel.u)  # calculates the velocity magnitude
bedmodel.assign_submesh_variable(bedmodel.p,  model.p)

## figure out some nice-looking contour levels :
#U_min  = srfmodel.u_mag.vector().min()
#U_max  = srfmodel.u_mag.vector().max()
##U_lvls = array([84, 86, 88, 90, 92, 94, 96, 98, 100])  # momentum comparison
#U_lvls = array([U_min, 87, 88, 89, 90, 91, 92, U_max])
#
#p_min  = bedmodel.p.vector().min()
#p_max  = bedmodel.p.vector().max()
#p_lvls = array([4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 1.1e7, 1.2e7, p_max])
#
#d_min  = drhodt_b.vector().min()
#d_max  = drhodt_b.vector().max()
#d_lvls = array([d_min, -5e-3, -2.5e-3, -1e-3,
#                1e-3, 2.5e-3, 5e-3, d_max])

u_s = srfmodel.u
u_x_s, u_y_s, u_z_s = srfmodel.u.split(True)
u_b = srfmodel.u
u_x_b, u_y_b, u_z_b = bedmodel.u.split(True)

# this time, let's plot the topography like a topographic map :
plt_kwargs['name']      = 'u_model_mag_s'
plot_variable(u=u_s,   **plt_kwargs)

plt_kwargs['name']      = 'u_model_x_s'
plot_variable(u=u_x_s, **plt_kwargs)

plt_kwargs['name']      = 'u_model_x_b'
plot_variable(u=u_x_b, **plt_kwargs)

plt_kwargs['name']      = 'u_model_y_s'
plot_variable(u=u_y_s, **plt_kwargs)

plt_kwargs['name']      = 'u_model_y_b'
plot_variable(u=u_y_b, **plt_kwargs)

plt_kwargs['name']      = 'u_model_z_s'
plot_variable(u=u_z_s, **plt_kwargs)

plt_kwargs['name']      = 'u_model_z_b'
plot_variable(u=u_z_b, **plt_kwargs)

plt_kwargs['name']      = 'p_model_b'
plot_variable(u=bedmodel.p,   **plt_kwargs)

#plt_kwargs['name']      = 'drho_dt'
#plot_variable(u=drhodt_b, **plt_kwargs)

if plot_analytic_solution:

	# calculate vertically-integrated variables :
	ver.init_r2_stress_balance()

	# get the coordinates :
	x, y, z = ver.x, ver.y, ver.z

	#=============================================================================
	# create functions which evaluate the sympy expressions with numpy arrays :
	S           = lambdify((x,y),   ver.s,              "numpy")
	B           = lambdify((x,y),   ver.b,              "numpy")
	H           = lambdify((x,y),   ver.h,              "numpy")
	dSdt        = lambdify((x,y),   ver.dsdt,           "numpy")
	dBdt        = lambdify((x,y),   ver.dbdt,           "numpy")
	dHdt        = lambdify((x,y),   ver.dhdt,           "numpy")
	n_mag_S     = lambdify((x,y),   ver.n_mag_s,        "numpy")
	n_mag_B     = lambdify((x,y),   ver.n_mag_b,        "numpy")
	U_xs        = lambdify((x,y),   ver.u_xs,           "numpy")
	U_xb        = lambdify((x,y),   ver.u_xb,           "numpy")
	U_ys        = lambdify((x,y),   ver.u_ys,           "numpy")
	U_yb        = lambdify((x,y),   ver.u_yb,           "numpy")
	U_zs        = lambdify((x,y),   ver.u_zs,           "numpy")
	U_zb        = lambdify((x,y),   ver.u_zb,           "numpy")
	ring_B      = lambdify((x,y),   ver.b_ring,         "numpy")
	ring_S      = lambdify((x,y),   ver.s_ring,         "numpy")
	div_Hu      = lambdify((x,y),   ver.div_hu,         "numpy")
	ring_i      = lambdify((x,y),   ver.b_ring_inverse, "numpy")
	#lei_resid   = lambdify((x,y),   ver.leibniz_resid,  "numpy")
	u_mag       = lambdify((x,y,z), ver.u_mag,          "numpy")
	p           = lambdify((x,y,z), ver.p,              "numpy")
	div_sigma_x = lambdify((x,y,z), ver.div_sigma[0],   "numpy")
	div_sigma_y = lambdify((x,y,z), ver.div_sigma[1],   "numpy")
	div_sigma_z = lambdify((x,y,z), ver.div_sigma[2],   "numpy")

	# convert sympy to numpy :
	nx    = 1000
	ny    = 1000

	# create a genreic box mesh, we'll fit it to geometry below :
	x_a   = np.linspace(0, L, nx)
	y_a   = np.linspace(0, L, ny)
	X,Y   = np.meshgrid(x_a, y_a)

	#=============================================================================
	# plot :

	# set the coordinates :
	plt_kwargs['coords'] = (X,Y)

	# this time, let's plot the topography like a topographic map :
	plt_kwargs['name']      = 'u_mag_s'
	plot_variable(u=u_mag(X,Y,S(X,Y)), **plt_kwargs)

	plt_kwargs['name']      = 'u_ys'
	plot_variable(u=U_ys(X,Y),       **plt_kwargs)

	plt_kwargs['name']      = 'u_yb'
	plot_variable(u=U_yb(X,Y),       **plt_kwargs)

	plt_kwargs['name']      = 'u_xs'
	plot_variable(u=U_xs(X,Y),       **plt_kwargs)

	#plt_kwargs['name']      = 'u_xb'
	#plot_variable(u=U_xb(X,Y),       **plt_kwargs)

	plt_kwargs['name']      = 'p_b'
	plot_variable(u=p(X,Y,B(X,Y)),   **plt_kwargs)

	#plt_kwargs['name']      = 'u_zs'
	#plot_variable(u=U_zs(X,Y),       **plt_kwargs)

	#plt_kwargs['name']      = 'u_zb'
	#plot_variable(u=U_zb(X,Y),       **plt_kwargs)

	#plt_kwargs['name']      = 'ring_S'
	#plot_variable(u=ring_S(X,Y),     **plt_kwargs)

	#plt_kwargs['name']      = 'ring_B'
	#plot_variable(u=ring_B(X,Y),     **plt_kwargs)

	#plt_kwargs['name']      = 'ring_i'
	#plot_variable(u=ring_i(X,Y),     **plt_kwargs)

	#plt_kwargs['name']      = 'div_Hu'
	#plot_variable(u=div_Hu(X,Y),     **plt_kwargs)

	#plt_kwargs['name']      = 'S'
	#plot_variable(u=S(X,Y),          **plt_kwargs)

	#plt_kwargs['name']      = 'B'
	#plot_variable(u=B(X,Y),          **plt_kwargs)

	#plt_kwargs['name']      = 'H'
	#plot_variable(u=H(X,Y),          **plt_kwargs)

	#plt_kwargs['name']      = 'dsdt'
	#plot_variable(u=dSdt(X,Y),       **plt_kwargs)

	#plt_kwargs['name']      = 'dbdt'
	#plot_variable(u=dBdt(X,Y),       **plt_kwargs)

	#plt_kwargs['name']      = 'dhdt'
	#plot_variable(u=dHdt(X,Y),       **plt_kwargs)

	#plt_kwargs['name']      = 'n_mag_S'
	#plot_variable(u=n_mag_S(X,Y),    **plt_kwargs)

	#plt_kwargs['name']      = 'n_mag_B'
	#plot_variable(u=n_mag_B(X,Y),    **plt_kwargs)

	#plt_kwargs['name']      = 'lei_resid'
	#plot_variable(u=lei_resid(X,Y),  **plt_kwargs)

	plt_kwargs['name']      = 'div_sigma_x_s'
	plot_variable(u=div_sigma_x(X,Y,S(X,Y)),    **plt_kwargs)

	plt_kwargs['name']      = 'div_sigma_y_s'
	plot_variable(u=div_sigma_y(X,Y,S(X,Y)),    **plt_kwargs)

	plt_kwargs['name']      = 'div_sigma_z_s'
	plot_variable(u=div_sigma_z(X,Y,S(X,Y)),    **plt_kwargs)


"""
# these functions allow the plotting of an arbitrary FEniCS function or
# vector that reside on a two-dimensional mesh (hence the D2Model
# instantiations above.
plot_variable(u = srfmodel.u, name = 'u_mag', direc = plt_dir,
              ext           = '.pdf',
              title         = r'$\underline{u} |_S$',
              levels        = None,#U_lvls,
              cmap          = 'viridis',
              normalize_vec = True,
              plot_tp       = True,
              show          = False,
              extend        = 'neither',
              cb_format     = '%g')

plot_variable(u = bedmodel.p, name = 'p_model', direc = plt_dir,
              ext         = '.pdf',
              title       = r'$p |_B$',
              levels      = None,#p_lvls,
              cmap        = 'viridis',
              plot_tp     = True,
              show        = False,
              extend      = 'neither',
              cb_format   = '%.1e')

plot_variable(u = drhodt_b, name = 'drhodt_model', direc = plt_dir,
              ext         = '.pdf',
              title       = r'$\left. \frac{\partial \rho}{\partial t} \right|_B$',
              cmap        = 'RdGy',
              levels      = None,#d_lvls,
              plot_tp     = True,
              show        = False,
              extend      = 'neither',
              cb_format   = '%.1e')
"""



