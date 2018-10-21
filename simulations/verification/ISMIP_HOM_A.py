from cslvr                   import *
from sympy.utilities.codegen import ccode
import sympy                     as sp

parameters['form_compiler']['quadrature_degree'] = 4

a       = 0.5 * sp.pi / 180     # surface slope in radians
L       = 40000.0               # width of domain (also 8000, 10000, 14000)
b_bar   = 1000.0                # average ice thickness
amp     = 0.5                   # surface modulation amplitude
lam     = 2                     # deformation parameter in x direction
u_mag   = 1.0                   # average x component of velocity at the surf.
u_x_amp = 0.5                   # x-velocity modulation amplitude
L       = 1.0                   # width of domain (also 8000, 10000, 14000)
b_bar   = 0.5                   # average ice thickness
amp     = 0.1                   # surface modulation amplitude
lam     = 2                     # deformation parameter in x direction
u_mag   = 1.0                   # average x component of velocity at the surf.
u_x_amp = 0.5                   # x-velocity modulation amplitude

# upper surface :
def s(x,y):
	return amp * sp.sin(3*sp.pi*x/L)
	#return -x*sp.tan(a)
	#return sp.Rational(0.0)

# lower surface
def b(x,y):
	#return s(x,y) - b_bar
	return s(x,y) - b_bar + amp * sp.sin(2*sp.pi*x/L)# * sp.sin(2*sp.pi*y/L)

# rate of change of upper surface :
def dsdt(x,y):
	return 3 * ((y - L/2)**2 - (x - L/2)**2)

# rate of change of lower surface :
def dbdt(x,y):
	return sp.sin(sp.pi*x/L) * sp.sin(sp.pi*y/L)

# upper-surface-mass balance :
def ring_s(x,y):
	#return sp.Rational(1.0)
	return sp.sin(4*sp.pi*x/L) * sp.sin(4*sp.pi*y/L)

# lower-surface-mass balance :
def ring_b(x,y):
	return sp.sin(2*sp.pi*x/L) * sp.sin(2*sp.pi*y/L)

# x-component of velocity at the upper surface :
def u_xs(x,y):
	return u_mag - u_x_amp * sp.sin(2*sp.pi*x/L) * sp.sin(2*sp.pi*y/L)

# x-component of velocity at the lower surface :
def u_xb(x,y):
	#return 0.5 * u_xs(x,y)
	return 0.5*(u_mag - u_x_amp * sp.sin(3*sp.pi*x/L) * sp.sin(3*sp.pi*y/L))

ver = Verification(s, b, u_xs, u_xb, dsdt, dbdt, ring_s, ring_b, lam)
ver.verify_analytic_solution(nx=1000, ny=1000, Lx=L, Ly=L)

import sys; sys.exit(0)

# create a genreic box mesh, we'll fit it to geometry below :
p1    = Point(0.0, 0.0, 0.0)          # origin
p2    = Point(L,   L,   1)            # x, y, z corner
mesh  = BoxMesh(p1, p2, 15, 15, 5)    # a box to fill the void

# output directiories :
mdl_odr = 'FS'
out_dir = './ISMIP_HOM_A_results/' + mdl_odr + '/'
plt_dir = '../../images/momentum/ISMIP_HOM_A/' + mdl_odr + '/'

# we have a three-dimensional problem here, with periodic lateral boundaries :
model   = D3Model(mesh, out_dir = out_dir, use_periodic = True, order=1)


# the ISMIP-HOM experiment A geometry :
surface = Expression(ccode(ver.s), degree=2)
bed     = Expression(ccode(ver.b), degree=2)

# mark the exterior facets and interior cells appropriately :
model.calculate_boundaries()

# deform the mesh to match our desired geometry :
model.deform_mesh_to_geometry(surface, bed)

model.save_xdmf(model.ff, 'ff')

# initialize all the pertinent variables :
model.init_Fb(0.0)
model.init_beta(1e16)                       # really high friction
model.init_A(1e-16)                         # cold, isothermal rate-factor
#model.init_lam_basal_pressure()
#model.solve_hydrostatic_pressure(annotate=False)

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
	mom = MomentumDukowiczBP(model)
elif mdl_odr == 'RS':
	mom = MomentumDukowiczStokesReduced(model)
elif mdl_odr == 'FS':
	mom = MomentumNitscheStokes(model, stabilized=True)
mom.solve()

# let's investigate the velocity divergence :
if mdl_odr == 'BP':
	u,v      = mom.get_U()
	w        = mom.wf
elif mdl_odr == 'FS':
	u,v,w,p  = mom.get_U()
drhodt = project(model.rhoi*div(as_vector([u,v,w])), model.Q)

# the purpose for everything below this line is data visualization :
#===============================================================================

# save these files with a name that makes sense for use with paraview :
model.save_xdmf(model.p,  'p')
model.save_xdmf(model.U3, 'U')
model.save_xdmf(drhodt,     'drhodt')

# create the bed and surface meshes :
model.form_bed_mesh()
model.form_srf_mesh()

# create 2D models :
bedmodel = D2Model(model.bedmesh, out_dir)
srfmodel = D2Model(model.srfmesh, out_dir)

# we don't have a function for this included in the `model' instance,
# so we have to make one ourselves :
drhodt_b = Function(bedmodel.Q, name='drhodt')

# function allows Lagrange interpolation between different meshes :
bedmodel.assign_submesh_variable(drhodt_b, drhodt)
srfmodel.assign_submesh_variable(srfmodel.U3, model.U3)
srfmodel.init_U_mag(srfmodel.U3)  # calculates the velocity magnitude
bedmodel.assign_submesh_variable(bedmodel.p,  model.p)

# figure out some nice-looking contour levels :
U_min  = srfmodel.U_mag.vector().min()
U_max  = srfmodel.U_mag.vector().max()
#U_lvls = array([84, 86, 88, 90, 92, 94, 96, 98, 100])  # momentum comparison
U_lvls = array([U_min, 87, 88, 89, 90, 91, 92, U_max])

p_min  = bedmodel.p.vector().min()
p_max  = bedmodel.p.vector().max()
p_lvls = array([4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 1.1e7, 1.2e7, p_max])

d_min  = drhodt_b.vector().min()
d_max  = drhodt_b.vector().max()
d_lvls = array([d_min, -5e-3, -2.5e-3, -1e-3,
                1e-3, 2.5e-3, 5e-3, d_max])

# these functions allow the plotting of an arbitrary FEniCS function or
# vector that reside on a two-dimensional mesh (hence the D2Model
# instantiations above.
plot_variable(u = srfmodel.U3, name = 'U_mag', direc = plt_dir,
              ext         = '.pdf',
              title       = r'$\underline{u} |_S$',
              levels      = None,#U_lvls,
              cmap        = 'viridis',
              tp          = True,
              show        = False,
              extend      = 'neither',
              cb_format   = '%g')

plot_variable(u = bedmodel.p, name = 'p', direc = plt_dir,
              ext         = '.pdf',
              title       = r'$p |_B$',
              levels      = None,#p_lvls,
              cmap        = 'viridis',
              tp          = True,
              show        = False,
              extend      = 'neither',
              cb_format   = '%.1e')

plot_variable(u = drhodt_b, name = 'drhodt', direc = plt_dir,
              ext         = '.pdf',
              title       = r'$\left. \frac{\partial \rho}{\partial t} \right|_B$',
              cmap        = 'RdGy',
              levels      = None,#d_lvls,
              tp          = True,
              show        = False,
              extend      = 'neither',
              cb_format   = '%.1e')



