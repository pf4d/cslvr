from __future__ import print_function
from dolfin import *
import ufl
from six.moves import xrange as range
import os

from cslvr import *
from sympy.utilities.codegen import ccode
from sympy import symbols
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

parameters['form_compiler']['quadrature_degree'] = 4

x, y = symbols('x[0], x[1]')

a     = 0.5 * pi / 180     # surface slope in radians
L     = 40000              # width of domain (also 8000, 10000, 14000)
D     = 1000               # average ice thickness
amp   = 0.5

# create a genreic box mesh, we'll fit it to geometry below :
p1    = Point(0.0, 0.0, 0.0)          # origin
p2    = Point(L,   L,   1)            # x, y, z corner
mesh  = BoxMesh(p1, p2, 15, 15, 5)   # a box to fill the void

# output directiories :
mdl_odr = 'FS'
out_dir = './ISMIP_HOM_A_results/' + mdl_odr + '/'
plt_dir = '../../images/momentum/ISMIP_HOM_A/' + mdl_odr + '/'

# we have a three-dimensional problem here, with periodic lateral boundaries :
model = D3Model(mesh, out_dir = out_dir, use_periodic = True, order=1)

s     = -x*sp.tan(a)
#s     = -x*sp.tan(a) + amp * D * sp.sin(2*sp.pi*x/L) * sp.sin(2*sp.pi*y/L)
sx_s  = s.diff(x, 1)

b     = s - D + amp * D * sp.sin(2*sp.pi*x/L) * sp.sin(2*sp.pi*y/L)

# the ISMIP-HOM experiment A geometry :
surface = Expression(ccode(s).replace('M_PI', 'pi'), degree=2)
bed     = Expression(ccode(b).replace('M_PI', 'pi'), a=a, L=L, D=D, degree=2)

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
	#mom = MomentumBP(model)
	#mom = MomentumDukowiczBP(model)
	#mom = MomentumDukowiczStokesReduced(model)
	#mom = MomentumDukowiczStokes(model)
	#mom = MomentumDukowiczStokesOpt(model)
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
              extend      = 'min',
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



