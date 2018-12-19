from cslvr      import *
from fenics_viz import plot_variable

a      = 0.5 * pi / 180     # surface slope in radians
L      = 14000               # width of domain (also 8000, 10000, 14000)
order  = 2
linear = True

# use inexact integration (for full stokes) :
parameters['form_compiler']['quadrature_degree']  = order + 1

# create a genreic box mesh, we'll fit it to geometry below :
p1    = Point(0.0, 0.0, 0.0)          # origin
p2    = Point(L,   L,   1)            # x, y, z corner
nxy   = 15
nz    = 5
mesh  = BoxMesh(p1, p2, nxy, nxy, nz)    # a box to fill the void

# output directiories :
mdl_odr = 'FS_th'
out_dir = './ISMIP_HOM_A_results/' + mdl_odr + '/'
plt_dir = '../../images/momentum/ISMIP_HOM_A/' + mdl_odr + '/'

# we have a three-dimensional problem here, with periodic lateral boundaries :
model = D3Model(mesh, out_dir=out_dir, use_periodic=False, order=order)

# the ISMIP-HOM experiment A geometry :
surface = Expression('500 - x[0] * tan(a)', a=a,
                     element=model.Q.ufl_element())
bed     = Expression(  'S - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     S=surface, a=a, L=L, element=model.Q.ufl_element())

# deform the mesh to match our desired geometry :
model.deform_mesh_to_geometry(surface, bed)

model.calc_normal_vector()
#model.calc_basal_surface_normal_vector()

# initialize all the pertinent variables :
model.init_B_ring(-2)           # negative bmb (melting)
model.init_beta(1e4)            # friction coefficient
model.init_A(1e-16)             # warm isothermal flow-rate factor

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
	mom = MomentumBP(model, linear=linear)
elif mdl_odr == 'BP_duk':
	mom = MomentumDukowiczBP(model, linear=linear)
elif mdl_odr == 'RS':
	mom = MomentumDukowiczStokesReduced(model, linear=linear)
elif mdl_odr == 'FS_duk_stab':
	mom = MomentumDukowiczStokes(model, linear=linear, stabilized=True)
elif mdl_odr == 'FS_duk_th':
	mom = MomentumDukowiczStokes(model, linear=linear, stabilized=False)
elif mdl_odr == 'FS_nit_stab':
	mom = MomentumNitscheStokes(model, linear=linear, stabilized=True)
elif mdl_odr == 'FS_nit_th':
	mom = MomentumNitscheStokes(model, linear=linear, stabilized=False)
elif mdl_odr == 'FS_stab':
	mom = MomentumStokes(model, linear=linear, stabilized=True)
elif mdl_odr == 'FS_th':
	mom = MomentumStokes(model, linear=linear, stabilized=False)
if linear:
	momNL = MomentumDukowiczStokes(model, linear=False, stabilized=False)
	momNL.solve()
	model.save_xdmf(model.p, 'p_true')
	model.save_xdmf(model.u, 'u_true')

"""
A_n    = Matrix(PETScMatrix(A_n))
B_n    = Matrix(PETScMatrix(B_n))
BT_n   = Matrix(PETScMatrix(BT_n))
S_n    = Matrix(PETScMatrix(S_n))
f_n    = Vector(PETScVector(f_n))

plt.imshow(A_n.array())
plt.colorbar()
plt.tight_layout()
plt.show()

plt.imshow(B_n.array())
plt.colorbar()
plt.tight_layout()
plt.show()

plt.imshow(BT_n.array())
plt.colorbar()
plt.tight_layout()
plt.show()

plt.imshow(S_n.array())
plt.colorbar()
plt.tight_layout()
plt.show()
"""

mom.solve()

# let's investigate the velocity divergence :
if   mdl_odr == 'BP' or mdl_odr == 'BP_duk' or mdl_odr == 'RS':
	u_x, u_y = mom.get_unknown()
	u_z      = mom.u_z
elif    mdl_odr == 'FS' \
     or mdl_odr == 'FS_th'     or mdl_odr == 'FS_stab' \
     or mdl_odr == 'FS_nit_th' or mdl_odr == 'FS_nit_stab' \
     or mdl_odr == 'FS_duk_th' or mdl_odr == 'FS_duk_stab' :
	u_x, u_y, u_z, p  = mom.get_unknown()
drhodt = project(model.rho_i*div(as_vector([u_x, u_y, u_z])), model.Q)

# the purpose for everything below this line is data visualization :
#===============================================================================

# save these files with a name that makes sense for use with paraview :
model.save_xdmf(model.p, 'p')
model.save_xdmf(model.u, 'u')
model.save_xdmf(drhodt,  'drhodt')

# create the bed and surface meshes :
model.form_bed_mesh()
model.form_srf_mesh()

# create 2D models :
# note that this is only used for plotting, so we only need O(1) function
# spaces.  If you want to communication calculations between models, you will
# have lower error if a higher order function space is used :
bedmodel = D2Model(model.bedmesh, out_dir, order=1)
srfmodel = D2Model(model.srfmesh, out_dir, order=1)

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

# figure out some nice-looking contour levels :
U_min  = srfmodel.u_mag.vector().min()
U_max  = srfmodel.u_mag.vector().max()
#U_lvls = array([84, 86, 88, 90, 92, 94, 96, 98, 100])  # momentum comparison
U_lvls = array([U_min, 87, 88, 89, 90, 91, 92, U_max])

p_min  = bedmodel.p.vector().min()
p_max  = bedmodel.p.vector().max()
p_lvls = array([4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 1.1e7, 1.2e7, p_max])

d_min  = drhodt_b.vector().min()
d_max  = drhodt_b.vector().max()
d_lvls = array([d_min, -5e-3, -2.5e-3, -1e-3,
                1e-3, 2.5e-3, 5e-3, d_max])

cmap = 'inferno'

quiver_kwargs = {'pivot'          : 'middle',
                 'color'          : '0.8',
                 'alpha'          : 1.0,
                 'width'          : 0.004,
                 'headwidth'      : 4.0,
                 'headlength'     : 4.0,
                 'headaxislength' : 4.0}

plt_kwargs =    {"direc"          : plt_dir,
                 "ext"            : '.pdf',
                 "levels"         : None,
                 "cmap"           : cmap,
                 "plot_tp"        : True,
                 "show"           : False,
                 "normalize_vec"  : True,
                 "quiver_kwargs"  : quiver_kwargs,
                 "extend"         : 'neither'}

# these functions allow the plotting of an arbitrary FEniCS function or
# vector that reside on a two-dimensional mesh (hence the D2Model
# instantiations above.
plot_variable(u         = srfmodel.u,
              name      = 'u_s',
              title     = r'$\underline{u} |_S$',
              cb_format = '%g',
              **plt_kwargs)

plot_variable(u         = bedmodel.u,
              name      = 'u_b',
              title     = r'$\underline{u} |_B$',
              cb_format = '%g',
              **plt_kwargs)

plot_variable(u         = bedmodel.p,
              name      = 'p',
              title     = r'$p |_B$',
              cb_format = '%.1e',
              **plt_kwargs)

plot_variable(u         = drhodt_b,
              name      = 'drhodt',
              title     = r'$\left.\frac{\partial \rho}{\partial t}\right|_B$',
              cb_format = '%.1e',
              **plt_kwargs)



