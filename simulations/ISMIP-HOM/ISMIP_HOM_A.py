from cslvr import *

a     = 0.5 * pi / 180     # surface slope in radians
L     = 8000               # width of domain (also 8000, 10000, 14000)

# create a genreic box mesh, we'll fit it to geometry below :
p1    = Point(0.0, 0.0, 0.0)          # origin
p2    = Point(L,   L,   1)            # x, y, z corner 
mesh  = BoxMesh(p1, p2, 15, 15, 5)    # a box to fill the void 

# output directiories :
mdl_odr = 'FS'
out_dir = './ISMIP_HOM_A_results/' + mdl_odr + '/'
plt_dir = '../../images/momentum/ISMIP_HOM_A/' + mdl_odr + '/'

# we have a three-dimensional problem here, with periodic lateral boundaries :
model = D3Model(mesh, out_dir = out_dir, use_periodic = True)

# the ISMIP-HOM experiment A geometry :
surface = Expression('- x[0] * tan(a)', a=a,
                     element=model.Q.ufl_element())
bed     = Expression(  '- x[0] * tan(a) - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     a=a, L=L, element=model.Q.ufl_element())

# mark the exterior facets and interior cells appropriately :
model.calculate_boundaries()

# deform the mesh to match our desired geometry :
model.deform_mesh_to_geometry(surface, bed)

# initialize all the pertinent variables :
model.init_beta(1e16)                      # really high friction
model.init_A(1e-16)                        # cold, isothermal rate-factor

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
  mom = MomentumDukowiczBP(model)
elif mdl_odr == 'RS':
  mom = MomentumDukowiczStokesReduced(model)
elif mdl_odr == 'FS':
  mom = MomentumDukowiczStokes(model)
mom.solve()

# let's investigate the velocity divergence :
divU = project(div(model.U3))

# the purpose for everything below this line is data visualization :
#===============================================================================

# save these files with a name that makes sense for use with paraview :
model.save_xdmf(model.p,  'p')
model.save_xdmf(model.U3, 'U')
model.save_xdmf(divU,     'divU')

# create the bed and surface meshes :
model.form_bed_mesh()
model.form_srf_mesh()

# create 2D models :
bedmodel = D2Model(model.bedmesh, out_dir)
srfmodel = D2Model(model.srfmesh, out_dir)

# we don't have a function for this included in the `model' instance, 
# so we have to make one ourselves :
divU_b   = Function(bedmodel.Q)

# function allows Lagrange interpolation between different meshes :
bedmodel.assign_submesh_variable(divU_b, divU)
srfmodel.assign_submesh_variable(srfmodel.U3, model.U3)
srfmodel.init_U_mag(srfmodel.U3)  # calculates the velocity magnitude 
bedmodel.assign_submesh_variable(bedmodel.p,  model.p)

# figure out some nice-looking contour levels :
U_min  = srfmodel.U_mag.vector().min()
U_max  = srfmodel.U_mag.vector().max()
#U_lvls = array([84, 86, 88, 90, 92, 94, 96, 98, 100])  # momentum comparison
U_lvls = array([U_min, 87, 88, 89, 90, 91, 92, U_max])  # full-Stokes

p_min  = bedmodel.p.vector().min()
p_max  = bedmodel.p.vector().max()
p_lvls = array([4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 1.1e7, 1.2e7, p_max])

d_min  = divU_b.vector().min()
d_max  = divU_b.vector().max()
d_lvls = array([d_min, -5e-3, -2.5e-3, -1e-3, 
                1e-3, 2.5e-3, 5e-3, d_max])

# these functions allow the plotting of an arbitrary FEniCS function or 
# vector that reside on a two-dimensional mesh (hence the D2Model
# instantiations above.
plot_variable(u = srfmodel.U3, name = 'U_mag', direc = plt_dir,
              ext                 = '.pdf',
              title               = r'$\mathbf{u} |_S$',
              levels              = None,#U_lvls,
              cmap                = 'viridis',
              tp                  = True,
              show                = False,
              extend              = 'neither',
              cb_format           = '%g')

plot_variable(u = bedmodel.p, name = 'p', direc = plt_dir,
              ext                 = '.pdf',
              title               = r'$p |_B$',
              levels              = p_lvls,
              cmap                = 'viridis',
              tp                  = True,
              show                = False,
              extend              = 'min',
              cb_format           = '%.1e')

plot_variable(u = divU_b, name = 'divU', direc = plt_dir,
              ext                 = '.pdf',
              title               = r'$\nabla \cdot \mathbf{u} |_B$',
              cmap                = 'RdGy',
              levels              = None,#d_lvls,
              tp                  = True,
              show                = False,
              extend              = 'neither',
              cb_format           = '%.1e')



