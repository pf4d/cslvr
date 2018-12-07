from cslvr import *

# use inexact integration (for full stokes) :
parameters['form_compiler']['quadrature_degree']  = 3

# this problem has a few more constants than the last :
l       = 400000.0          # width of the domain
Hmax    = 4000              # thickness at the divide
S0      = 100               # terminus height above water
B0      = -200              # terminus depth below water
nb      = 25                # number of basal bumps
b       = 50                # amplitude of basal bumps
betaMax = 50.0              # maximum traction coefficient
betaMin = 0.2               # minimum traction coefficient

# this time, we solve over a 2D domain, we'll deform to geometry below :
p1      = Point(-l/2, 0.0)  # lower left corner
p2      = Point( l/2, 1.0)  # upper right corner
nx      = 150               # number of x-element divisions
nz      = 10                # number of z-elemnet divisions
mesh    = RectangleMesh(p1, p2, nx, nz)
order   = 2
if order == 1: stabilized = True
else:          stabilized = False

# set some output directories :
out_dir = 'ps_results/'
plt_dir = '../../images/momentum/plane_strain/'

# this is a `lateral' model, defined in the x,z plane :
model = LatModel(mesh, out_dir=out_dir, use_periodic=False, order=order)

# the geometry and desired traction :
S = Expression('(Hmax+B0-S0)/2*cos(2*pi*x[0]/l) + (Hmax+B0+S0)/2',
               Hmax=Hmax, B0=B0, S0=S0, l=l,
               element = model.Q.ufl_element())
B = Expression('b*cos(nb*2*pi*x[0]/l) + B0',
               b=b, l=l, B0=B0, nb=nb,
               element = model.Q.ufl_element())
b = Expression('(bMax - bMin)/2.0*cos(2*pi*x[0]/l) + (bMax + bMin)/2.0',
               bMax=betaMax, bMin=betaMin, l=l,
               element = model.Q.ufl_element())

# deform the mesh, just like we did with the 3D model :
model.deform_mesh_to_geometry(S, B)

# save the facet marker to check :
model.save_xdmf(model.ff, 'ff')

# initialize the constants that we want, here like the ISMIP-HOM exp. :
model.init_beta(b)    # traction
model.init_A(1e-16)   # flow-rate factor

# only one type of momentum physics for this problem :
mom = MomentumDukowiczPlaneStrain(model, stabilized=stabilized)
mom.solve()

# plotting :
#===============================================================================

# open with paraview :
model.save_xdmf(model.u, 'u')
model.save_xdmf(model.p,  'p')

# let's calculate the velocity speed :
model.init_u_mag(model.u)

u_min  = model.u_mag.vector().min()
u_max  = model.u_mag.vector().max()
u_lvls = array([u_min, 5e3, 1e4, 2e4, 3e4, 4e4, 5e4, u_max])

p_min  = model.p.vector().min()
p_max  = model.p.vector().max()
p_lvls = array([p_min, 5e6, 1e7, 1.5e7, 2e7, 2.5e7, 3e7, 3.5e7, p_max])

beta_min = model.beta.vector().min()
beta_max = model.beta.vector().max()
beta_lvls = array([beta_min, 5.0, 10, 15, 20, 25, 30, 35, 40, 45, beta_max])

plot_variable(u = model.u_mag, name = 'u_mag', direc = plt_dir,
              figsize             = (8,3),
              title               = r'$\Vert \mathbf{u} \Vert$',
              cmap                = 'viridis',
              levels              = u_lvls,
              tp                  = True,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              cb_format           = '%.1e')

plot_variable(u = model.p, name = 'p', direc = plt_dir,
              figsize             = (8,3),
              title               = r'$p$',
              cmap                = 'viridis',
              levels              = None,#p_lvls,
              tp                  = True,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              cb_format           = '%.1e')

plot_variable(u = model.beta, name = 'beta', direc = plt_dir,
              figsize             = (8,3),
              title               = r'$\beta$',
              cmap                = 'viridis',
              levels              = beta_lvls,
              tp                  = True,
              show                = False,
              ylabel              = r'$z$',
              equal_axes          = False,
              cb_format           = '%g')

