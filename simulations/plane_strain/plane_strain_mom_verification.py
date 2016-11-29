from cslvr import *
import numpy as np

parameters['allow_extrapolation'] = True

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
nx      = 20               # initial number of x-element divisions
nz      = 4                # intiial number of z-elemnet divisions
mesh    = RectangleMesh(p1, p2, nx, nz)

# need Lagrange interpolator for comparing solutions on non-matching meshes :
lg = LagrangeInterpolator()

# set some output directories :
out_dir = 'ps_results/'
plt_dir = '../../images/momentum/plane_strain/'

for i in range(10):

  # this is a `lateral' model, defined in the x,z plane :
  model = LatModel(Mesh(mesh), out_dir = out_dir, use_periodic = False)
  
  # the geometry and desired traction :
  S = Expression('(Hmax+B0-S0)/2*cos(2*pi*x[0]/l) + (Hmax+B0+S0)/2',
                 Hmax=Hmax, B0=B0, S0=S0, l=l,
                 element = model.Q.ufl_element())
  B = Expression('b*cos(nb*2*pi*x[0]/l) + B0',
                 b=b, l=l, B0=B0, nb=nb,
                 element = model.Q.ufl_element())
  beta = Expression('(bMax - bMin)/2.0*cos(2*pi*x[0]/l) + (bMax + bMin)/2.0',
                    bMax=betaMax, bMin=betaMin, l=l,
                    element = model.Q.ufl_element())
  
  # deform the mesh, just like we did with the 3D model :
  model.deform_mesh_to_geometry(S, B)
  
  # calculate the boundaries for proper variational-form integration :
  model.calculate_boundaries(mask=None)
  
  # initialize the constants that we want, here like the ISMIP-HOM exp. :
  model.init_beta(beta)    # traction
  model.init_A(1e-16)      # flow-rate factor
  
  # only one type of momentum physics for this problem :
  mom = MomentumDukowiczPlaneStrain(model)
  mom.solve()

  model.save_xdmf(model.U3,  'U3')

  # keep the solution for error calculation : 
  if i == 0:
    old_u = model.U3
    mesh = refine(mesh)

  if i > 0:
    # interpolate the old solution onto the current mesh :
    old_u_int = Function(model.Q3)
    lg.interpolate(old_u_int, old_u)
    old_u     = model.U3
    
    model.save_xdmf(old_u_int, 'old_u_int')
    
    # first, calculate the error in velocities :
    u_o,w_o,x_x  = old_u_int.split(True)
    u,w,x        = model.U3.split(True)
    u_o_v        = u_o.vector().array()
    w_o_v        = w_o.vector().array()
    u_v          = u.vector().array()
    w_v          = w.vector().array()
    err_v        = np.sqrt((u_v - u_o_v)**2 + (w_v - w_o_v)**2 + DOLFIN_EPS)
    err          = Function(model.Q)
    model.assign_variable(err, err_v)

    # next, refine off of the error :
    cell_markers = CellFunction("bool", model.mesh)
    cell_markers.set_all(False)
    for cell in cells(model.mesh):
      if err(cell.midpoint()) > 1e3: cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

    model.save_xdmf(err, 'err')
    Q_m = FunctionSpace(mesh, 'CG', 1)
    model.save_xdmf(interpolate(Constant(0.0), Q_m), 'mesh')

