import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput,DataOutput
from data.data_factory import DataFactory
from src.physics       import VelocityBalance_2
from numpy             import sqrt
from pylab             import *
from dolfin            import *
from scipy.optimize    import fmin_l_bfgs_b
import os

set_log_active(True)

# collect the raw data :
searise = DataFactory.get_searise()
measure = DataFactory.get_gre_measures()
bamber  = DataFactory.get_bamber()

direc = os.path.dirname(os.path.realpath(__file__))

# load a mesh :
mesh    = Mesh("./mesh_5km.xml")

# create data objects to use with varglas :
dsr     = DataInput(searise, mesh=mesh)
dbam    = DataInput(bamber,  mesh=mesh)
dms     = DataInput(measure, mesh=mesh)
dms.change_projection(dsr)

# Bound data to managable values
MAX_V_ERR = 500
NO_DATA = -99
MAX_V = 1e5
dbam.set_data_min('H', 200.0, 200.0)
dbam.set_data_min('h', 1.0, 1.0)
dms.set_data_min('sp',0.0,-1.0)
dms.set_data_min('ex',-MAX_V_ERR,NO_DATA)
dms.set_data_max('ex',MAX_V_ERR,NO_DATA)
dms.set_data_min('ey',-MAX_V_ERR,NO_DATA)
dms.set_data_max('ey',MAX_V_ERR,NO_DATA)

dms.set_data_min('vx',-MAX_V,NO_DATA)
dms.set_data_max('vx',MAX_V,NO_DATA)
dms.set_data_min('vy',-MAX_V,NO_DATA)
dms.set_data_max('vy',MAX_V,NO_DATA)


print "Projecting data onto mesh..."
H     = dbam.get_interpolation("H",kx=1,ky=1)
H0    = dbam.get_interpolation("H",kx=1,ky=1)
S     = dbam.get_interpolation("h",kx=1,ky=1)
Herr  = dbam.get_interpolation("Herr",kx=1,ky=1)
adot  = dsr.get_interpolation("adot",kx=1,ky=1)
Uobs   = dms.get_interpolation("sp",kx=1,ky=1)
dhdt = dsr.get_interpolation('dhdt',kx=1,ky=1)
vx  = dms.get_interpolation("vx",kx=1,ky=1) # no interpolation
vy  = dms.get_interpolation("vy",kx=1,ky=1)
vxerr  = dms.get_interpolation("ex",kx=1,ky=1) # no interpolation
vyerr  = dms.get_interpolation("ey",kx=1,ky=1)

N = as_vector([vx,vy])

verr = project(abs(sqrt(abs(vxerr*2+vyerr**2 + 1e-3))))

Uobs_o = Uobs.copy() # So that we can eliminate 'no data' point in a minute

# Multiply the observed velocities by a 'shape factor' to go
# from the surface to the vertical averaged velocities.
# I think that this can be represented pretty well with a sigmoid
# (logistic) function:
# |u| / |u_s| = .8 + .2 / (1 + exp(-(|u| - 50) * .25))
#
# Which is centered at 50 m/a and has about a 25 m/a width.
# We might experiment with these values later.

Uobs = project(Uobs * (.8 + .2 / (1 + exp(-(Uobs - 50) * .25))))

Uobs.vector()[Uobs_o.vector().array()<0] = NO_DATA # Relies on copy, Uobs_o

U_sar_spline = dms.get_spline_expression("sp")

# Create a mask that only uses lower error velocities.

VELOCITY_THRESHOLD = 0.5   # A lower limit on credible InSAR measurements.
V_ERROR_THRESHOLD = 0.15  # Errors larger than this fraction ignored

insar_mask = CellFunctionSizet(mesh)
insar_mask.set_all(0)
for c in cells(mesh):
    x,y = c.midpoint().x(),c.midpoint().y()
    Uval = U_sar_spline(x,y)
    vxerrval = vxerr(x,y)
    verrval  = verr(x,y)

    if Uval>VELOCITY_THRESHOLD and vxerrval != NO_DATA\
                            and verrval/Uval < V_ERROR_THRESHOLD:
        insar_mask[c] = 1

# Create a per vertex data type that has velocity errors where mask is true
# and infinity else where
Uerr = VertexFunctionDouble(mesh)
Uerr.set_all(MAX_V_ERR)  # This is 'infinite' error, the search is unbound
for v in vertices(mesh):
    x,y      = v.x(0), v.x(1)
    Uval     = U_sar_spline(x,y)
    vxerrval = vxerr(x,y)
    verrval  = verr(x,y)

    if Uval>VELOCITY_THRESHOLD and vxerrval != NO_DATA\
                            and verrval/Uval < V_ERROR_THRESHOLD:
        Uerr[v] = verrval
 
# Problem definition
SMOOTH_RADIUS = 20.  # How many ice thicknesses to smooth over 
prb   = VelocityBalance_2(mesh, H, S, adot, SMOOTH_RADIUS,\
              Uobs=Uobs,Uobs_mask=insar_mask,N_data = N,NO_DATA = NO_DATA,\
              alpha = [1.e9,1.e10,1.e1,1.e5])

# bounds = Uobs_bounds + ahat_bounds + H_bounds + Ny_bounds

n = len(mesh.coordinates())

# IO
Uopt_file = File('results5km/Uopt.pvd')
Uopt_file_xml = File('results5km/Uopt.xml')
Uobs_file = File('results5km/Uobs.pvd')
Uerr_file = File('results5km/Uerr.pvd')
H_file = File('results5km/H.pvd')
bed_file = File('results5km/bed.pvd')
adot_file = File('results5km/adot.pvd')
residual_file = File('results5km/residuals.pvd')
ny_file = File('results5km/ny.pvd')

H_file_xml = File('results5km/H.xml')
vx_file_xml = File('results5km/vx.xml')
bed_file_xml = File('results5km/bed.xml')
vy_file_xml = File('results5km/vy.xml')

delta_H_file = File('results5km/deltaH.pvd')
delta_U_file = File('results5km/deltaU.pvd')
delta_adot_file = File('results5km/deltaadot.pvd')

Herr_file = File('results5km/Herr.pvd')
Uerr_file = File('results5km/Uerr.pvd')
Herr_file << Herr
Uerr_file << Uerr


def _I_fun(x,*args):
    prb.Uobs.vector()[:]    = x[:n]
    prb.adot.vector()[:]    = x[n:2*n]
    prb.H.vector()[:]       = x[2*n:3*n]
    prb.dS[1].vector()[:]   = x[3*n:]

    prb.solve_forward()
    I = assemble(prb.I)
    return I

def _J_fun(x,*args):
    prb.Uobs.vector()[:]    = x[:n]
    prb.adot.vector()[:]    = x[n:2*n]
    prb.H.vector()[:]       = x[2*n:3*n]
    prb.dS[1].vector()[:]   = x[3*n:]

    # I/O
    Uopt_file << prb.Ubmag
    Uobs_file << prb.Uobs
    H_file << prb.H
    adot_file << prb.adot
    delta_U_file<<project(prb.Ubmag - prb.Uobs,dbam.func_space)
    delta_H_file << project(prb.H - H0,dbam.func_space)
    delta_adot_file << project(prb.adot - adot,dbam.func_space)
    bed_file << project(S - prb.H,dbam.func_space)
    residual_file << prb.residual
    ny_file << prb.dS[1]

    print "================================================================"
    print "Total apparent mass balance (Gigatons):"
    print assemble(prb.adot * dx) / 1.e12
    print "================================================================"
    print "RMS Error in Velocity:"
    print sqrt(assemble((prb.Ubmag - prb.Uobs)**2 * prb.dx_masked(1)) / \
          assemble(project(Constant(1),prb.Q) * prb.dx_masked(1)))
    print "================================================================"

    prb.solve_adjoint()
    g = prb.get_gradient()
    print args
    for i in args:
        g[i][:] = 0.

    return hstack(g)


amerr = 1.0
aaerr = 40.0
ahat_bounds = [(min(r-amerr*abs(r),r-aaerr),max(r+amerr*abs(r),r+aaerr)) \
              for r in adot.vector().array()] 

small_u = VELOCITY_THRESHOLD/2. # Minimal error in velocity
Uobs_bounds = [(min(Uobs_i - Uerr_i,Uobs_i-small_u), max(Uobs_i+Uerr_i, \
                Uobs_i+small_u)) for Uobs_i,Uerr_i in \
                zip(Uobs.vector().array(),Uerr.array())] 

small_h = 35. # Minimal uncertainty: replaces Bamber's zeros. 
              # ~35 is what Morlighem used
H_bounds = [(min(H_i - Herr_i,H_i-small_h), max(H_i + Herr_i,H_i+small_h)) \
            for H_i,Herr_i in zip(H.vector().array(),Herr.vector().array())] 

tan2 = tan(deg2rad(10.))**2
Ny_err = sqrt(tan2 / (1+tan2)) # Arguement is in degrees

prb.solve_forward() # To assure the following has values to refer to:
Ny_err = project(Constant(Ny_err) * (.1 + (1. / (1 + exp(-(prb.Ubmag - 50.) * \
                 .25)))),prb.Q)


Ny_bounds = [(min(Ny-Ny_err_i,-1.),max(Ny+Ny_err_i,1.)) for Ny,Ny_err_i \
              in zip(prb.dS[1].vector().array(),Ny_err.vector().array())] 

bounds = Uobs_bounds + ahat_bounds + H_bounds + Ny_bounds
x0 = hstack((Uobs.vector().array(),adot.vector().array(),\
         H.vector().array(),prb.dS[1].vector().array()))

# Search in the direction of each gradient, sequentially
for f in range(20):
    freeze = [(f)%4,(f+1)%4,(f+2)%4]
    fmin_l_bfgs_b(_I_fun,x0,fprime=_J_fun,bounds=bounds,iprint=1, \
                  args=(freeze),maxfun=20)
    x0 = hstack((prb.Uobs.vector().array(),prb.adot.vector().array(),\
             prb.H.vector().array(),prb.dS[1].vector().array()))

Uopt_file_xml << prb.Ubmag
bed_file_xml << project(S - prb.H,dbam.func_space)
vx_file_xml << project(prb.Ubmag * prb.dS[0],dbam.func_space)
vy_file_xml << project(prb.Ubmag * prb.dS[1],dbam.func_space)
H_file_xml << prb.H
