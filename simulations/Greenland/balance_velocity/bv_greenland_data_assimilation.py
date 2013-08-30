import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput,DataOutput
from data.data_factory import DataFactory
from src.physics       import VelocityBalance_2
from pylab import *
from dolfin            import *
import os

set_log_active(True)

# collect the raw data :
searise = DataFactory.get_searise()
measure = DataFactory.get_gre_measures()
#v2      = DataFactory.get_V2()

direc = os.path.dirname(os.path.realpath(__file__))

# load a mesh :
mesh    = Mesh("../meshes/mesh.xml")

# create data objects to use with varglas :
dsr     = DataInput(None, searise, mesh=mesh, create_proj=True)
#dv2     = DataInput(None, v2,      mesh=mesh)
dms     = DataInput(None, measure, mesh=mesh, create_proj=True, flip=True)

dms.change_projection(dsr)

dsr.set_data_min('H', 10.0, 10.0)
dsr.set_data_min('U_ob',0.0,0.0)
dsr.set_data_val('U_sar',0.0,-2e9)

H     = dsr.get_projection("H")
S     = dsr.get_projection("h")
adot  = dsr.get_projection("adot")
Uobs  = dsr.get_projection("U_ob")
Uobs.vector()[Uobs.vector().array()<0] = 0.0

U_sar_spline = dsr.get_spline_expression("U_sar",kx=1,ky=1)
insar_mask = CellFunctionSizet(mesh)
insar_mask.set_all(0)
for c in cells(mesh):
    x,y = c.midpoint().x(),c.midpoint().y()
    Uval = U_sar_spline(x,y)
    print Uval
    if Uval>0:
        insar_mask[c]=1

prb   = VelocityBalance_2(mesh, H, S, adot, 8.0,Uobs=Uobs,Uobs_mask=insar_mask)
n = len(mesh.coordinates())

Uopt_file = File('results/Uopt.pvd')
Uobs_file = File('results/Uobs.pvd')
H_file = File('results/H.pvd')
adot_file = File('results/adot.pvd')

def _I_fun(x):
    prb.Uobs.vector()[:] = x[:n]
    prb.adot.vector()[:] = x[n:2*n]
    prb.H.vector()[:]    = x[2*n:]
    prb.solve_forward()
    I = assemble(prb.I)
    return I

def _J_fun(x):
    prb.Uobs.vector()[:] = x[:n]
    prb.adot.vector()[:] = x[n:2*n]
    prb.H.vector()[:]    = x[2*n:]
    Uopt_file << prb.Ubmag
    Uobs_file << prb.Uobs
    H_file << prb.H
    adot_file << prb.adot

    prb.solve_adjoint()
    g = prb.get_gradient()


    return hstack(g)

from scipy.optimize import fmin_l_bfgs_b

x0 = hstack((Uobs.vector().array(),adot.vector().array(),H.vector().array()))

Umerr = 0.2
Uaerr = 50.0

amerr = 0.25
aaerr = 0.5

Hmerr = 0.1
Haerr = 100.0

Uobs_bounds = [(max(min(r-Umerr*abs(r),r-Uaerr),0),max(r+Umerr*abs(r),r+Uaerr)) for r in Uobs.vector().array()] 
ahat_bounds = [(min(r-amerr*abs(r),r-aaerr),max(r+amerr*abs(r),r+aaerr)) for r in adot.vector().array()] 
H_bounds = [(max(min(r-Hmerr*abs(r),r-Haerr),0),max(r+Hmerr*abs(r),r+Haerr)) for r in H.vector().array()] 

bounds = Uobs_bounds+ahat_bounds+H_bounds

fmin_l_bfgs_b(_I_fun,x0,fprime=_J_fun,bounds=bounds,iprint=1)


