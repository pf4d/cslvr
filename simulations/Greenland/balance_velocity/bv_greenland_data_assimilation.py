import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities import DataInput,DataOutput
from data.data_factory import DataFactory
from src.physics import VelocityBalance_2
from pylab import *
from dolfin import *
import os

set_log_active(True)

# collect the raw data :
searise = DataFactory.get_searise()
measure = DataFactory.get_gre_measures()
bamber = DataFactory.get_bamber()

direc = os.path.dirname(os.path.realpath(__file__))

# load a mesh :
mesh = Mesh("./mesh.xml")

# create data objects to use with varglas :
dsr = DataInput(None, searise, mesh=mesh, create_proj=True)
dbam = DataInput(None, bamber, mesh=mesh)
dms = DataInput(None, measure, mesh=mesh, create_proj=True, flip=True)

dms.change_projection(dsr)

dbam.set_data_min('H', 10.0, 10.0)
dsr.set_data_min('U_ob',0.0,0.0)
dms.set_data_val('sp',-0.1,-2e9)

H = dbam.get_interpolation("H")
S = dbam.get_interpolation("h")
adot = dsr.get_interpolation("adot")
#Uobs = dms.get_interpolation("sp",kx=1,ky=1)
Uobs = dsr.get_interpolation("U_ob",kx=1,ky=1)
Uobs.vector()[Uobs.vector().array()<0] = 0.0

U_sar_spline = dms.get_spline_expression("sp",kx=1,ky=1)
insar_mask = CellFunctionSizet(mesh)
insar_mask.set_all(0)
for c in cells(mesh):
    x,y = c.midpoint().x(),c.midpoint().y()
    Uval = U_sar_spline(x,y)
    print Uval
    if Uval>0:
        insar_mask[c]=1

prb = VelocityBalance_2(mesh, H, S, adot, 8.0,Uobs=Uobs,Uobs_mask=insar_mask,alpha=[0.0,1e7,0.0])
n = len(mesh.coordinates())

Uopt_file = File('results/Uopt.pvd')
Uobs_file = File('results/Uobs.pvd')
H_file = File('results/H.pvd')
adot_file = File('results/adot.pvd')
Uopt_file = File('results/Uopt.pvd')
Uobs_diff_file = File('results/Uobs_diff.pvd')
H_diff_file = File('results/H_diff.pvd')
adot_diff_file = File('results/adot_diff.pvd')

Uobs_init = Function(Uobs)
H_init = Function(H)
adot_init = Function(adot)

def _I_fun(x):
    prb.Uobs.vector()[:] = x[:n]
    prb.adot.vector()[:] = x[n:2*n]
    prb.H.vector()[:] = x[2*n:]
    prb.solve_forward()
    I = assemble(prb.I)
    return I

def _J_fun(x):
    prb.Uobs.vector()[:] = x[:n]
    prb.adot.vector()[:] = x[n:2*n]
    prb.H.vector()[:] = x[2*n:]
    Uopt_file << prb.Ubmag
    Uobs_file << prb.Uobs
    H_file << prb.H
    adot_file << prb.adot
    Uobs_diff_file << project(prb.Uobs-Uobs_init)
    adot_diff_file << project(prb.adot-adot_init)
    H_diff_file << project(prb.H-H_init)

    prb.solve_adjoint()
    g = prb.get_gradient()
    print mean(g[0]),mean(g[1]),mean(g[2])
    return hstack(g)

from scipy.optimize import fmin_l_bfgs_b,fmin_tnc

x0 = hstack((Uobs.vector().array(),adot.vector().array(),H.vector().array()))

Umerr = 0.1
Uaerr = 100.0

amerr = 0.5
aaerr = 1.0

Hmerr = 0.1
Haerr = 100.0

Uobs_bounds = []
for r in Uobs.vector().array():
    if r>0:
        Uobs_bounds.append((max(min(r-Umerr*abs(r),r-Uaerr),0),max(r+Umerr*abs(r),r+Uaerr)))
    else:
        Uobs_bounds.append((0.0,1e5))

#Uobs_bounds = [(max(min(r-Umerr*abs(r),r-Uaerr),0),max(r+Umerr*abs(r),r+Uaerr)) for r in Uobs.vector().array()]
ahat_bounds = [(min(r-amerr*abs(r),r-aaerr),max(r+amerr*abs(r),r+aaerr)) for r in adot.vector().array()]
H_bounds = [(max(min(r-Hmerr*abs(r),r-Haerr),10.),max(r+Hmerr*abs(r),r+Haerr)) for r in H.vector().array()]

bounds = Uobs_bounds+ahat_bounds+H_bounds

fmin_l_bfgs_b(_I_fun,x0,fprime=_J_fun,m=100,bounds=bounds,iprint=1,factr=1.0,maxiter=200)
#fmin_tnc(_I_fun,x0,fprime=_J_fun,bounds=bounds)
