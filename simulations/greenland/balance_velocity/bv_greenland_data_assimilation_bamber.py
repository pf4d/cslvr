import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput,DataOutput
from data.data_factory import DataFactory
from src.physics       import VelocityBalance_2
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
mesh    = Mesh("./mesh.xml")

# create data objects to use with varglas :
dsr     = DataInput(searise, mesh=mesh, create_proj=True)
dbam    = DataInput(bamber,  mesh=mesh)
dms     = DataInput(measure, mesh=mesh, create_proj=True)

dms.change_projection(dsr)

dbam.set_data_min('H', 200.0, 200.0)
dbam.set_data_min('h', 1.0, 1.0)
dms.set_data_min('sp',0.0,-2e9)

H     = dbam.get_projection("H")
H0     = dbam.get_projection("H")
S     = dbam.get_projection("h")
adot  = dsr.get_projection("adot")

Uobs  = dms.get_projection("sp")

Uobs.vector()[Uobs.vector().array()<0] = 0.0

U_sar_spline = dms.get_spline_expression("sp")

insar_mask = CellFunctionSizet(mesh)
insar_mask.set_all(0)
for c in cells(mesh):
    x,y = c.midpoint().x(),c.midpoint().y()
    Uval = U_sar_spline(x,y)
    if Uval>0:
        insar_mask[c]=1

prb   = VelocityBalance_2(mesh, H, S, adot, 8.0,Uobs=Uobs,Uobs_mask=insar_mask)
n = len(mesh.coordinates())

Uopt_file = File('results/Uopt.pvd')
Uopt_file_xml = File('results/Uopt.xml')
Uobs_file = File('results/Uobs.pvd')
H_file = File('results/H.pvd')
adot_file = File('results/adot.pvd')
dHdt_file = File('results/dHdt.pvd')
delta_H_file = File('results/deltaH.pvd')

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
    # I/O
    Uopt_file << prb.Ubmag
    Uobs_file << prb.Uobs
    H_file << prb.H
    adot_file << prb.adot

    delta_H_file << project(prb.H - H0)
    dHdt_file << project(prb.residual-prb.adot)

    Uopt_file_xml << prb.Ubmag

    prb.solve_adjoint()
    g = prb.get_gradient()

    return hstack(g)

x0 = hstack((Uobs.vector().array(),adot.vector().array(),H.vector().array()))

Umerr = 0.05
Uaerr = 20.0

amerr = 0.1
aaerr = 1.0

Hmerr = 0.1
Haerr = 200.0

Uobs_bounds = [(max(min(r-Umerr*abs(r),r-Uaerr),0),max(r+Umerr*abs(r), \
                r+Uaerr)) for r in Uobs.vector().array()] 
ahat_bounds = [(min(r-amerr*abs(r),r-aaerr),max(r+amerr*abs(r),r+aaerr)) \ 
                for r in adot.vector().array()] 
H_bounds = [(max(min(r-Hmerr*abs(r),r-Haerr),0),max(r+Hmerr*abs(r),r+Haerr)) \
             for r in H.vector().array()] 

bounds = Uobs_bounds+ahat_bounds+H_bounds

fmin_l_bfgs_b(_I_fun,x0,fprime=_J_fun,bounds=bounds,iprint=1)
