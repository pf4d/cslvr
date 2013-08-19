import pylab 
import dolfin
import pickle

nx = 50
ny = 50
nz = 8

m = dolfin.UnitCubeMesh(nx,ny,nz)
Q = dolfin.FunctionSpace(m,"CG",1)
u = dolfin.Function(Q)
v = dolfin.Function(Q)
w = dolfin.Function(Q)
S = dolfin.Function(Q)

dolfin.File('./results_stokes/u.xml') >> u
dolfin.File('./results_stokes/v.xml') >> v
dolfin.File('./results_stokes/w.xml') >> w
dolfin.File('./results_stokes/S.xml') >> S

theta = pylab.deg2rad(-3.0)
profile = pylab.linspace(0,1,100)
S0 = pylab.sin(theta)/pylab.cos(theta)*profile*100000.0
U = pylab.zeros(100)
s = pylab.zeros(100)

for ii,x in enumerate(profile):
    uu = u(x,0.5,0.99999)
    vv = v(x,0.5,0.99999)
    ww = w(x,0.5,0.99999)
    SS = S(x,0.5,0.99999)
    U[ii] = pylab.sqrt(uu**2 + vv**2 + ww**2)
    s[ii] = SS

data = zip(profile,s-S0,U)
pickle.dump(data,open("djb2f{0:03d}.p".format(0),'w'))

pylab.plot(profile,U)
pylab.show()

pylab.plot(profile,s-S0)
pylab.show()
    


