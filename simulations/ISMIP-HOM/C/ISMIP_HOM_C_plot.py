import pylab 
import dolfin
import pickle

nx = 50
ny = 50
nz = 10

m = dolfin.UnitCubeMesh(nx,ny,nz)
Q = dolfin.FunctionSpace(m,"CG",1)
u = dolfin.Function(Q)
v = dolfin.Function(Q)
w = dolfin.Function(Q)
S = dolfin.Function(Q)

for L in [5000,10000,20000,40000,80000,160000]:

    dolfin.File('./results_BP/'+str(L)+'/u.xml') >> u
    dolfin.File('./results_BP/'+str(L)+'/v.xml') >> v
    dolfin.File('./results_BP/'+str(L)+'/w.xml') >> w

    U = pylab.zeros(100)
    profile = pylab.linspace(0,1,100)

    for ii,x in enumerate(profile):
        uu = u(x,0.25,0.99999)
        vv = v(x,0.25,0.99999)
        ww = w(x,0.25,0.99999)
        U[ii] = pylab.sqrt(uu**2 + vv**2 + ww**2)

    pylab.plot(profile,U)
    data = zip(profile,U)
    pickle.dump(data,open("djb1c{0:03d}.p".format(L/1000),'w'))

pylab.show()

