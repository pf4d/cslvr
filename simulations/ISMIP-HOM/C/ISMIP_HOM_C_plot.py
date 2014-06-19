from pylab                  import zeros, linspace, sqrt, plot, show
from pickle                 import dump
from dolfin                 import UnitCubeMesh, FunctionSpace, Function, File

nx = 50
ny = 50
nz = 10

m = UnitCubeMesh(nx,ny,nz)
Q = FunctionSpace(m,"CG",1)
u = Function(Q)
v = Function(Q)
w = Function(Q)
S = Function(Q)

for L in [5000,10000,20000,40000,80000,160000]:

    File('./results_BP/'+str(L)+'/u.xml') >> u
    File('./results_BP/'+str(L)+'/v.xml') >> v
    File('./results_BP/'+str(L)+'/w.xml') >> w

    U = zeros(100)
    profile = linspace(0,1,100)

    for ii,x in enumerate(profile):
        uu = u(x,0.25,0.99999)
        vv = v(x,0.25,0.99999)
        ww = w(x,0.25,0.99999)
        U[ii] = sqrt(uu**2 + vv**2 + ww**2)

    plot(profile,U)
    data = zip(profile,U)
    dump(data,open("djb1c{0:03d}.p".format(L/1000),'w'))

show()

