from pylab        import linspace, zeros, sqrt, plot, show
from dolfin       import UnitCubeMesh, FunctionSpace, Function, File, pi, tan
from pickle       import dump

nx = 50
ny = 50
nz = 8

m = UnitCubeMesh(nx,ny,nz)
Q = FunctionSpace(m,"CG",1)
u = Function(Q)
v = Function(Q)
w = Function(Q)
S = Function(Q)

File('./results_stokes/u.xml') >> u
File('./results_stokes/v.xml') >> v
File('./results_stokes/w.xml') >> w
File('./results_stokes/S.xml') >> S

theta = -3.0 * pi / 180
profile = linspace(0,1,100)
S0 = tan(theta)*profile*100000.0
U = zeros(100)
s = zeros(100)

for ii,x in enumerate(profile):
    uu = u(x,0.5,0.99999)
    vv = v(x,0.5,0.99999)
    ww = w(x,0.5,0.99999)
    SS = S(x,0.5,0.99999)
    U[ii] = sqrt(uu**2 + vv**2 + ww**2)
    s[ii] = SS

data = zip(profile,s-S0,U)
dump(data,open("djb2f{0:03d}.p".format(0),'w'))

plot(profile,U)
show()

plot(profile,s-S0)
show()
    


