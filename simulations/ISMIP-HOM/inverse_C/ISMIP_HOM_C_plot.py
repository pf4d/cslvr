import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.model import Model
from pylab     import zeros, linspace, sqrt
from dolfin    import project, Function, File, as_vector

nx = 40
ny = 40
nz = 7

model = Model()
model.generate_uniform_mesh(nx,ny,nz,0,1,0,1,deform=False,generate_pbcs=True)

Q = model.Q
U_obs = project(dolfin.as_vector([Function(Q),Function(Q)]))
b_obs = Function(Q)
U_opt = project(as_vector([Function(Q),Function(Q),Function(Q)]))
b_opt = Function(Q)

rcParams['text.usetex']=True
rcParams['font.size'] = 12
rcParams['font.family'] = 'serif'

for L in [10000]:
    
    File('./results/U_obs.xml') >> U_obs
    File('./results/U_opt.xml') >> U_opt
    File('./results/beta2_obs.xml') >> b_obs
    File('./results/beta2_opt.xml') >> b_opt

    U_b = zeros(100)
    U_p = zeros(100)
    b_b = zeros(100)
    b_p = zeros(100)
    profile = linspace(0,1,100)

    for ii,x in enumerate(profile):
        uu,vv = U_obs(x,0.25,0.99999)
        U_b[ii] = sqrt(uu**2 + vv**2)
        uu,vv,ww = U_opt(x,0.25,0.99999)
        U_p[ii] = sqrt(uu**2 + vv**2)
        b_b[ii] = b_obs(x,0.25,0.0)
        b_p[ii] = b_opt(x,0.25,0.0)

fig,axs = subplots(2,1,sharex=True)
fig.set_size_inches(8,4)

ax = axs[0]
ax.plot(profile,U_p,'k-')
ax.plot(profile,U_b,'b.',ms=4.0)
ax.set_ylabel('Surface Speed (m/a)')

ax = axs[1]
ax.plot(profile,b_p,'k-',label='Computed')
ax.plot(profile,b_b,'b.',ms=4.0,label='Known')
ax.set_ylabel('$\\beta^2$ (Pa s/m)')
ax.set_xlabel('Normalized x-coordinate')
ax.legend(fontsize=12)

fig.savefig('inverse_C.pdf')







