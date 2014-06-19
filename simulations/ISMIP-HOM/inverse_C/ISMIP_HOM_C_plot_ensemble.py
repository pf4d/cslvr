import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.model import Model
from pylab     import zeros, linspace, sqrt
from dolfin    import project, as_vector, Function, File

nx = 20
ny = 20
nz = 6

model = Model()
model.generate_uniform_mesh(nx,ny,nz,0,1,0,1,deform=False,generate_pbcs=True)

Q = model.Q
U_opt = project(as_vector([Function(Q),Function(Q),Function(Q)]))
b_opt = Function(Q)
n = len(b_opt.compute_vertex_values())

rcParams['text.usetex']=True
rcParams['font.size'] = 12
rcParams['font.family'] = 'serif'

Us = zeros((50,n))
betas = zeros((50,n))

fig,axs = subplots(2,1,sharex=True)
fig.set_size_inches(8,4)

bed_indices = model.mesh.coordinates()[:,2]==0
surface_indices = model.mesh.coordinates()[:,2]==1
prof_indices = model.mesh.coordinates()[:,1]==0.25

for ii in range(50):
    File('./results/run_'+str(ii)+'/U_opt.xml') >> U_opt
    File('./results/run_'+str(ii)+'/beta2_opt.xml') >> b_opt

    betas[ii] = b_opt.compute_vertex_values()
    us = U_opt.compute_vertex_values()
    Us[ii] = sqrt(us[:n]**2 + us[n:2*n]**2 + us[2*n:]**2)
betas = betas[:,bed_indices*prof_indices]
Us = Us[:,surface_indices*prof_indices]

profile = linspace(0,1,21)
axs[0].errorbar(profile,mean(Us,axis=0),yerr=std(Us,axis=0),fmt='k-', \
                linewidth=2.0)
axs[0].set_ylabel('Velocity')
axs[1].errorbar(profile,mean(betas,axis=0),yerr=std(betas,axis=0),fmt='k-', \
                linewidth=2.0)
axs[1].set_ylabel('\\beta^2')


