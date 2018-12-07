from fenics  import *
from varglas import *

in_dir  = "dump/vars_jakobshavn_small/"
out_dir = 'dump/jakob_small/balance_velocity/'

# load a mesh :
fmeshes = HDF5File(mpi_comm_world(), in_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), in_dir + 'state.h5', 'r')

mesh = Mesh()
fmeshes.read(mesh, 'bedmesh', False)

d2model = D2Model(mesh,  out_dir)
d3model = D3Model(fdata, out_dir)

d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_S_ring(fdata)

d2model.assign_submesh_variable(d2model.S,    d3model.S)
d2model.assign_submesh_variable(d2model.B,    d3model.B)
d2model.assign_submesh_variable(d2model.S_ring, d3model.S_ring)

#plotIce(dsr, 'S_ring', name='', direc='.', title=r'$\dot{a}$', cmap='gist_yarg',
#        scale='lin', numLvls=12, tp=False, tpAlpha=0.5)

bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.save_pvd(d2model.Ubar, 'Ubar')
d2model.save_xml(d2model.Ubar, 'Ubar')



