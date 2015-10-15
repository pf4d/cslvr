from fenics  import *
from varglas import *

in_dir  = "dump/vars_jakobshavn/"
out_dir = 'dump/jakob_da_ipopt_SIA0_SR/00/balance_velocity'

# load a mesh :
fmeshes = HDF5File(mpi_comm_world(), in_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), in_dir + 'state.h5', 'r')

mesh = Mesh()
fmeshes.read(mesh, 'bedmesh', False)

d2model = D2Model(mesh,  out_dir + '/pvd/')
d3model = D3Model(fdata, out_dir + '/pvd/')

d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_adot(fdata)

d2model.assign_boundary_variable(d2model.S,    d3model.S)
d2model.assign_boundary_variable(d2model.B,    d3model.B)
d2model.assign_boundary_variable(d2model.adot, d3model.adot)

#plotIce(dsr, 'adot', name='', direc='.', title=r'$\dot{a}$', cmap='gist_yarg',
#        scale='lin', numLvls=12, tp=False, tpAlpha=0.5)

bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.save_pvd(d2model.Ubar, 'Ubar')
d2model.save_xml(d2model.Ubar, 'Ubar')



