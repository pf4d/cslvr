from varglas import D3Model, MeshFactory, DataFactory, DataInput
from fenics  import *

out_dir  = 'dump/vars_high/'
thklim   = 1.0
measures = DataFactory.get_ant_measures(res=900)
bedmap1  = DataFactory.get_bedmap1(thklim=thklim)
bedmap2  = DataFactory.get_bedmap2(thklim=thklim)

mesh = Mesh('dump/meshes/ant_mesh_high.xml.gz')

dm = DataInput(measures, mesh=mesh)
d1 = DataInput(bedmap1,  mesh=mesh)
d2 = DataInput(bedmap2,  mesh=mesh)

S     = d2.get_expression("S",      near=False)
B     = d2.get_expression("B",      near=False)
M     = d2.get_expression("mask",   near=True)
adot  = d1.get_expression("acca",   near=False)
T_s   = d1.get_expression("temp",   near=False)
q_geo = d1.get_expression("ghfsr",  near=False)
u_ob  = dm.get_expression("vx",     near=False)
v_ob  = dm.get_expression("vy",     near=False)

model = D3Model(mesh=mesh, out_dir=out_dir, save_state=True)
model.calculate_boundaries(mask=M, adot=adot)
model.deform_mesh_to_geometry(S, B)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

model.state.write(model.mesh,   'mesh')

#XDMFFile(mesh.mpi_comm(),    out_dir + 'mesh.xdmf')    << model.mesh
#

## save the state of the model :
#f = HDF5File(mesh.mpi_comm(), out_dir + 'vars.h5', 'w')
#f.write(model.ff,     'ff')
#f.write(model.cf,     'cf')
#f.write(model.ff_acc, 'ff_acc')
#f.write(model.S,      'S')
#f.write(model.B,      'B')
#f.write(model.adot,   'adot')
#f.write(model.mask,   'mask')
#f.write(T_s,          'T_s')
#f.write(q_geo,        'q_geo')
#f.write(u,            'u_ob')
#f.write(v,            'v_ob')
#f.write(U_ob,         'U_ob')



