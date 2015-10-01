from varglas import D3Model, MeshFactory, DataFactory, DataInput
from fenics  import *

out_dir  = 'dump/vars_high/'
thklim   = 1.0

# collect the raw data :
searise  = DataFactory.get_searise(thklim = thklim)
bamber   = DataFactory.get_bamber(thklim = thklim)
rignot   = DataFactory.get_rignot()

mesh = Mesh('dump/meshes/gre_mesh_high.xml.gz')

# create data objects to use with varglas :
dsr     = DataInput(searise,  mesh=mesh)
dbm     = DataInput(bamber,   mesh=mesh)
drg     = DataInput(rignot,   mesh=mesh)

# change all the data to the same projection (Rignot) :
dbm.change_projection(rignot)
dsr.change_projection(drg)

S     = dbm.get_expression("S",      near=False)
B     = dbm.get_expression("B",      near=False)
M     = dbm.get_expression("mask",   near=True)
adot  = dsr.get_expression("adot",   near=False)
T_s   = dsr.get_expression("T",      near=False)
q_geo = dsr.get_expression("q_geo",  near=False)
u_ob  = drg.get_expression("vx",     near=False)
v_ob  = drg.get_expression("vy",     near=False)

model = D3Model(mesh=mesh, out_dir=out_dir, save_state=True)
model.calculate_boundaries(mask=M, adot=adot)
model.deform_mesh_to_geometry(S, B)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

model.state.write(model.mesh,   'mesh')



