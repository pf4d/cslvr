from varglas import *
from fenics  import *

out_dir  = 'dump/vars_low/'
thklim   = 1.0

# collect the raw data :
searise  = DataFactory.get_searise(thklim = thklim)
bamber   = DataFactory.get_bamber(thklim = thklim)
rignot   = DataFactory.get_rignot()

mesh = Mesh('dump/meshes/gre_mesh_low.xml.gz')

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
S_ring  = dsr.get_expression("S_ring",   near=False)
T_s   = dsr.get_expression("T",      near=False)
q_geo = dsr.get_expression("q_geo",  near=False)
u_ob  = drg.get_expression("vx",     near=False)
v_ob  = drg.get_expression("vy",     near=False)
U_msk = drg.get_expression("mask",   near=True)

model = D3Model(mesh=mesh, out_dir=out_dir, save_state=True)
model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=M, S_ring=S_ring, U_mask=U_msk, mark_divide=False)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

model.save_xdmf(model.ff, 'ff')
model.state.close()


