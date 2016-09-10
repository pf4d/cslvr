from varglas import *
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

S     = d2.get_cslvr_expression("S",      near=False)
B     = d2.get_cslvr_expression("B",      near=False)
M     = d2.get_cslvr_expression("mask",   near=True)
adot  = d1.get_cslvr_expression("acca",   near=False)
T_s   = d1.get_cslvr_expression("temp",   near=False)
q_geo = d1.get_cslvr_expression("ghfsr",  near=False)
u_ob  = dm.get_cslvr_expression("vx",     near=False)
v_ob  = dm.get_cslvr_expression("vy",     near=False)

model = D3Model(mesh=mesh, out_dir=out_dir, save_state=True)
model.calculate_boundaries(mask=M, adot=adot, mark_divide=False)
model.deform_mesh_to_geometry(S, B)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)



