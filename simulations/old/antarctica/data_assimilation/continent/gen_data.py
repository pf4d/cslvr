from cslvr   import *
from fenics  import *

msh_lvl  = 'high'

out_dir  = 'dump/vars_%s/' % msh_lvl
thklim   = 1.0
measures = DataFactory.get_ant_measures()
bedmap1  = DataFactory.get_bedmap1(thklim=thklim)
bedmap2  = DataFactory.get_bedmap2(thklim=thklim)

mesh = Mesh('dump/meshes/ant_mesh_%s.xml.gz' % msh_lvl)

dm = DataInput(measures, mesh=mesh)
d1 = DataInput(bedmap1,  mesh=mesh)
d2 = DataInput(bedmap2,  mesh=mesh)

S     = d2.get_expression("S",      near=False)
B     = d2.get_expression("B",      near=False)
M     = d2.get_expression("mask",   near=True)
S_ring  = d1.get_expression("acca",   near=False)
T_s   = d1.get_expression("temp",   near=False)
q_geo = d1.get_expression("ghfsr",  near=False)
u_ob  = dm.get_expression("vx",     near=False)
v_ob  = dm.get_expression("vy",     near=False)
U_msk = dm.get_expression("mask",   near=True)

model = D3Model(mesh=mesh, out_dir=out_dir)
model.calculate_boundaries(mask=M, U_mask=U_msk, S_ring=S_ring) 
model.deform_mesh_to_geometry(S, B)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

lst = [model.S,
       model.B,
       model.mask,
       model.S_ring,
       model.T_surface,
       model.q_geo,
       model.u_ob,
       model.v_ob,
       model.U_mask]

f = HDF5File(mpi_comm_world(), out_dir + 'state_%s.h5' % msh_lvl, 'w')

model.save_list_to_hdf5(lst, f)
model.save_subdomain_data(f)
model.save_mesh(f)

f.close()



