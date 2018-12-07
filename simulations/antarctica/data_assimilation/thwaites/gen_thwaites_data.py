from cslvr   import *
from fenics  import *

msh_lvl  = 'crude'
out_dir  = 'dump/vars_thwaites_basin_%s/' % msh_lvl
thklim   = 100.0
measures = DataFactory.get_ant_measures()
bedmap1  = DataFactory.get_bedmap1(thklim=thklim)
bedmap2  = DataFactory.get_bedmap2(thklim=thklim)

m_n  = 'dump/meshes/thwaites_3D_U_mesh_basin_%s.xdmf' % msh_lvl
f_m  = XDMFFile(mpi_comm_world(), m_n)
mesh = Mesh()
f_m.read(mesh)

dm = DataInput(measures, mesh=mesh)
d1 = DataInput(bedmap1,  mesh=mesh)
d2 = DataInput(bedmap2,  mesh=mesh)

S     = d2.get_expression("S",        near=False)
B     = d2.get_expression("B",        near=False)
M     = d2.get_expression("mask",     near=True)
L     = d2.get_expression('lat_mask', near=True)
S_ring  = d1.get_expression("acca",     near=False)
T_s   = d1.get_expression("temp",     near=False)
q_geo = d1.get_expression("ghfsr",    near=False)
u_ob  = dm.get_expression("vx",       near=False)
v_ob  = dm.get_expression("vy",       near=False)
U_msk = dm.get_expression("mask",     near=True)

model = D3Model(mesh=mesh, out_dir=out_dir)
model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=M, lat_mask=L, U_mask=U_msk, S_ring=S_ring, 
                           mark_divide=True)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

lst = [model.S,
       model.B,
       model.mask,
       model.q_geo,
       model.T_surface,
       model.S_ring,
       model.u_ob,
       model.v_ob,
       model.U_mask,
       model.lat_mask]

f = HDF5File(mpi_comm_world(), out_dir + 'state_%s.h5' % msh_lvl, 'w')

model.save_list_to_hdf5(lst, f)
model.save_subdomain_data(f)
model.save_mesh(f)

f.close()



