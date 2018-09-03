from   cslvr import *
import numpy     as np

out_dir  = './dump/vars/'
msh_dir  = './dump/meshes/'

# collect the raw data :
searise  = DataFactory.get_searise()
bedmach  = DataFactory.get_bedmachine(thklim=1.0)
mouginot = DataFactory.get_mouginot()

# define the mesh :
mesh     = Mesh(msh_dir + 'nioghalvfjerdsbrae.xml.gz')

# retrieve the domain contour 
contour  = np.loadtxt(msh_dir + 'contour.txt')

# create data objects to use with varglas :
dsr      = DataInput(searise,  mesh=mesh)
dbm      = DataInput(bedmach,  mesh=mesh)
dmg      = DataInput(mouginot, mesh=mesh)

dbm.data['S'][dbm.data['S'] < 1.0] = 1.0

# change the projection of all data to be the same as the mesh :
dsr.change_projection(dbm)

# get the expressions used by varglas :
S     = dbm.get_expression('S',        near=False)
B     = dbm.get_expression('B',        near=False)
M     = dbm.get_expression('mask',     near=True)
adot  = dsr.get_expression('adot',     near=False)
q_geo = dsr.get_expression('q_geo',    near=False)
T_s   = dsr.get_expression('T',        near=False)
u_ob  = dmg.get_expression('vx',       near=False)
v_ob  = dmg.get_expression('vy',       near=False)
U_msk = dmg.get_expression('mask',     near=True)

model = D3Model(mesh=mesh, out_dir=out_dir)
model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=M, U_mask=U_msk, adot=adot,
                           mark_divide=True, contour=contour)

model.save_xdmf(model.ff, 'ff')

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

lst = [model.S,
       model.B,
       model.sigma,
       model.mask,
       model.q_geo,
       model.T_surface,
       model.adot,
       model.u_ob,
       model.v_ob,
       model.U_mask]

f = HDF5File(mpi_comm_world(), out_dir + 'state.h5', 'w')

model.save_list_to_hdf5(lst, f)
model.save_subdomain_data(f)
model.save_mesh(f)

f.close()



