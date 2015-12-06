from varglas import *
from fenics  import *
from pylab   import *

out_dir  = 'dump/vars_jakobshavn_basin/'

# collect the raw data :
searise  = DataFactory.get_searise()
bamber   = DataFactory.get_bamber(1.0)
rignot   = DataFactory.get_rignot()

# define the mesh :
mesh = Mesh('dump/meshes/jakobshavn_3D_U_mesh_basin.xml.gz')

# create data objects to use with varglas :
dsr     = DataInput(searise,  mesh=mesh)
dbm     = DataInput(bamber,   mesh=mesh)
drg     = DataInput(rignot,   mesh=mesh)

# change the projection of all data to be the same as the mesh :
dbm.change_projection(drg)
dsr.change_projection(drg)
#drg.change_projection(dbm)

# get the expressions used by varglas :
S     = dbm.get_expression('S',        near=False)
B     = dbm.get_expression('B',        near=False)
M     = dbm.get_expression('mask',     near=True)
L     = dbm.get_expression('lat_mask', near=True)
adot  = dsr.get_expression('adot',     near=False)
q_geo = dsr.get_expression('q_geo',    near=False)
T_s   = dsr.get_expression('T',        near=False)
u_ob  = drg.get_expression('vx',       near=False)
v_ob  = drg.get_expression('vy',       near=False)
U_msk = drg.get_expression('mask',     near=True)

model = D3Model(mesh=mesh, out_dir=out_dir, save_state=True)
model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=M, lat_mask=L, U_mask=U_msk, adot=adot, 
                           mark_divide=True)

model.init_T_surface(T_s)
model.init_q_geo(q_geo)
model.init_U_ob(u_ob, v_ob)

model.state.close()



