from varglas.io         import DataInput
from varglas.data.data_factory import DataFactory
from fenics                     import File, Mesh, set_log_active, \
                                       VectorFunctionSpace, project, as_vector

"""
A simple, but useful script that projects a complete set of ice sheet modeling
variables onto a specified mesh. The variables can then be loaded efficiently as
.xml file, saving considerable time on model start up. Note that the .pvd files
corresponding to each can be written for inspection of the data on the mesh.
"""

# set the output directory :
output_path = 'Issun_2H/'
# set the mesh :
mesh = Mesh("meshes/Issun_2H.xml")
# write pvd files too?
PVD = True

set_log_active(True)

thklim = 10.0

# collect the raw data :
searise  = DataFactory.get_searise(thklim = thklim)
bamber   = DataFactory.get_bamber(thklim = thklim)
fm_qgeo  = DataFactory.get_gre_qgeo_fox_maule()
rignot   = DataFactory.get_gre_rignot()


# create data objects to use with varglas :
dsr     = DataInput(searise,  mesh=mesh)
dbm     = DataInput(bamber,   mesh=mesh)
dfm     = DataInput(fm_qgeo,  mesh=mesh)
drg     = DataInput(rignot,   mesh=mesh)

# change the projection of the measures data to fit with other data :
drg.change_projection(dbm)
dsr.change_projection(dbm)

# get the expressions used by varglas :
H      = dbm.get_interpolation('H')
Herr   = dbm.get_interpolation('Herr')
S      = dbm.get_interpolation('S')
B      = dbm.get_interpolation('B')
T_s    = dsr.get_interpolation('T')
dhdt   = dsr.get_interpolation('dhdt')
q_geo  = dfm.get_interpolation('q_geo')
adot   = dsr.get_interpolation('adot')
vx      = drg.get_interpolation('vx')
vy      = drg.get_interpolation('vy')
vmag   = drg.get_interpolation('U_ob')
verr   = drg.get_interpolation('v_err')

Q = VectorFunctionSpace(mesh,'CG',1,dim=2)
U = project(as_vector([vx,vy]),Q)
H_file = File(output_path+'H.xml')
Herr_file = File(output_path+'Herr.xml')
adot_file = File(output_path+'adot.xml')
dhdt_file = File(output_path+'dhdt.xml')
S_file = File(output_path+'S.xml')
Bed_file = File(output_path+'Bed.xml')
Ts_file = File(output_path+'Ts.xml')
Qgeo_file = File(output_path+'Qgeo.xml')
ubar_file = File(output_path+'vx.xml')
vbar_file = File(output_path+'vy.xml')
umag_file = File(output_path+'vmag.xml')
verr_file = File(output_path+'verr.xml')

# Output
H_file << H
Herr_file << Herr
adot_file << adot
dhdt_file << dhdt
S_file << S
Bed_file << B
Ts_file << T_s
Qgeo_file << q_geo
ubar_file << vx
vbar_file << vy
umag_file << vmag
verr_file << verr

if PVD:
    Hp_file = File(output_path+'H.pvd')
    Herrp_file = File(output_path+'Herr.pvd')
    adotp_file = File(output_path+'adot.pvd')
    dhdtp_file = File(output_path+'dhdt.pvd')
    Sp_file = File(output_path+'S.pvd')
    Bedp_file = File(output_path+'Bed.pvd')
    Tsp_file = File(output_path+'Ts.pvd')
    Qgeop_file = File(output_path+'Qgeo.pvd')
    ubarp_file = File(output_path+'vx.pvd')
    vbarp_file = File(output_path+'vy.pvd')
    umagp_file = File(output_path+'vmag.pvd')
    verrp_file = File(output_path+'verr.pvd')
    U_file = File(output_path+'U.pvd')


    U_file << U
    Hp_file << H
    Herrp_file << Herr
    adotp_file << adot
    dhdtp_file << dhdt
    Sp_file << S
    Bedp_file << B
    Tsp_file << T_s
    Qgeop_file << q_geo
    ubarp_file << vx
    vbarp_file << vy
    umagp_file << vmag
    verrp_file << verr
