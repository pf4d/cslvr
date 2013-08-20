import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput,DataOutput
from data.data_factory import DataFactory
from src.physics       import VelocityBalance_2
from dolfin            import Mesh, set_log_active

set_log_active(True)

# collect the raw data :
searise  = DataFactory.get_searise()
measure  = DataFactory.get_gre_measures()
meas_shf = DataFactory.get_shift_gre_measures()
v2       = DataFactory.get_V2()

# load a mesh :
mesh    = Mesh("../meshes/coarse_mesh.xml")

# create data objects to use with varglas :
dsr     = DataInput(None, searise,  mesh=mesh, create_proj=True)
dv2     = DataInput(None, v2,       mesh=mesh)
dms     = DataInput(None, measure,  mesh=mesh, create_proj=True, flip=True)
dmss    = DataInput(None, meas_shf, mesh=mesh, flip=True)

dms.change_projection(dsr)

#dv2.set_data_min('H', 10.0, 10.0)

#H     = dv2.get_projection("H")
#S     = dv2.get_projection("h")
#adot  = dsr.get_projection("adot")

#prb   = VelocityBalance_2(mesh, H, S, adot, 12.0)

# File ouput
do    = DataOutput('results/greenland_balance_velocity_v2/')

#d_out = {'U_bal_mag' : prb.Ubmag,
#         'H'         : prb.H,
#         'adot'      : prb.adot,
#         'S'         : prb.S,
#         'slope'     : prb.slope,
#         'residual'  : prb.residual}

#do.write_dict_of_files(d_out)
#do.write_dict_of_files(d_out, extension='.xml')
do.write_one_file('sp', dms.get_projection('sp'))

#do.write_matlab(dv2, prb.Ubmag, 'results/Ubmag.mat')

dbv = DataInput('results/', ('Ubmag.mat',), mesh=mesh)

dbv.set_data_min('Ubmag', 0.0, 0.0)

ass = dbv.integrate_field('sp', dmss, 'Ubmag', val=50)

do.write_one_file('Ubmag_measures',  ass)
#do.write_matlab(dv2, ass, 'results/Ubmag_measures.mat')



