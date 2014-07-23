import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities       import DataInput,DataOutput
from data.data_factory   import DataFactory
from meshes.mesh_factory import MeshFactory
from src.physics         import VelocityBalance_2
from dolfin              import Mesh, set_log_active

set_log_active(True)

thklim = 20.0

# collect the raw data :
searise  = DataFactory.get_searise()
measure  = DataFactory.get_gre_measures()
meas_shf = DataFactory.get_shift_gre_measures()
bamber   = DataFactory.get_bamber(thklim = thklim)

# load a mesh :
mesh    = Mesh("results/meshes/refined_mesh.xml")

# create data objects to use with varglas :
dsr     = DataInput(searise,  mesh=mesh)
dbm     = DataInput(bamber,   mesh=mesh)
dms     = DataInput(measure,  mesh=mesh)
dmss    = DataInput(meas_shf, mesh=mesh)

dms.change_projection(dsr)

H     = dbm.get_projection("H_n")
S     = dbm.get_projection("h_n")
adot  = dsr.get_projection("adot")

prb   = VelocityBalance_2(mesh, H, S, adot, 12.0)
prb.solve_forward()

# File ouput
do    = DataOutput('results/greenland_balance_velocity_v2/')

d_out = {'U_bal_mag' : prb.Ubmag,
         'H'         : prb.H,
         'adot'      : prb.adot,
         'S'         : prb.S,
         'slope'     : prb.slope,
         'residual'  : prb.residual}

do.write_dict_of_files(d_out)
do.write_dict_of_files(d_out, extension='.xml')
do.write_one_file('sp', dms.get_projection('sp'))

#do.write_matlab(dbm, prb.Ubmag, 'results/Ubmag.mat')

dbv = DataInput('results/', ('Ubmag.mat',), mesh=mesh)

dbv.set_data_min('Ubmag', 0.0, 0.0)

ass = dbv.integrate_field('sp', dmss, 'Ubmag', val=50)

do.write_one_file('Ubmag_measures',  ass)
#do.write_matlab(dbm, ass, 'results/Ubmag_measures.mat')



