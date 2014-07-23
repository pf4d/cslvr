from varglas.utilities              import DataInput,DataOutput
from varglas.data.data_factory      import DataFactory
from varglas.mesh.mesh_factory      import MeshFactory
from varglas.physics                import VelocityBalance_2
from fenics                         import Mesh, set_log_active
import os

set_log_active(True)

thklim = 10.0

# collect the raw data :
searise = DataFactory.get_searise(thklim = thklim)
rignot  = DataFactory.get_gre_rignot()
bamber  = DataFactory.get_bamber(thklim = thklim)

direc = os.path.dirname(os.path.realpath(__file__))

# load a mesh :
mesh  = MeshFactory.get_greenland_coarse()

# create data objects to use with varglas :
dsr   = DataInput(searise, mesh=mesh)
drg   = DataInput(rignot,  mesh=mesh)
dbm   = DataInput(bamber,   mesh=mesh)

# change the projection of the measures data to fit with other data :
drg.change_projection(dsr)

H     = dbm.get_projection("H")
S     = dbm.get_projection("S")
adot  = dsr.get_projection("adot")

prb   = VelocityBalance_2(mesh, H, S, adot, 12.0)
prb.solve_forward()

# File ouput
do    = DataOutput('results/greenland_balance_velocity_v2/')

d_out = {'Ubmag' : prb.Ubmag,
         'H'         : prb.H,
         'adot'      : prb.adot,
         'S'         : prb.S,
         'slope'     : prb.slope,
         'residual'  : prb.residual}

do.write_dict_of_files(d_out)
do.write_dict_of_files(d_out, extension='.xml')



