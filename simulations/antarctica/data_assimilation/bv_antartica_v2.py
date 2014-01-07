import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities       import DataInput,DataOutput
from data.data_factory   import DataFactory
from meshes.mesh_factory import MeshFactory
from src.physics         import VelocityBalance_2
from dolfin              import Mesh, set_log_active

set_log_active(True)

thklim = 50.0

bedmap1 = DataFactory.get_bedmap1(thklim=thklim)
bedmap2 = DataFactory.get_bedmap2(thklim=thklim)

# load a mesh :
mesh = Mesh("meshes/2dmesh.xml")

db1  = DataInput(None, bedmap1, mesh=mesh)
db2  = DataInput(None, bedmap2, mesh=mesh)

h    = db2.get_projection("h_n")
H    = db2.get_projection("H_n")
adot = db1.get_projection("adot")

prb   = VelocityBalance_2(mesh, H, h, adot, 12.0)
prb.solve_forward()

# File ouput
do    = DataOutput('results/antartica_bv/')

d_out = {'U_bal_mag' : prb.Ubmag,
         'H'         : prb.H,
         'adot'      : prb.adot,
         'S'         : prb.S,
         'slope'     : prb.slope,
         'residual'  : prb.residual}

do.write_dict_of_files(d_out)
do.write_dict_of_files(d_out, extension='.xml')

do.write_matlab(db2, prb.Ubmag, 'results/Ubmag.mat')



