from varglas.utilities              import DataInput,DataOutput
from varglas.data.data_factory      import DataFactory
from varglas.mesh.mesh_factory      import MeshFactory
from varglas.physics                import VelocityBalance_2
from fenics                         import Mesh, set_log_active
import os

set_log_active(True)

thklim = 10.0

# collect the raw data :
bm1 = DataFactory.get_bedmap1(thklim = thklim)
bm2 = DataFactory.get_bedmap2(thklim = thklim)

direc = os.path.dirname(os.path.realpath(__file__))

# load a mesh :
mesh  = MeshFactory.get_antarctica_2D_coarse()

# create data objects to use with varglas :
d1     = DataInput(bm1, mesh=mesh)
d3     = DataInput(bm2, mesh=mesh)

# get projections for use with FEniCS :
adot   = d1.get_projection("adot")
H      = d3.get_projection("H")
S      = d3.get_projection("S")
B      = d3.get_projection("B")

prb   = VelocityBalance_2(mesh, H, S, adot, 12.0)
prb.solve_forward()

# remove values of matrices with no data :
d3.set_data_val("S",    32767,  0.0)
d3.set_data_val("mask", 127,    1.0)

prb    = VelocityBalance_2(mesh, H, S, adot, 12.0)

do     = DataOutput('results/antarctia_balance_velocity/')

data_out = {'Ubmag'    : prb.Ubmag,
            'H'        : prb.H,
            'adot'     : prb.adot,
            'S'        : prb.S,
            'slope'    : prb.slope,
            'residual' : prb.residual}

do.write_dict_of_files(data_out)
do.write_dict_of_files(data_out,extension='.xml')

#plotIce(prb.Ubmag, direc, 'jet', scale='log', name='BVmag', units='m/a', 
#        proj_in='ant', numLvls=100, plot_type='tripcolor')



