import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

from pylab                  import *
from dolfin                 import *
from src.utilities          import DataInput, DataOutput
from src.physics            import VelocityBalance_2
from plot.plothelp.plotting import plotIce
from data.data_factory      import DataFactory
from meshes.mesh_factory    import MeshFactory

set_log_level(PROGRESS)

bm1 = DataFactory.get_bedmap1()
bm2 = DataFactory.get_bedmap2()

direc = os.path.dirname(os.path.realpath(__file__)) 

mesh   = Mesh("../meshes/mesh.xml")

# Import data :
d1     = DataInput(None, bm1, mesh=mesh)
d3     = DataInput(None, bm2, mesh=mesh, flip=True)

# set minimum values for data :
d3.set_data_val("H",    32767, 10.0)
d3.set_data_val("h",    32767,  0.0)
d3.set_data_val("mask", 127,    1.0)

# get projections for use with FEniCS :
adot   = d1.get_projection("adot")
H      = d3.get_projection("H")
h      = d3.get_projection("h")
b      = d3.get_projection("b")


prb    = VelocityBalance_2(mesh,H,h,adot,12.0)

do     = DataOutput('results/antarctia_balance_velocity/')

data_out = {'Ubmag'    : prb.Ubmag,
            'H'        : prb.H,
            'adot'     : prb.adot,
            'S'        : prb.S,
            'slope'    : prb.slope,
            'residual' : prb.residual}

do.write_dict_of_files(data_out)
do.write_dict_of_files(data_out,extension='.xml')

plotIce(prb.Ubmag, direc, 'jet', scale='log', name='BVmag', units='m/a', 
        proj_in='ant', numLvls=100, plot_type='tripcolor')



