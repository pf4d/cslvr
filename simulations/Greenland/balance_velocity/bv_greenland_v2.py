import sys
src_directory = '../../../'
sys.path.append(src_directory)

from src.utilities     import DataInput,DataOutput
from data.data_factory import DataFactory
from src.physics       import VelocityBalance_2
from dolfin            import Mesh, set_log_active
from plot.plothelp.plotting import plotIce
import os

set_log_active(True)

# collect the raw data :
searise = DataFactory.get_searise()
measure = DataFactory.get_gre_measures()
#bamber  = DataFactory.get_bamber()

direc = os.path.dirname(os.path.realpath(__file__))

# load a mesh :
mesh    = Mesh("../meshes/mesh.xml")

# create data objects to use with varglas :
dsr     = DataInput(None, searise, mesh=mesh, create_proj=True)
#dbm     = DataInput(None, bamber,  mesh=mesh)
dms     = DataInput(None, measure, mesh=mesh, create_proj=True, flip=True)

dms.change_projection(dsr)

dsr.set_data_min('H', 10.0, 10.0)

H     = dsr.get_projection("H")
S     = dsr.get_projection("h")
adot  = dsr.get_projection("adot")

prb   = VelocityBalance_2(mesh, H, S, adot, 12.0)

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

#Plotting not quite working with Greenland.
#plotIce(prb.Ubmag, direc, 'jet', scale='log', name='BVmag', units='m/a', 
#        numLvls=100, plot_type='tripcolor')




