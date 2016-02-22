from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy as np
import sys

# set the relavent directories :
base_dir_1 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'
base_dir_2 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100' + \
             '_disc_kappa/tmc/10/'
in_dir_1 = base_dir_1
in_dir_2 = base_dir_2
out_dir  = 'dump/jakob_small/alpha_compare/'
var_dir  = 'dump/vars_jakobshavn_small/'

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

#===============================================================================
# retrieve the bed mesh :
bedmesh = d3model.get_bed_mesh()

# create 2D model for balance velocity :
bedmodel = D2Model(bedmesh, out_dir)

#===============================================================================
# open the hdf5 file :
f_1     = HDF5File(mpi_comm_world(), in_dir_1  + 'tmc.h5', 'r')
f_2     = HDF5File(mpi_comm_world(), in_dir_2  + 'tmc.h5', 'r')

alpha_1 = Function(d3model.Q, name='alpha')
alpha_2 = Function(d3model.Q, name='alpha')

alphab_1 = Function(bedmodel.Q)
alphab_2 = Function(bedmodel.Q)

a_d = Function(bedmodel.Q)

# initialize the variables :
d3model.assign_variable(alpha_1, f_1)
d3model.assign_variable(alpha_2, f_2)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(alphab_1, alpha_1)
bedmodel.assign_submesh_variable(alphab_2, alpha_2)

a_1_v  = alphab_1.vector().array()
a_2_v  = alphab_2.vector().array()
a_d_v  = a_2_v - a_1_v
a_d.vector().set_local(a_d_v)
a_d.vector().apply('insert')


#===============================================================================
drg  = DataFactory.get_rignot()

zoom_box_kwargs = {'zoom'             : 6,      # ammount to zoom 
                   'loc'              : 1,      # location of box
                   'loc1'             : 2,      # loc of first line
                   'loc2'             : 3,      # loc of second line
                   'x1'               : 51000,  # first x-coord
                   'y1'               : 80000,  # first y-coord
                   'x2'               : 88500,  # second x-coord
                   'y2'               : 102000, # second y-coord
                   'scale_font_color' : 'k',    # scale font color
                   'scale_length'     : 25,     # scale length in km
                   'scale_loc'        : 1,      # 1=top, 2=bottom
                   'plot_grid'        : True}   # plot the triangles

zoom_box_kwargs_2 = zoom_box_kwargs.copy()
zoom_box_kwargs_2['scale_font_color'] = 'w'

zoom_box_kwargs_3 = zoom_box_kwargs.copy()
zoom_box_kwargs_3['scale_font_color'] = 'r'

amax  = a_d.vector().max()
amin  = a_d.vector().min()

a_d_lvls  = np.array([-4, -0.5, -1e-1, -1e-2, 1e-2, 1e-1, 0.5, 4])

#===============================================================================
# plot :

#cmap = 'RdGy'
#cmap = 'viridis'
#cmap = 'inferno'
#cmap = 'plasma'
#cmap = 'magma'
cmap = 'gist_yarg'
  
plotIce(drg, a_d, name='delta_alpha', direc=out_dir, 
        title=r'$\alpha_d - \alpha_c$', cmap='RdGy',  scale='lin',
        levels=a_d_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)



