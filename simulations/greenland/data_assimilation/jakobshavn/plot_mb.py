from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
base_dir = 'dump/jakob_small/rstrt_FS_a_0_1_cont_crap/'
in_dir   = base_dir
out_dir  = base_dir + 'plot_mb/'
var_dir  = 'dump/vars_jakobshavn_small/'

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

#===============================================================================
# retrieve the bed mesh :
d3model.form_bed_mesh()

# create 2D model for balance velocity :
bedmodel = D2Model(d3model.bedmesh, out_dir)

#===============================================================================
# open the hdf5 file :
f     = HDF5File(mpi_comm_world(), in_dir  + 'Mb.h5',    'r')
fdata = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')

# initialize the variables :
d3model.init_Fb(f)
d3model.init_Mb(f)
d3model.init_PE(f)
d3model.init_p(f)

dp = Function(d3model.Q, name='dp')
d3model.assign_variable(dp, f)
dp_b = Function(bedmodel.Q, name='dp')

divU   = Function(d3model.Q, name='divU')
d3model.assign_variable(divU, f)
divU_b = Function(bedmodel.Q, name='divU')

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(bedmodel.Fb,   d3model.Fb)
bedmodel.assign_submesh_variable(bedmodel.Mb,   d3model.Mb)
bedmodel.assign_submesh_variable(bedmodel.PE,   d3model.PE)
bedmodel.assign_submesh_variable(bedmodel.p,    d3model.p)
bedmodel.assign_submesh_variable(bedmodel.p,    d3model.p)
bedmodel.assign_submesh_variable(dp_b,          dp)
bedmodel.assign_submesh_variable(divU_b,        divU)

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

Mbmax = bedmodel.Mb.vector().max()
Mbmin = bedmodel.Mb.vector().min()

Fbmax = bedmodel.Fb.vector().max()
Fbmin = bedmodel.Fb.vector().min()

Pmax  = bedmodel.PE.vector().max()
Pmin  = bedmodel.PE.vector().min()

dpmin = dp_b.vector().min()
dpmax = dp_b.vector().max()

divUmin = divU_b.vector().min()
divUmax = divU_b.vector().max()

dp_lvls = np.array([dpmin, -1e6, -1e5, -1e4, 1e4, 1e5, 1e6, dpmax])

divU_lvls = np.array([divUmin, -1, -1e-1, -1e-3, 1e-3, 1e-1, 1, divUmax])

#Mb_lvls = np.array([Mbmin, -0.5, -1e-1, -1e-2, -1e-5,
#                    1e-5, 1e-2, 1e-1, 0.5, Mbmax])
Mb_lvls = np.array([-7.2, -0.5, -1e-1, -1e-2, -1e-5,
                    1e-5, 1e-2, 1e-1, 0.5, 7.2])
Fb_lvls = np.array([0.0, 1e-5, 1e-2, 1e-1, 0.5, Fbmax])
Pe_lvls = np.array([1e2, 1e3, 5e3, 1e4, 2.5e4, 5e4, Pmax])


cmap = 'gist_yarg'

plotIce(drg, bedmodel.Mb, name='Mb', direc=out_dir, 
        title=r'$M_b$', cmap='RdGy',  scale='lin',
        levels=Mb_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.Fb, name='Fb', direc=out_dir, 
        title=r'$F_b$', cmap='RdGy',  scale='lin',
        levels=Mb_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.p, name='p', direc=out_dir, 
        title=r'$p$', cmap=cmap,  scale='lin',
        levels=None, umin=4.5e5, umax=2.7e7, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, dp_b, name='dp', direc=out_dir, 
        title=r'$\delta p$', cmap='RdGy',  scale='lin',
        levels=dp_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, divU_b, name='divU', direc=out_dir, 
        title=r'$\nabla \cdot \mathbf{u}$', cmap='RdGy',  scale='lin',
        levels=divU_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

#plotIce(drg, bedmodel.PE, name='PE', direc=out_dir, 
#        title=r'$P_e$', cmap=cmap,  scale='lin',
#        levels=Pe_lvls, tp=False, tpAlpha=0.2,
#        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
#        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)



