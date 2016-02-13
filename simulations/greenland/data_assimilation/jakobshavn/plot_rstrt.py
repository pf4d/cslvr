from cslvr    import *
from fenics   import *
import sys

# set the relavent directories :
#base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized/'
#base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_a_0_1/'
#base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_a_0_1/'
base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_1/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

#===============================================================================
# retrieve the bed mesh :
bedmesh = d3model.get_bed_mesh()
srfmesh = d3model.get_srf_mesh()

# create 2D model for balance velocity :
bedmodel = D2Model(bedmesh, out_dir)
srfmodel = D2Model(srfmesh, out_dir)

#===============================================================================
# open the hdf5 file :
f     = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5', 'r')

# initialize the variables :
d3model.init_T(f)
d3model.init_W(f)
d3model.init_Wb_flux(f)
d3model.init_Mb(f)
d3model.init_alpha(f)
d3model.init_PE(f)
d3model.init_W_int(f)
d3model.init_U(f)
d3model.init_beta(f)
d3model.init_theta(f)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(bedmodel.T,         d3model.T)
bedmodel.assign_submesh_variable(bedmodel.W,         d3model.W)
bedmodel.assign_submesh_variable(bedmodel.Wb_flux,   d3model.Wb_flux)
bedmodel.assign_submesh_variable(bedmodel.Mb,        d3model.Mb)
bedmodel.assign_submesh_variable(bedmodel.alpha,     d3model.alpha)
bedmodel.assign_submesh_variable(bedmodel.PE,        d3model.PE)
bedmodel.assign_submesh_variable(bedmodel.W_int,     d3model.W_int)
srfmodel.assign_submesh_variable(srfmodel.U_mag,     d3model.U_mag)
bedmodel.assign_submesh_variable(bedmodel.beta,      d3model.beta)
bedmodel.assign_submesh_variable(bedmodel.theta,     d3model.theta)

#===============================================================================
# collect the raw data :
drg  = DataFactory.get_rignot()

#cmap = 'RdGy'
cmap = 'viridis'
#cmap = 'inferno'
#cmap = 'plasma'
#cmap = 'magma'
  
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

#===============================================================================
# plot :

plotIce(drg, bedmodel.T, name='T', direc=out_dir, 
        title='$T_B$', cmap=cmap,  scale='lin',
        umin=265, umax=None, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.W, name='W', direc=out_dir, 
        title=r'$W_B$', cmap=cmap,  scale='lin',
        umin=None, umax=0.1, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.Wb_flux, name='Wb_flux', direc=out_dir, 
        title=r'$W_b$', cmap=cmap,  scale='log',
        umin=1e-2, umax=5, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.Mb, name='Mb', direc=out_dir, 
        title=r'$M_b$', cmap=cmap,  scale='log',
        umin=1e-2, umax=5, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.alpha, name='alpha', direc=out_dir, 
        title=r'$\alpha$', cmap=cmap,  scale='lin',
        umin=0, umax=1, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.PE, name='PE', direc=out_dir, 
        title=r'$P_e$', cmap=cmap,  scale='log',
        umin=1e3, umax=1e5, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.W_int, name='W_int', direc=out_dir, 
        title=r'$W_i$', cmap=cmap,  scale='log',
        umin=1e-1, umax=20, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, srfmodel.U_mag, name='U_mag', direc=out_dir, 
        title=r'$\Vert \mathbf{u}_S \Vert$', cmap=cmap,  scale='log',
        umin=5e1, umax=1e4, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.beta, name='beta', direc=out_dir, 
        title=r'$\beta$', cmap=cmap,  scale='log',
        umin=2e-4, umax=2e4, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.theta, name='theta', direc=out_dir, 
        title=r'$\theta_B$', cmap=cmap,  scale='lin',
        umin=3e5, umax=None, numLvls=13, tp=False, tpAlpha=0.5,
        basin='jakobshavn', extend='neither', show=False, ext='.png', res=200,
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)



