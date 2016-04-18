from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.01/11_tik_1e-1/'
base_dir = 'dump/jakob_small/hdf5/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'
var_dir  = 'dump/vars_jakobshavn_small/'

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

#===============================================================================
# retrieve the bed mesh :
d3model.form_srf_mesh()

# create 2D model for balance velocity :
srfmodel = D2Model(d3model.srfmesh, out_dir)

#===============================================================================
# open the hdf5 file :
f     = HDF5File(mpi_comm_world(), in_dir  + 'inverted.h5',   'r')
fdata = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')
#fQ    = HDF5File(mpi_comm_world(), in_dir  + 'Q_int.h5', 'r')

# initialize the variables :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_T(f)

# 2D model gets balance-velocity appropriate variables initialized :
srfmodel.assign_submesh_variable(srfmodel.S,         d3model.S)
srfmodel.assign_submesh_variable(srfmodel.B,         d3model.B)
srfmodel.assign_submesh_variable(srfmodel.T,         d3model.T)
srfmodel.assign_submesh_variable(srfmodel.U_ob,      d3model.U_ob)


#===============================================================================
bamber  = DataFactory.get_bamber()
rignot  = DataFactory.get_rignot()

bamber['Bo'][bamber['Bo'] == -9999] = 0.0
bamber['S'][bamber['mask'] == 0] = 0.0
bamber['S'][bamber['S'] < 500] = 500.0

dbm   = DataInput(bamber,  gen_space=False)
drg   = DataInput(rignot,  gen_space=False)

dbm.change_projection(drg)

bc = '#880cbc'

lat_1 = 69.210
lat_2 = 69.168
lon_1 = -48.78
lon_2 = -48.759

dlat  = (lat_2 - lat_1) / 2.0
dlon  = (lon_2 - lon_1) / 2.0

lat_3 = lat_1 + dlat
lon_3 = lon_1 + dlon

lat_a   = [ 69.235,    lat_1, lat_2, lat_3]
lon_a   = [-48.686944, lon_1, lon_2, lon_3]
color_a = ['c', 'y', 'g',  bc]
style_a = ['o', 'p', '^', 's']

plot_pts = {'lat'   : lat_a,
            'lon'   : lon_a,
            'style' : style_a,
            'color' : color_a}

zoom_box_kwargs = {'zoom'             : 4,        # ammount to zoom 
                   'loc'              : 9,        # location of box
                   'loc1'             : 3,        # loc of first line
                   'loc2'             : 4,        # loc of second line
                   'llcrnrlon'        : -51,      # first x-coord
                   'llcrnrlat'        : 68.32,    # first y-coord
                   'urcrnrlon'        : -42.5,    # second x-coord
                   'urcrnrlat'        : 70.1,     # second y-coord
                   'plot_zoom_scale'  : False,    # draw the scale
                   'scale_font_color' : bc,       # scale font color
                   'scale_length'     : 20,       # scale length in km
                   'scale_loc'        : 1,        # 1=top, 2=bottom
                   'plot_grid'        : True,     # plot the triangles
                   'axes_color'       : bc,       # color of axes
                   'plot_points'      : None}     # dict of points to plot

box_params = {'llcrnrlat'    : 68.99,
              'urcrnrlat'    : 69.31,
              'llcrnrlon'    : -49.8,
              'urcrnrlon'    : -48.3,
              'color'        : 'r'}

params = {'llcrnrlon'    : -50.8,      # first x-coord
          'llcrnrlat'    : 68.32,      # first y-coord
          'urcrnrlon'    : -42,        # second x-coord
          'urcrnrlat'    : 70.1,       # second y-coord
          'scale_color'  : 'r',
          'scale_length' : 200,
          'scale_loc'    : 2,
          'figsize'      : (8,4),
          'lat_interval' : 0.5,
          'lon_interval' : 1.0,
          'plot_grid'    : True,
          'plot_scale'   : True}

cont_plot_params = {'width'  : 0.8,
                    'height' : 1.2,
                    'loc'    : 1}

Bmax  = srfmodel.B.vector().max()
Bmin  = srfmodel.B.vector().min()

Smax  = srfmodel.S.vector().max()
Smin  = srfmodel.S.vector().min()

Uobmax  = srfmodel.U_ob.vector().max()
Uobmin  = srfmodel.U_ob.vector().min()

B_lvls    = np.array([Bmin, -1e3, -5e2, -1e2, -1e1, 1e1, 1e2, 2e2, 3e2, Bmax])
S_lvls    = np.array([Smin, 10, 100, 500, 1000, 1500, 2000, 2500, 3000, Smax])
U_ob_lvls = np.array([Uobmin, 50, 100, 250, 500, 1000, 2500, Uobmax])
          
#===============================================================================
# plot :

#cmap = 'RdGy'
#cmap = 'viridis'
#cmap = 'inferno'
#cmap = 'plasma'
#cmap = 'magma'
cmap = 'gist_yarg'
  
#plotIce(drg, srfmodel.B, name='B', direc=out_dir,
#        title=r'$B$', cmap='RdGy',  scale='lin',
#        levels=B_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)
#
#plotIce(drg, srfmodel.B, name='B', direc=out_dir,
#        title=r'$B$', cmap='RdGy',  scale='lin',
#        levels=B_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        params=params, plot_continent=True, cont_plot_params=cont_plot_params)

plotIce(drg, srfmodel.U_ob, name='U_ob', direc=out_dir,
        title=r'$\Vert \mathbf{u}_{ob} \Vert$', cmap=cmap, scale='lin',
        levels=U_ob_lvls, tp=True, tpAlpha=0.4, box_params=box_params,
        extend='neither', show=False, ext='.pdf', cb_format="%i",
        params=params, plot_continent=True, cont_plot_params=cont_plot_params)

#plotIce(dbm, 'S', name='S', direc=out_dir, title='$S$', ext='.pdf',
#        contour_type='lines', cmap=cmap,  scale='lin', levels=S_lvls)



