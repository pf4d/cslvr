from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.01/11_tik_1e-1/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

#===============================================================================
# retrieve the bed mesh :
d3model.form_bed_mesh()
d3model.form_srf_mesh()

# create 2D model for balance velocity :
bedmodel = D2Model(d3model.bedmesh, out_dir)
srfmodel = D2Model(d3model.srfmesh, out_dir)

#===============================================================================
# open the hdf5 file :
f     = HDF5File(mpi_comm_world(), in_dir  + 'stress.h5', 'r')

# initialize the variables :
d3model.init_tau_ii(f)
d3model.init_tau_ij(f)
d3model.init_tau_ik(f)
d3model.init_tau_ji(f)
d3model.init_tau_jj(f)
d3model.init_tau_jk(f)
d3model.init_tau_ki(f)
d3model.init_tau_kj(f)
d3model.init_tau_kk(f)
d3model.init_N_ii(f)
d3model.init_N_ij(f)
d3model.init_N_ik(f)
d3model.init_N_ji(f)
d3model.init_N_jj(f)
d3model.init_N_jk(f)
d3model.init_N_ki(f)
d3model.init_N_kj(f)
d3model.init_N_kk(f)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(bedmodel.tau_ii, d3model.tau_ii)
bedmodel.assign_submesh_variable(bedmodel.tau_ij, d3model.tau_ij)
bedmodel.assign_submesh_variable(bedmodel.tau_ik, d3model.tau_ik)
bedmodel.assign_submesh_variable(bedmodel.tau_ji, d3model.tau_ji)
bedmodel.assign_submesh_variable(bedmodel.tau_jj, d3model.tau_jj)
bedmodel.assign_submesh_variable(bedmodel.tau_jk, d3model.tau_jk)
bedmodel.assign_submesh_variable(bedmodel.tau_ki, d3model.tau_ki)
bedmodel.assign_submesh_variable(bedmodel.tau_kj, d3model.tau_kj)
bedmodel.assign_submesh_variable(bedmodel.tau_kk, d3model.tau_kk)
bedmodel.assign_submesh_variable(bedmodel.N_ii, d3model.N_ii)
bedmodel.assign_submesh_variable(bedmodel.N_ij, d3model.N_ij)
bedmodel.assign_submesh_variable(bedmodel.N_ik, d3model.N_ik)
bedmodel.assign_submesh_variable(bedmodel.N_ji, d3model.N_ji)
bedmodel.assign_submesh_variable(bedmodel.N_jj, d3model.N_jj)
bedmodel.assign_submesh_variable(bedmodel.N_jk, d3model.N_jk)
bedmodel.assign_submesh_variable(bedmodel.N_ki, d3model.N_ki)
bedmodel.assign_submesh_variable(bedmodel.N_kj, d3model.N_kj)
bedmodel.assign_submesh_variable(bedmodel.N_kk, d3model.N_kk)

#===============================================================================
drg  = DataFactory.get_rignot()

bc      = '#880cbc'

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

zoom_box_kwargs = {'zoom'             : 5.8,      # ammount to zoom 
                   'loc'              : 1,        # location of box
                   'loc1'             : 2,        # loc of first line
                   'loc2'             : 3,        # loc of second line
                   'x1'               : 40000,    # first x-coord
                   'y1'               : 80000,    # first y-coord
                   'x2'               : 90000,    # second x-coord
                   'y2'               : 105000,   # second y-coord
                   'scale_font_color' : bc,       # scale font color
                   'scale_length'     : 20,       # scale length in km
                   'scale_loc'        : 1,        # 1=top, 2=bottom
                   'plot_grid'        : True,     # plot the triangles
                   'axes_color'       : bc,       # color of axes
                   'plot_points'      : plot_pts} # dict of points to plot

params = {'llcrnrlat'    : 68.99,
          'urcrnrlat'    : 69.31,
          'llcrnrlon'    : -49.8,
          'urcrnrlon'    : -48.3,
          'scale_color'  : bc,
          'scale_length' : 50,
          'scale_loc'    : 1,
          'figsize'      : (7,4),
          'lat_interval' : 0.05,
          'lon_interval' : 0.25,
          'plot_grid'    : False,
          'plot_scale'   : False}


tau_ii_min = bedmodel.tau_ii.vector().min()
tau_ii_max = bedmodel.tau_ii.vector().max()

tau_ij_min = bedmodel.tau_ij.vector().min()
tau_ij_max = bedmodel.tau_ij.vector().max()

tau_ik_min = bedmodel.tau_ik.vector().min()
tau_ik_max = bedmodel.tau_ik.vector().max()

tau_ji_min = bedmodel.tau_ji.vector().min()
tau_ji_max = bedmodel.tau_ji.vector().max()

tau_jj_min = bedmodel.tau_jj.vector().min()
tau_jj_max = bedmodel.tau_jj.vector().max()

tau_jk_min = bedmodel.tau_jk.vector().min()
tau_jk_max = bedmodel.tau_jk.vector().max()

tau_ki_min = bedmodel.tau_ki.vector().min()
tau_ki_max = bedmodel.tau_ki.vector().max()

tau_kj_min = bedmodel.tau_kj.vector().min()
tau_kj_max = bedmodel.tau_kj.vector().max()

tau_kk_min = bedmodel.tau_kk.vector().min()
tau_kk_max = bedmodel.tau_kk.vector().max()


N_ii_min = bedmodel.N_ii.vector().min()
N_ii_max = bedmodel.N_ii.vector().max()

N_ij_min = bedmodel.N_ij.vector().min()
N_ij_max = bedmodel.N_ij.vector().max()

N_ik_min = bedmodel.N_ik.vector().min()
N_ik_max = bedmodel.N_ik.vector().max()

N_ji_min = bedmodel.N_ji.vector().min()
N_ji_max = bedmodel.N_ji.vector().max()

N_jj_min = bedmodel.N_jj.vector().min()
N_jj_max = bedmodel.N_jj.vector().max()

N_jk_min = bedmodel.N_jk.vector().min()
N_jk_max = bedmodel.N_jk.vector().max()

N_ki_min = bedmodel.N_ki.vector().min()
N_ki_max = bedmodel.N_ki.vector().max()

N_kj_min = bedmodel.N_kj.vector().min()
N_kj_max = bedmodel.N_kj.vector().max()

N_kk_min = bedmodel.N_kk.vector().min()
N_kk_max = bedmodel.N_kk.vector().max()



tau_ii_lvls = np.array([tau_ii_min, -8e4, -4e4, -2e4, -5e3, 
                        5e3, 2e4, 4e4, 8e4, tau_ii_max])
tau_ij_lvls = np.array([tau_ij_min, -1.5e5, -5e4, -2.5e4, -5e3, 
                        5e3, 2.5e4, 5e4, 1.5e5, tau_ij_max])
tau_ik_lvls = np.array([tau_ik_min, -8e4, -4e4, -2e4, -5e3, 
                        5e3, 2e4, 4e4, 8e4, tau_ik_max])

tau_ji_lvls = np.array([tau_ji_min, -8e4, -4e4, -2e4, -5e3, 
                        5e3, 2e4, 4e4, 8e4, tau_ji_max])
tau_jj_lvls = np.array([tau_jj_min, -1.5e5, -5e4, -2.5e4, -5e3, 
                        5e3, 2.5e4, 5e4, 1.5e5, tau_jj_max])
tau_jk_lvls = np.array([tau_jk_min, -2e5, -1e5, -5e4, -5e3, 
                        5e3, 5e4, 1e5, 2e5, tau_jk_max])

tau_ki_lvls = np.array([tau_ki_min, -8e4, -4e4, -2e4, -5e3, 
                        5e3, 2e4, 4e4, 8e4, tau_ki_max])
tau_kj_lvls = np.array([tau_kj_min, -1.5e5, -4e4, -2e4, -5e3, 
                        5e3, 2e4, 4e4, 8e4, tau_kj_max])
tau_kk_lvls = np.array([tau_kk_min, -2e5, -1e5, -5e4, -5e3, 
                        5e3, 5e4, 1e5, 2e5, tau_kk_max])



N_ii_lvls = np.array([N_ii_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_ii_max])
N_ij_lvls = np.array([N_ij_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_ij_max])
N_ik_lvls = np.array([N_ik_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_ik_max])

N_ji_lvls = np.array([N_ji_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_ji_max])
N_jj_lvls = np.array([N_jj_min, -1.5e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_jj_max])
N_jk_lvls = np.array([N_jk_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_jk_max])

N_ki_lvls = np.array([N_ki_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_ki_max])
N_kj_lvls = np.array([N_kj_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_kj_max])
N_kk_lvls = np.array([N_kk_min, -2e8, -1e8, -5e7, -1e7, 
                        1e7, 5e7, 1e8, 2e8, N_kk_max])


#===============================================================================
# plot :

#cmap = 'RdGy'
#cmap = 'viridis'
#cmap = 'inferno'
#cmap = 'plasma'
#cmap = 'magma'
cmap = 'gist_yarg'
  

plotIce(drg, bedmodel.tau_ii, name='tau_ii', direc=out_dir,
        title=r'$\tau_{ii}$', cmap='RdGy',  scale='lin',
        levels=tau_ii_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.tau_ij, name='tau_ij', direc=out_dir,
        title=r'$\tau_{ij}$', cmap='RdGy',  scale='lin',
        levels=tau_ij_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.tau_ik, name='tau_ik', direc=out_dir,
        title=r'$\tau_{ik}$', cmap='RdGy',  scale='lin',
        levels=tau_ik_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

  
plotIce(drg, bedmodel.tau_ji, name='tau_ji', direc=out_dir,
        title=r'$\tau_{ji}$', cmap='RdGy',  scale='lin',
        levels=tau_ji_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.tau_jj, name='tau_jj', direc=out_dir,
        title=r'$\tau_{jj}$', cmap='RdGy',  scale='lin',
        levels=tau_jj_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.tau_jk, name='tau_jk', direc=out_dir,
        title=r'$\tau_{jk}$', cmap='RdGy',  scale='lin',
        levels=tau_jk_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

  
plotIce(drg, bedmodel.tau_ki, name='tau_ki', direc=out_dir,
        title=r'$\tau_{ki}$', cmap='RdGy',  scale='lin',
        levels=tau_ki_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.tau_kj, name='tau_kj', direc=out_dir,
        title=r'$\tau_{kj}$', cmap='RdGy',  scale='lin',
        levels=tau_kj_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.tau_kk, name='tau_kk', direc=out_dir,
        title=r'$\tau_{kk}$', cmap='RdGy',  scale='lin',
        levels=tau_kk_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)
 

  
plotIce(drg, bedmodel.N_ii, name='N_ii', direc=out_dir,
        title=r'$N_{ii}$', cmap='RdGy',  scale='lin',
        levels=N_ii_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.N_ij, name='N_ij', direc=out_dir,
        title=r'$N_{ij}$', cmap='RdGy',  scale='lin',
        levels=N_ij_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.N_ik, name='N_ik', direc=out_dir,
        title=r'$N_{ik}$', cmap='RdGy',  scale='lin',
        levels=N_ik_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

  
plotIce(drg, bedmodel.N_ji, name='N_ji', direc=out_dir,
        title=r'$N_{ji}$', cmap='RdGy',  scale='lin',
        levels=N_ji_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.N_jj, name='N_jj', direc=out_dir,
        title=r'$N_{jj}$', cmap='RdGy',  scale='lin',
        levels=N_jj_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.N_jk, name='N_jk', direc=out_dir,
        title=r'$N_{jk}$', cmap='RdGy',  scale='lin',
        levels=N_jk_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

  
plotIce(drg, bedmodel.N_ki, name='N_ki', direc=out_dir,
        title=r'$N_{ki}$', cmap='RdGy',  scale='lin',
        levels=N_ii_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.N_kj, name='N_kj', direc=out_dir,
        title=r'$N_{kj}$', cmap='RdGy',  scale='lin',
        levels=N_ij_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.N_kk, name='N_kk', direc=out_dir,
        title=r'$N_{kk}$', cmap='RdGy',  scale='lin',
        levels=N_ik_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)



