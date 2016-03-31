from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
base_dir = 'dump/jakob_small/02/'
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
d3model.init_tau_id(f)
d3model.init_tau_jd(f)
d3model.init_tau_ii(f)
d3model.init_tau_ij(f)
d3model.init_tau_iz(f)
d3model.init_tau_ji(f)
d3model.init_tau_jj(f)
d3model.init_tau_jz(f)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(bedmodel.tau_id, d3model.tau_id)
bedmodel.assign_submesh_variable(bedmodel.tau_jd, d3model.tau_jd)
bedmodel.assign_submesh_variable(bedmodel.tau_ii, d3model.tau_ii)
bedmodel.assign_submesh_variable(bedmodel.tau_ij, d3model.tau_ij)
bedmodel.assign_submesh_variable(bedmodel.tau_iz, d3model.tau_iz)
bedmodel.assign_submesh_variable(bedmodel.tau_ji, d3model.tau_ji)
bedmodel.assign_submesh_variable(bedmodel.tau_jj, d3model.tau_jj)
bedmodel.assign_submesh_variable(bedmodel.tau_jz, d3model.tau_jz)


#===============================================================================
drg  = DataFactory.get_rignot()

zoom_box_kwargs = {'zoom'             : 5.8,    # ammount to zoom 
                   'loc'              : 1,      # location of box
                   'loc1'             : 2,      # loc of first line
                   'loc2'             : 3,      # loc of second line
                   'x1'               : 40000,  # first x-coord
                   'y1'               : 80000,  # first y-coord
                   'x2'               : 77500,  # second x-coord
                   'y2'               : 102000, # second y-coord
                   'scale_font_color' : 'r',    # scale font color
                   'scale_length'     : 20,     # scale length in km
                   'scale_loc'        : 1,      # 1=top, 2=bottom
                   'plot_grid'        : True,   # plot the triangles
                   'axes_color'       : 'r'}    # color of axes

tau_ij_min = bedmodel.tau_ij.vector().min()
tau_ij_max = bedmodel.tau_ij.vector().max()

tau_ii_min = bedmodel.tau_ii.vector().min()
tau_ii_max = bedmodel.tau_ii.vector().max()

tau_iz_min = bedmodel.tau_iz.vector().min()
tau_iz_max = bedmodel.tau_iz.vector().max()

tau_id_min = bedmodel.tau_id.vector().min()
tau_id_max = bedmodel.tau_id.vector().max()

tau_ij_lvls = np.array([tau_ij_min, -2.5e5, -1e5, -1e4, -5e3, 
                        5e3, 1e4, 5e4, 1e5, tau_ij_max])
tau_ii_lvls = np.array([tau_ii_min, -2.5e5, -1e5, -1e4, -5e3, 
                        5e3, 1e4, 5e4, 1e5, tau_ii_max])
tau_iz_lvls = np.array([tau_iz_min, -2.5e5, -1e5, -1e4, -5e3, 
                        5e3, 1e4, 5e4, 1e5, tau_iz_max])
tau_id_lvls = np.array([tau_id_min, -2.5e5, -1e5, -1e4, -5e3, 
                        5e3, 1e4, 5e4, 1e5, tau_id_max])

#m = plotIce(drg, bedmodel.W_int, name='crap_to_delete', direc=out_dir, 
#            levels=Wi_lvls, tp=False, tpAlpha=0.2,
#            basin='jakobshavn', extend='neither', show=False, ext='.pdf')
#
#d3model.W.set_allow_extrapolation(True)
#d3model.T.set_allow_extrapolation(True)
#d3model.S.set_allow_extrapolation(True)
#d3model.B.set_allow_extrapolation(True)
#d3model.p.set_allow_extrapolation(True)
#d3model.theta.set_allow_extrapolation(True)
#
#gamma = d3model.gamma(0)
#Tw    = d3model.T_w(0)
#L     = d3model.L(0)
#a     = 146.3
#b     = 7.253
#
#x_w = 63550
#y_w = 89748
#    
#lon, lat  = m(x_w, y_w, inverse=True)
#x_w, y_w  = drg['pyproj_Proj'](lon, lat)
#
#zmin = mesh.coordinates()[:,2].min()
#zmax = mesh.coordinates()[:,2].max()
#
#z_a = arange(zmin, zmax, 1000)
#
#S = d3model.S(x_w, y_w, 1.0)
#B = d3model.B(x_w, y_w, 1.0)
#
#T_z     = []
#W_z     = []
#for z_w in z_a:
#  theta_i = d3model.theta(x_w, y_w, z_w)
#  p_i     = d3model.p(x_w, y_w, z_w)
#  Tm_i    = Tw - gamma*p_i
#  theta_m = a*Tm_i + b/2*Tm_i**2
#  if theta_i > theta_m:
#    W_z.append( (theta_i - theta_m)/L )
#    T_z.append( Tm_i )
#  else:
#    W_z.append( 0.0 )
#    T_z.append( (-a + np.sqrt(a**2 + 2*b*theta_i)) / b )
#
#T_z = array(T_z)
#W_z = array(W_z)
#
#z_n = []
#for z_w in z_a:
#  z_i = (z_w / zmax) * (S - B) - (S - B)
#  z_n.append(z_i)
#z_n = array(z_n)
#
#if not os.path.exists(base_dir + 'profile_data'):
#  os.makedirs(base_dir + 'profile_data')
#
#np.savetxt(base_dir + 'profile_data/T.txt', T_z)
#np.savetxt(base_dir + 'profile_data/W.txt', W_z)
#np.savetxt(base_dir + 'profile_data/z.txt', z_n)
#
#fig = figure(figsize=(4,4))
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
##ax2 = ax1.twiny()
#
#ax1.plot(T_z, z_n, 'k-', lw=3.0)
##plt.subplots_adjust(wspace = 0.001)
#ax2.plot(W_z, z_n, 'k-', lw=3.0)
#ax2.set_yticklabels([])
#
#ax2.set_xlim([0,0.15])
#
#xloc1 = plt.MaxNLocator(4)
#xloc2 = plt.MaxNLocator(4)
#ax1.xaxis.set_major_locator(xloc1)
#ax2.xaxis.set_major_locator(xloc2)
#
#ax1.set_xlabel(r'$T$')
#ax2.set_xlabel(r'$W$')
#ax1.set_ylabel(r'depth')
##ax2.tick_params(axis='x', colors='r')
##ax2.xaxis.label.set_color('r')
#ax1.grid()
#ax2.grid()
#plt.tight_layout()
#plt.savefig(out_dir + 'profile_plot.pdf')
#plt.close(fig)

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
        levels=tau_ii_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.tau_ij, name='tau_ij', direc=out_dir,
        title=r'$\tau_{ij}$', cmap='RdGy',  scale='lin',
        levels=tau_ij_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.tau_iz, name='tau_iz', direc=out_dir,
        title=r'$\tau_{iz}$', cmap='RdGy',  scale='lin',
        levels=tau_iz_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)

plotIce(drg, bedmodel.tau_id, name='tau_id', direc=out_dir,
        title=r'$\tau_{id}$', cmap='RdGy',  scale='lin',
        levels=tau_id_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs)



