from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
#base_dir = 'dump/jakob_small/inversion_k_1e-3_FSTMC/10/'
base_dir = 'dump/jakob_small/tmc_cond_disc_vars/'
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
d3model.form_bed_mesh()
d3model.form_srf_mesh()

# create 2D model for balance velocity :
bedmodel = D2Model(d3model.bedmesh, out_dir)
srfmodel = D2Model(d3model.srfmesh, out_dir)

#===============================================================================
# open the hdf5 file :
f     = HDF5File(mpi_comm_world(), in_dir  + 'tmc.h5',   'r')
fdata = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')
fQ    = HDF5File(mpi_comm_world(), in_dir  + 'Q_int.h5', 'r')

# initialize the variables :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_T(f)
d3model.init_W(f)
d3model.init_Fb(f)
d3model.init_Mb(f)
d3model.init_alpha(f)
d3model.init_PE(f)
d3model.init_W_int(f)
d3model.init_U(f)
d3model.init_p(f)
d3model.init_beta(f)
d3model.init_theta(f)
d3model.init_Q_int(fQ)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(bedmodel.B,         d3model.B)
bedmodel.assign_submesh_variable(bedmodel.T,         d3model.T)
bedmodel.assign_submesh_variable(bedmodel.W,         d3model.W)
bedmodel.assign_submesh_variable(bedmodel.Fb,        d3model.Fb)
bedmodel.assign_submesh_variable(bedmodel.Mb,        d3model.Mb)
bedmodel.assign_submesh_variable(bedmodel.alpha,     d3model.alpha)
bedmodel.assign_submesh_variable(bedmodel.PE,        d3model.PE)
bedmodel.assign_submesh_variable(bedmodel.W_int,     d3model.W_int)
srfmodel.assign_submesh_variable(srfmodel.U_mag,     d3model.U_mag)
bedmodel.assign_submesh_variable(bedmodel.beta,      d3model.beta)
bedmodel.assign_submesh_variable(bedmodel.theta,     d3model.theta)
bedmodel.assign_submesh_variable(bedmodel.Q_int,     d3model.Q_int)


#===============================================================================
drg  = DataFactory.get_rignot()

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

Bmax  = bedmodel.B.vector().max()
Bmin  = bedmodel.B.vector().min()

amax  = bedmodel.alpha.vector().max()
amin  = bedmodel.alpha.vector().min()

Mbmax = bedmodel.Mb.vector().max()
Mbmin = bedmodel.Mb.vector().min()

Fbmax = bedmodel.Fb.vector().max()
Fbmin = bedmodel.Fb.vector().min()

Wmax  = bedmodel.W.vector().max()
Wmin  = bedmodel.W.vector().min()

Wimax  = bedmodel.W_int.vector().max()
Wimin  = bedmodel.W_int.vector().min()

betamax  = bedmodel.beta.vector().max()
betamin  = bedmodel.beta.vector().min()

Umax  = srfmodel.U_mag.vector().max()
Umin  = srfmodel.U_mag.vector().min()

Pmax  = bedmodel.PE.vector().max()
Pmin  = bedmodel.PE.vector().min()

Tmax  = bedmodel.T.vector().max()
Tmin  = bedmodel.T.vector().min()

Q_int_max = bedmodel.Q_int.vector().max()
Q_int_min = bedmodel.Q_int.vector().min()


B_lvls  = np.array([Bmin, -1e3, -5e2, -1e2, -1e1, 1e1, 1e2, 2e2, 3e2, Bmax])

a_lvls  = np.array([0.0, 1e-2, 1e-1, 0.5, amax])
Mb_lvls = np.array([Mbmin, -1e-2, -5e-3, -1e-3, -1e-4, -1e-5,
                    1e-5, 1e-2, 1e-1, 2e-1, 0.5, Mbmax])
#Mb_lvls = np.array([0.0, 1e-5, 1e-2, 1e-1, 0.5, Mbmax])
Fb_lvls = np.array([0.0, 1e-3, 1e-2, 1e-1, 0.25, 1.0, Fbmax])

#b_lvls  = np.array([betamin, 1, 1e2, 1e3, 2.5e3, 5e3, betamax])
b_lvls  = np.array([betamin, 1e-2, 1, 1e2, 2.5e2, 5e2, 1e3, betamax])

#W_lvls  = np.array([0.0, 1e-2, 3e-2, 4e-2, 5e-2, 1e-1, 2e-1, Wmax])
W_lvls  = np.array([0.0, 1e-3, 2.5e-2, 3e-2, 3.5e-2, Wmax])
U_lvls  = np.array([0.0, 500, 1e3, 1.5e3, 2e3, 4e3, 1e4, Umax])
Pe_lvls = np.array([1e2, 1e3, 5e3, 1e4, 2.5e4, 5e4, Pmax])
T_lvls  = np.array([Tmin, 268, 271.5, 272, 272.5, Tmax])

Wi_lvls = np.array([0.0, 1e-1, 1.0, 2.0, Wimax])
#Wi_lvls = np.array([0.0, 1e-1, 1.0, 2.5, 5.0, 10.0, 15.0, Wimax])

Q_int_lvls  = np.array([0.0, 1e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, Q_int_max])

gamma = d3model.gamma(0)
Tw    = d3model.T_w(0)
L     = d3model.L(0)
a     = 146.3
b     = 7.253

x_w = 63550
y_w = 89748
    
x_a, y_a  = drg['pyproj_Proj'](lon_a, lat_a)

zmin = mesh.coordinates()[:,2].min()
zmax = mesh.coordinates()[:,2].max()

z_s = linspace(zmin, zmax, 100)

T_a = []
W_a = []
z_a = []

for x_w, y_w in zip(x_a, y_a):
  S    = d3model.S(x_w, y_w, 1.0)
  B    = d3model.B(x_w, y_w, 1.0)
  T_z  = []
  W_z  = []
  for z_w in z_s:
    theta_i = d3model.theta(x_w, y_w, z_w)
    p_i     = d3model.p(x_w, y_w, z_w)
    Tm_i    = Tw - gamma*p_i
    theta_m = a*Tm_i + b/2*Tm_i**2
    if theta_i > theta_m:
      W_z.append( (theta_i - theta_m)/L )
      T_z.append( Tm_i )
    else:
      W_z.append( 0.0 )
      T_z.append( (-a + np.sqrt(a**2 + 2*b*theta_i)) / b )
  
  T_z = array(T_z)
  W_z = array(W_z)

  # get z-coordinates :  
  z_z = []
  for z_w in z_s:
    z_i = (z_w / zmax)# * (S - B) - (S - B)
    z_z.append(z_i)
  z_z = array(z_z)
  
  T_a.append(T_z)
  W_a.append(W_z)
  z_a.append(z_z)

T_a = array(T_a)
W_a = array(W_a)
z_a = array(z_a)

if not os.path.exists(base_dir + 'profile_data'):
  os.makedirs(base_dir + 'profile_data')

np.savetxt(base_dir + 'profile_data/T.txt', T_a)
np.savetxt(base_dir + 'profile_data/W.txt', W_a)
np.savetxt(base_dir + 'profile_data/z.txt', z_a)

fig = figure(figsize=(4,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#ax2 = ax1.twiny()

for T_i, W_i, z_i, color_i, style_i in zip(T_a, W_a, z_a, color_a, style_a):
  ax1.plot(T_i, z_i, color=color_i, lw=3.0)
  #plt.subplots_adjust(wspace = 0.001)
  ax2.plot(W_i, z_i, color=color_i, lw=3.0)
  ax2.set_yticklabels([])

#ax2.set_xlim([0,0.15])

xloc1 = plt.MaxNLocator(4)
xloc2 = plt.MaxNLocator(4)
ax1.xaxis.set_major_locator(xloc1)
ax2.xaxis.set_major_locator(xloc2)

ax1.set_xlabel(r'$T$')
ax2.set_xlabel(r'$W$')
ax1.set_ylabel(r'depth')
#ax2.tick_params(axis='x', colors='r')
#ax2.xaxis.label.set_color('r')
ax1.grid()
ax2.grid()
plt.tight_layout()
plt.savefig(out_dir + 'profile_plot.pdf')
plt.close(fig)

sys.exit(0)


#===============================================================================
# plot :

#cmap = 'RdGy'
#cmap = 'viridis'
#cmap = 'inferno'
#cmap = 'plasma'
#cmap = 'magma'
cmap = 'gist_yarg'
  
plotIce(drg, bedmodel.B, name='B', direc=out_dir,
        title=r'$B$', cmap='RdGy',  scale='lin',
        levels=B_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)
  
plotIce(drg, bedmodel.Q_int, name='Q_int', direc=out_dir,
        title=r'$Q_i$', cmap=cmap,  scale='lin',
        levels=Q_int_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)
  
plotIce(drg, bedmodel.T, name='T', direc=out_dir, 
        title='$T_B$', cmap=cmap,  scale='lin',
        levels=T_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.W, name='W', direc=out_dir, 
        title=r'$W_B$', cmap=cmap,  scale='lin',
        levels=W_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.Fb, name='Fb', direc=out_dir, 
        title=r'$F_b$', cmap=cmap,  scale='lin',
        levels=Fb_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.Mb, name='Mb', direc=out_dir, 
        title=r'$M_b$', cmap='RdGy',  scale='lin',
        levels=Mb_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.alpha, name='alpha', direc=out_dir, 
        title=r'$\alpha$', cmap=cmap,  scale='lin',
        levels=a_lvls, tp=True, tpAlpha=0.2, cb_format='%.2e',
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.PE, name='PE', direc=out_dir, 
        title=r'$P_e$', cmap=cmap,  scale='lin',
        levels=Pe_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.W_int, name='W_int', direc=out_dir, 
        title=r'$W_i$', cmap=cmap,  scale='lin',
        levels=Wi_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, srfmodel.U_mag, name='U_mag', direc=out_dir, 
        title=r'$\Vert \mathbf{u}_S \Vert$', cmap=cmap,  scale='lin',
        levels=U_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.theta, name='theta', direc=out_dir, 
        title=r'$\theta_B$', cmap=cmap,  scale='lin',
        tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)

plotIce(drg, bedmodel.beta, name='beta', direc=out_dir, 
        title=r'$\beta$', cmap=cmap,  scale='lin',
        levels=b_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
        params=params, plot_pts=plot_pts)



