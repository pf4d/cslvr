from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.01/10/'
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
f     = HDF5File(mpi_comm_world(), in_dir  + 'inverted_10.h5',  'r')
f2    = HDF5File(mpi_comm_world(), in_dir  + 'alpha_int.h5',    'r')
#f     = HDF5File(mpi_comm_world(), in_dir  + 'inverted.h5',  'r')
#f     = HDF5File(mpi_comm_world(), in_dir  + 'theta_opt.h5', 'r')
#f     = HDF5File(mpi_comm_world(), in_dir  + 'u_opt.h5',     'r')
fdata = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')

# initialize the variables :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_T(f)
d3model.init_W(f)
d3model.init_Fb(f)
d3model.init_Mb(f)
d3model.init_alpha(f)
d3model.init_PE(f)
d3model.init_U(f)
d3model.init_p(f)
d3model.init_beta(f)
d3model.init_theta(f)
d3model.init_alpha_int(f2)
d3model.init_temp_rat(f2)
d3model.init_Wbar(f2)
#d3model.init_Qbar(f)

#d3model.save_xdmf(d3model.theta, 'theta')
#d3model.save_xdmf(d3model.W, 'W')
#d3model.save_xdmf(d3model.U3, 'U')
#sys.exit(0)

u3,v3,w3 = d3model.U3.split(True)
u,v,w    = srfmodel.U3.split(True)
srfmodel.assign_submesh_variable(u, u3)
srfmodel.assign_submesh_variable(v, v3)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(bedmodel.alpha_int, d3model.alpha_int)
bedmodel.assign_submesh_variable(bedmodel.temp_rat,  d3model.temp_rat)
bedmodel.assign_submesh_variable(bedmodel.S,         d3model.S)
bedmodel.assign_submesh_variable(bedmodel.B,         d3model.B)
bedmodel.assign_submesh_variable(bedmodel.T,         d3model.T)
bedmodel.assign_submesh_variable(bedmodel.W,         d3model.W)
bedmodel.assign_submesh_variable(bedmodel.Wbar,      d3model.Wbar)
bedmodel.assign_submesh_variable(bedmodel.Fb,        d3model.Fb)
bedmodel.assign_submesh_variable(bedmodel.Mb,        d3model.Mb)
bedmodel.assign_submesh_variable(bedmodel.alpha,     d3model.alpha)
bedmodel.assign_submesh_variable(bedmodel.PE,        d3model.PE)
srfmodel.assign_submesh_variable(srfmodel.B,         d3model.B)
srfmodel.assign_submesh_variable(srfmodel.U_mag,     d3model.U_mag)
srfmodel.assign_submesh_variable(srfmodel.U_ob,      d3model.U_ob)
srfmodel.assign_submesh_variable(srfmodel.u_ob,      d3model.u_ob)
srfmodel.assign_submesh_variable(srfmodel.v_ob,      d3model.v_ob)
bedmodel.assign_submesh_variable(bedmodel.beta,      d3model.beta)
bedmodel.assign_submesh_variable(bedmodel.theta,     d3model.theta)
bedmodel.assign_submesh_variable(bedmodel.Qbar,     d3model.Qbar)


#===============================================================================
# calculate velocity misfit :
R = Function(srfmodel.Q)

u_v    = u.vector().array()
v_v    = v.vector().array()
u_ob_v = srfmodel.u_ob.vector().array()
v_ob_v = srfmodel.v_ob.vector().array()

U_v    = np.sqrt(u_v**2 + v_v**2 + DOLFIN_EPS)
U_ob_v = srfmodel.U_ob.vector().array()

#R_u = u_v - u_ob_v
#R_v = v_v - v_ob_v

R_v = U_v - U_ob_v

srfmodel.assign_variable(R, R_v)


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
          'plot_scale'   : False,
          'axes_color'   : 'r'}

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

Wbarmax  = bedmodel.Wbar.vector().max()
Wbarmin  = bedmodel.Wbar.vector().min()

betamax  = int(bedmodel.beta.vector().max())
betamin  = round(bedmodel.beta.vector().min(), 4)

Umax  = srfmodel.U_mag.vector().max()
Umin  = srfmodel.U_mag.vector().min()

Uobmax  = srfmodel.U_ob.vector().max()
Uobmin  = srfmodel.U_ob.vector().min()

Pmax  = bedmodel.PE.vector().max()
Pmin  = bedmodel.PE.vector().min()

Tmax  = bedmodel.T.vector().max()
Tmin  = bedmodel.T.vector().min()

Qbar_max = bedmodel.Qbar.vector().max()
Qbar_min = bedmodel.Qbar.vector().min()

a_int_max  = bedmodel.alpha_int.vector().max()
a_int_min  = bedmodel.alpha_int.vector().min()

R_max = R.vector().max()
R_min = R.vector().min()

R_lvls        = np.array([R_min, -750, -250, -50, -10, -1,
                          1, 10, 50, 250, 750, R_max])
#Wbar_lvls     = np.array([0.0, 1e-3, 2.5e-3, 5e-3, 1e-2, Wbarmax])
Wbar_lvls     = np.array([0.0, 1e-4, 1e-3, 2.5e-3, 5e-3, Wbarmax])
a_int_lvls    = np.array([0, 50, 100, 150, 200, 400, 800, a_int_max])
temp_rat_lvls = np.array([0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
B_lvls        = np.array([Bmin, -1e3, -5e2, -1e2, -1e1, 
                          1e1, 1e2, 2e2, 3e2, Bmax])
a_lvls        = np.array([0.0, 1e-2, 1e-1, 0.5, amax])
#Mb_lvls       = np.array([Mbmin, -1e-2, -5e-3, -1e-3, -1e-4, -1e-5,
#                        1e-5, 1e-2, 1e-1, 2e-1, 0.5, Mbmax])
Mb_lvls       = np.array([0.0, 1e-5, 1e-2, 1e-1, 2e-1, 5e-1, Mbmax])
#Fb_lvls       = np.array([0.0, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 6e-1, Fbmax])
Fb_lvls       = np.array([0.0, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 5e-1, Fbmax])
#b_lvls        = np.array([betamin, 1, 1e2, 1e3, 2.5e3, 5e3, betamax])
#b_lvls        = np.array([betamin, 1e-2, 1, 1e2, 2.5e2, 5e2, 1e3, betamax])
b_lvls        = np.array([betamin, 1e-2, 1e1, 2e2, 2.5e2,
                          3e2, 5e2, 1e3, betamax])
#W_lvls        = np.array([0.0, 1e-3, 2.5e-2, 3e-2, 3.5e-2, Wmax])
W_lvls        = np.array([0.0, 1e-4, 5e-3, 1e-2, 2e-2, 5e-2, Wmax])
U_lvls        = np.array([Umin, 500, 1e3, 1.5e3, 2e3, 4e3, 1e4, Umax])
U_ob_lvls     = np.array([Uobmin, 500, 1e3, 1.5e3, 2e3, 4e3, 1e4, Uobmax])
Pe_lvls       = np.array([1e2, 1e3, 5e3, 1e4, 2.5e4, 5e4, Pmax])
T_lvls        = np.array([Tmin, 268, 271.5, 272, 272.5, Tmax])
Qbar_lvls     = np.array([0.0, 1e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, Qbar_max])

gamma = d3model.gamma(0)
Tw    = d3model.T_w(0)
L     = d3model.L(0)
a     = 146.3
b     = 7.253

#===============================================================================
# correct temperate ratio :
temp_rat_v = bedmodel.temp_rat.vector()
temp_rat_v[temp_rat_v > 1.0] = 1.0

#===============================================================================
# plot profiles :

#x_a, y_a  = drg['pyproj_Proj'](lon_a, lat_a)
#
#zmin = mesh.coordinates()[:,2].min()
#zmax = mesh.coordinates()[:,2].max()
#
#z_s = linspace(zmin, zmax, 100)
#
#T_a = []
#W_a = []
#z_a = []
#
#for x_w, y_w in zip(x_a, y_a):
#  S    = d3model.S(x_w, y_w, 1.0)
#  B    = d3model.B(x_w, y_w, 1.0)
#  T_z  = []
#  W_z  = []
#  for z_w in z_s:
#    theta_i = d3model.theta(x_w, y_w, z_w)
#    p_i     = d3model.p(x_w, y_w, z_w)
#    Tm_i    = Tw - gamma*p_i
#    theta_m = a*Tm_i + b/2*Tm_i**2
#    if theta_i > theta_m:
#      W_z.append( (theta_i - theta_m)/L )
#      T_z.append( Tm_i )
#    else:
#      W_z.append( 0.0 )
#      T_z.append( (-a + np.sqrt(a**2 + 2*b*theta_i)) / b )
#  
#  T_z = array(T_z)
#  W_z = array(W_z)
#
#  # get z-coordinates :  
#  z_z = []
#  for z_w in z_s:
#    z_i = (z_w / zmax)# * (S - B) - (S - B)
#    z_z.append(z_i)
#  z_z = array(z_z)
#  
#  T_a.append(T_z)
#  W_a.append(W_z)
#  z_a.append(z_z)
#
#T_a = array(T_a)
#W_a = array(W_a)
#z_a = array(z_a)
#
#if not os.path.exists(base_dir + 'profile_data'):
#  os.makedirs(base_dir + 'profile_data')
#
#np.savetxt(base_dir + 'profile_data/T.txt', T_a)
#np.savetxt(base_dir + 'profile_data/W.txt', W_a)
#np.savetxt(base_dir + 'profile_data/z.txt', z_a)
#
#fig = figure(figsize=(4,4))
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
##ax2 = ax1.twiny()
#
#for T_i, W_i, z_i, color_i, style_i in zip(T_a, W_a, z_a, color_a, style_a):
#  ax1.plot(T_i, z_i, color=color_i, lw=3.0)
#  #plt.subplots_adjust(wspace = 0.001)
#  ax2.plot(W_i, z_i, color=color_i, lw=3.0)
#  ax2.set_yticklabels([])
#
##ax2.set_xlim([0,0.15])
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
# derive temperate zone thickness :

#d3model.theta.set_allow_extrapolation(True)
#d3model.p.set_allow_extrapolation(True)
#
#x_a = bedmodel.mesh.coordinates()[:,0]
#y_a = bedmodel.mesh.coordinates()[:,1]
#S_a = bedmodel.S.vector().array()
#B_a = bedmodel.B.vector().array()
#
#def line_intersection(line1, line2):
#  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
#  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
#
#  def det(a, b):
#    return a[0] * b[1] - a[1] * b[0]
#
#  div = det(xdiff, ydiff)
#  if div == 0:
#   raise Exception('lines do not intersect')
#
#  d = (det(*line1), det(*line2))
#  x = det(d, xdiff) / div
#  y = det(d, ydiff) / div
#  return x, y
#
#
#CTS = []
#for x_w, y_w, S_w, B_w in zip(x_a, y_a, S_a, B_a):
#
#  # get the energy values :
#  theta_i   = []
#  theta_m_i = []
#  for z_w in z_s:
#    theta_j   = d3model.theta(x_w, y_w, z_w)
#    p_j       = d3model.p(x_w, y_w, z_w)
#    Tm_j      = Tw - gamma*p_j
#    theta_m_j = a*Tm_j + b/2*Tm_j**2
#    theta_i.append(theta_j)
#    theta_m_i.append(theta_m_j)
#  theta_i   = array(theta_i)
#  theta_m_i = array(theta_m_i)
#  
#  # get z-coordinates :  
#  z_z = []
#  for z_w in z_s:
#    z_i = (z_w / zmax) * (S_w - B_w) + B_w
#    z_z.append(z_i)
#  z_z = array(z_z)
#
#  # get height of CTS :  
#  if sum(theta_i > theta_m_i) > 0:
#    temperate = where(theta_i > theta_m_i)[0]
#    if temperate.min() != 0:
#      CTS.append(0.0)
#    else:
#      CTS_idx_low   = temperate[-1]
#      CTS_idx_high  = CTS_idx_low + 1
#      theta_l   = ((theta_i[CTS_idx_low],    z_z[CTS_idx_low]),
#                   (theta_i[CTS_idx_high],   z_z[CTS_idx_high]))
#      theta_m_l = ((theta_m_i[CTS_idx_low],  z_z[CTS_idx_low]),
#                   (theta_m_i[CTS_idx_high], z_z[CTS_idx_high]))
#      P = line_intersection(theta_l, theta_m_l)
#      CTS.append(P[1] - B_w)
#  else:
#    CTS.append(0.0)
#CTS = array(CTS)
#
#CTS_f = Function(bedmodel.Q)
#bedmodel.assign_variable(CTS_f, CTS)
#
#CTS_max   = CTS_f.vector().max()
#CTS_min   = CTS_f.vector().min()
#CTS_lvls  = np.array([CTS_min, 1e-1, 1e0, 1e1, 1e2, CTS_max])
#
#plotIce(drg, CTS_f, name='CTS', direc=out_dir,
#        title=r'$CTS$', cmap='gist_yarg',  scale='lin',
#        levels=CTS_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None)
#
#
#sys.exit(0)

#===============================================================================
# plot :

u2_c    = 'r'#'#c1000e'
u2_lvls = [-500]

#plotIce(drg, R, name='misfit', direc=out_dir, 
#        title=r'$\Vert \mathbf{u} - \mathbf{u}_{ob} \Vert$',
#        cmap='RdGy',  scale='lin',
#        levels=R_lvls, tp=True, tpAlpha=0.2, cb_format='%i',
#        extend='neither', show=False, ext='.pdf',
#        params=params, plot_pts=None,
#        u2=srfmodel.B, u2_levels=u2_lvls, u2_color=bc)
#
#plotIce(drg, bedmodel.alpha_int, name='alpha_int', direc=out_dir, 
#        title=r'$\alpha_i$', cmap='gist_yarg',  scale='lin',
#        levels=a_int_lvls, tp=True, tpAlpha=0.2, cb_format='%i',
#        extend='neither', show=False, ext='.pdf',
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)

plotIce(drg, bedmodel.temp_rat, name='temp_rat', direc=out_dir, 
        title=r'$\alpha_i / H$', cmap='gist_yarg',  scale='lin',
        levels=temp_rat_lvls, tp=True, tpAlpha=0.2, cb_format='%.2f',
        extend='neither', show=False, ext='.pdf',
        params=params, plot_pts=None,
        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
  
#plotIce(drg, bedmodel.Qbar, name='Qbar', direc=out_dir,
#        title=r'$\bar{Q}$', cmap='gist_yarg',  scale='lin',
#        levels=Qbar_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
  
#plotIce(drg, bedmodel.Wbar, name='Wbar', direc=out_dir, 
#        title=r'$\bar{W}$', cmap='gist_yarg',  scale='lin',
#        levels=Wbar_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, bedmodel.W, name='W', direc=out_dir, 
#        title=r'$W_B$', cmap='gist_yarg',  scale='lin',
#        levels=W_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, bedmodel.T, name='T', direc=out_dir, 
#        title='$T_B$', cmap='gist_yarg',  scale='lin',
#        levels=T_lvls, tp=True, tpAlpha=0.2, cb_format='%.1f',
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#
#plotIce(drg, bedmodel.Fb, name='Fb', direc=out_dir, 
#        title=r'$F_b$', cmap='gist_yarg',  scale='lin',
#        levels=Fb_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, bedmodel.Mb, name='Mb', direc=out_dir, 
#        title=r'$M_b$', cmap='gist_yarg',  scale='lin',
#        levels=Mb_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, bedmodel.alpha, name='alpha', direc=out_dir, 
#        title=r'$\alpha$', cmap='gist_yarg',  scale='lin',
#        levels=a_lvls, tp=True, tpAlpha=0.2, cb_format='%.2e',
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
##plotIce(drg, bedmodel.PE, name='PE', direc=out_dir, 
##        title=r'$P_e$', cmap='gist_yarg',  scale='lin',
##        levels=Pe_lvls, tp=True, tpAlpha=0.2,
##        extend='neither', show=False, ext='.pdf',
##        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
##        params=params, plot_pts=None,
##        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, srfmodel.U_ob, name='U_ob', direc=out_dir, 
#        title=r'$\Vert \mathbf{u}_{ob} \Vert$', cmap='gist_yarg',  scale='lin',
#        levels=U_ob_lvls, tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf', cb_format='%i',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=srfmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, srfmodel.U_mag, name='U_mag', direc=out_dir, 
#        title=r'$\Vert \mathbf{u}_S \Vert$', cmap='gist_yarg',  scale='lin',
#        levels=U_lvls, tp=True, tpAlpha=0.2, cb_format='%i',
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=srfmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, bedmodel.theta, name='theta', direc=out_dir, 
#        title=r'$\theta_B$', cmap='gist_yarg',  scale='lin',
#        tp=True, tpAlpha=0.2,
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)
#
#plotIce(drg, bedmodel.beta, name='beta', direc=out_dir, 
#        title=r'$\beta$', cmap='gist_yarg',  scale='lin',
#        levels=b_lvls, tp=True, tpAlpha=0.2, cb_format='%g',
#        extend='neither', show=False, ext='.pdf',
#        zoom_box=False, zoom_box_kwargs=zoom_box_kwargs,
#        params=params, plot_pts=None,
#        u2=bedmodel.B, u2_levels=u2_lvls, u2_color=u2_c)



