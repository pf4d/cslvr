from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy             as np
import sys

# set the relavent directories :
base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'
base_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100' + \
           '_disc_kappa/tmc/10/'
in_dir   = base_dir
out_dir  = base_dir + 'plot/'
var_dir  = 'dump/vars_jakobshavn_small/'

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
fdata = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')

# initialize the variables :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_T(f)
d3model.init_W(f)
d3model.init_Wb_flux(f)
d3model.init_Mb(f)
d3model.init_alpha(f)
d3model.init_PE(f)
d3model.init_W_int(f)
d3model.init_U(f)
d3model.init_p(f)
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

Mb_v     = bedmodel.Mb.vector().array()
alpha_v  = bedmodel.alpha.vector().array()
wb_v     = alpha_v * Mb_v
bedmodel.init_Wb_flux(wb_v)

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

amax  = bedmodel.alpha.vector().max()
amin  = bedmodel.alpha.vector().min()

Mbmax = bedmodel.Mb.vector().max()
Mbmin = bedmodel.Mb.vector().min()

Wbmax = bedmodel.Wb_flux.vector().max()
Wbmin = bedmodel.Wb_flux.vector().min()

Wmax  = bedmodel.W.vector().max()
Wmin  = bedmodel.W.vector().min()

Wimax  = bedmodel.W_int.vector().max()
Wimin  = bedmodel.W_int.vector().min()

bmax  = bedmodel.beta.vector().max()
bmin  = bedmodel.beta.vector().min()

Umax  = srfmodel.U_mag.vector().max()
Umin  = srfmodel.U_mag.vector().min()

Pmax  = bedmodel.PE.vector().max()
Pmin  = bedmodel.PE.vector().min()

Tmax  = bedmodel.T.vector().max()
Tmin  = bedmodel.T.vector().min()

a_lvls  = np.array([0.0, 1e-3, 1e-1, 1e0, 2, 3, 10, amax])
#a_lvls  = np.array([0.0, 1e-3, 1e-1, 3e-1, 9.9e-1, 1.0])
Mb_lvls = np.array([-Mbmax, -0.5, -1e-1, -1e-2, -1e-5,
                    1e-5, 1e-2, 1e-1, 0.5, Mbmax])
Wb_lvls = np.array([-Wbmax, -0.5, -1e-1, -1e-2, -1e-5,
                    1e-5, 1e-2, 1e-1, 0.5, Wbmax])
b_lvls  = np.array([bmin, 1e-3, 1e-2, 1, 1e2, 1e3, 2.5e3, 5e3, bmax])
W_lvls  = np.array([0.0, 1e-2, 3e-2, 4e-2, 5e-2, 1e-1, 2e-1])
U_lvls  = np.array([50, 100, 250, 500, 1e3, 2.5e3, 5e3, Umax])
Pe_lvls = np.array([1e2, 1e3, 5e3, 1e4, 2.5e4, 5e4, Pmax])
T_lvls  = np.array([Tmin, 268, 271.5, 272, 272.5, Tmax])
Wi_lvls = np.array([0.0, 1e-1, 1.0, 5.0, 10])

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
  
plotIce(drg, bedmodel.T, name='T', direc=out_dir, 
        title='$T_B$', cmap=cmap,  scale='lin',
        levels=T_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.W, name='W', direc=out_dir, 
        title=r'$W_B$', cmap=cmap,  scale='lin',
        levels=W_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.Wb_flux, name='Wb_flux', direc=out_dir, 
        title=r'$W_b$', cmap='RdGy',  scale='lin',
        levels=Wb_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.Mb, name='Mb', direc=out_dir, 
        title=r'$M_b$', cmap='RdGy',  scale='lin',
        levels=Mb_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.alpha, name='alpha', direc=out_dir, 
        title=r'$\alpha$', cmap=cmap,  scale='lin',
        levels=a_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.PE, name='PE', direc=out_dir, 
        title=r'$P_e$', cmap=cmap,  scale='lin',
        levels=Pe_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.W_int, name='W_int', direc=out_dir, 
        title=r'$W_i$', cmap=cmap,  scale='lin',
        levels=Wi_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, srfmodel.U_mag, name='U_mag', direc=out_dir, 
        title=r'$\Vert \mathbf{u}_S \Vert$', cmap=cmap,  scale='lin',
        levels=U_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.beta, name='beta', direc=out_dir, 
        title=r'$\beta$', cmap=cmap,  scale='lin',
        levels=b_lvls, tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

plotIce(drg, bedmodel.theta, name='theta', direc=out_dir, 
        title=r'$\theta_B$', cmap=cmap,  scale='lin',
        tp=False, tpAlpha=0.2,
        basin='jakobshavn', extend='neither', show=False, ext='.pdf',
        zoom_box=True, zoom_box_kwargs=zoom_box_kwargs_3)

sys.exit(0)

#===============================================================================
# plot convergence :
abs_err = loadtxt(in_dir + 'tmc/convergence_history/abs_err.txt')
tht_nrm = loadtxt(in_dir + 'tmc/convergence_history/theta_norm.txt')

Ds = []
Js = []
Rs = []
Is = []
ns = []

for d in next(os.walk(in_dir + 'tmc/'))[1]:
  try:
    i = int(d)
    dn = in_dir + 'tmc/' + d + '/objective_ftnls_history/'
    Is.append(i)
    Ds.append(loadtxt(dn + 'Ds.txt'))
    Js.append(loadtxt(dn + 'Js.txt'))
    Rs.append(loadtxt(dn + 'Rs.txt'))
    ns.append(len(Js[-1]))
  except ValueError:
    pass

Ds = array(Ds)
Js = array(Js)
Rs = array(Rs)
Is = array(Is)
ns = array(ns)

idx = argsort(Is)
Ds  = Ds[idx]
Js  = Js[idx]
Rs  = Rs[idx]
ns  = ns[idx]

xn  = cumsum(ns - 1)

Rmax = Rs.max()
Jmax = Js.max()
amax = abs_err.max()
tmax = tht_nrm.max()

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(211)
ax2 = ax1.twinx()
ax3 = fig.add_subplot(212)
ax4 = ax3.twinx()

k = 0

ax3.plot(xn, abs_err - amax, 'ko-', lw=2.0)
ax4.plot(xn, tht_nrm - tmax, 'ro-', lw=2.0)

for i in range(len(Is)):
  xi = arange(k, k + ns[i])
  ax1.plot(xi, Js[i] - Jmax, 'k-', lw=2.0)
  ax2.plot(xi, Rs[i] - Rmax, 'r-', lw=2.0)

  k += ns[i] - 1

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax3.set_xlim([0, xn.max()])
ax4.set_xlim([0, xn.max()])

ax1.set_xlabel('iteration')
ax3.set_xlabel('iteration')
ax1.set_ylabel(r'$\mathscr{J}$')
ax2.set_ylabel(r'$\mathscr{R}$')
ax3.set_ylabel(r'$\Vert \theta - \theta_n \Vert$')
ax4.set_ylabel(r'$\Vert \theta \Vert$')

ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax2.tick_params(axis='y', colors='r')
ax4.tick_params(axis='y', colors='r')

yloc1 = plt.MaxNLocator(4)
yloc2 = plt.MaxNLocator(4)
yloc3 = plt.MaxNLocator(4)
yloc4 = plt.MaxNLocator(4)
ax1.yaxis.set_major_locator(yloc1)
ax2.yaxis.set_major_locator(yloc2)
ax3.yaxis.set_major_locator(yloc3)
ax4.yaxis.set_major_locator(yloc4)

#ax2.set_yscale('log')
ax2.yaxis.get_offset_text().set_color('r')
ax2.yaxis.label.set_color('r')
ax4.yaxis.get_offset_text().set_color('r')
ax4.yaxis.label.set_color('r')

plt.tight_layout()
plt.savefig(out_dir + 'convergence_plot.pdf')
plt.close(fig)


