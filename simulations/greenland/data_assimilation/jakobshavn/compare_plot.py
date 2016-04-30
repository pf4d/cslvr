from cslvr    import *
from fenics   import *
import matplotlib.pyplot as plt
import numpy as np
import sys

# set the relavent directories :
base_dir_1 = 'dump/jakob_small/inversion_Wc_0.01/10/'
base_dir_2 = 'dump/jakob_small/inversion_Wc_0.03/10/'
in_dir_1 = base_dir_1
in_dir_2 = base_dir_2
out_dir  = 'dump/jakob_small/deltas/'
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
fdata    = HDF5File(mpi_comm_world(), var_dir + 'state.h5', 'r')
f_1      = HDF5File(mpi_comm_world(), in_dir_1  + 'tmc.h5', 'r')
f_2      = HDF5File(mpi_comm_world(), in_dir_2  + 'tmc.h5', 'r')

Fb_1     = Function(d3model.Q,  name='Fb')
Fb_2     = Function(d3model.Q,  name='Fb')
beta_1   = Function(d3model.Q,  name='beta')
beta_2   = Function(d3model.Q,  name='beta')
U_1      = Function(d3model.Q3, name='U3')
U_2      = Function(d3model.Q3, name='U3')

Fb_1_b   = Function(bedmodel.Q)
Fb_2_b   = Function(bedmodel.Q)
beta_1_b = Function(bedmodel.Q)
beta_2_b = Function(bedmodel.Q)
U_1_s    = Function(srfmodel.Q3)
U_2_s    = Function(srfmodel.Q3)

a_d = Function(bedmodel.Q)
b_d = Function(bedmodel.Q)
u_d = Function(srfmodel.Q)

# initialize the variables :
d3model.assign_variable(Fb_1,   f_1)
d3model.assign_variable(Fb_2,   f_2)
d3model.assign_variable(beta_1, f_1)
d3model.assign_variable(beta_2, f_2)
d3model.assign_variable(U_1,    f_1)
d3model.assign_variable(U_2,    f_2)

# 2D model gets balance-velocity appropriate variables initialized :
bedmodel.assign_submesh_variable(Fb_1_b,   Fb_1)
bedmodel.assign_submesh_variable(Fb_2_b,   Fb_2)
bedmodel.assign_submesh_variable(beta_1_b, beta_1)
bedmodel.assign_submesh_variable(beta_2_b, beta_2)
srfmodel.assign_submesh_variable(U_1_s,    U_1)
srfmodel.assign_submesh_variable(U_2_s,    U_2)

a_1_v  = Fb_1_b.vector().array()
a_2_v  = Fb_2_b.vector().array()
a_d_v  = a_1_v - a_2_v
a_d.vector().set_local(a_d_v)
a_d.vector().apply('insert')

b_1_v  = beta_1_b.vector().array()
b_2_v  = beta_2_b.vector().array()
b_d_v  = b_1_v / (b_2_v + DOLFIN_EPS)
b_d.vector().set_local(b_d_v)
b_d.vector().apply('insert')

u_1, v_1, w_1 = U_1_s.split(True)
u_2, v_2, w_2 = U_2_s.split(True)

u_1_v = u_1.vector().array()
v_1_v = v_1.vector().array()
w_1_v = w_1.vector().array()
u_2_v = u_2.vector().array()
v_2_v = v_2.vector().array()
w_2_v = w_2.vector().array()

Umag_1_v = np.sqrt(u_1_v**2 + v_1_v**2 + w_1_v**2 + DOLFIN_EPS)
Umag_2_v = np.sqrt(u_2_v**2 + v_2_v**2 + w_2_v**2 + DOLFIN_EPS)
u_d_v    = Umag_1_v / Umag_2_v
u_d.vector().set_local(u_d_v)
u_d.vector().apply('insert')

#===============================================================================
d3model.init_B(fdata)
bedmodel.assign_submesh_variable(bedmodel.B, d3model.B)


#===============================================================================
drg  = DataFactory.get_rignot()

bc = '#880cbc'

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

amax  = a_d.vector().max()
amin  = a_d.vector().min()

bmax  = b_d.vector().max()
bmin  = b_d.vector().min()

umax  = u_d.vector().max()
umin  = u_d.vector().min()

a_d_lvls  = np.array([amin, -1e-1, -5e-2, -2.5e-2, -1e-3, 
                      1e-3, 2.5e-2, 5e-2, 1e-1, amax])
b_d_lvls  = np.array([bmin, -200, -80, -40, -10, 
                      10, 40, 80, 200, bmax])
u_d_lvls  = np.array([umin, -50, -25, -10, -1, 1, 10, 25, 50, umax])

b_d_lvls  = np.array([0.2,  0.4,   0.6,   0.8,  0.95,
                      1.05, 1.20,  1.5,   4.0,  6])
u_d_lvls  = np.array([0.95,  0.97, 0.99, 0.995, 0.9975,
                      1.0025, 1.005, 1.01, 1.03,  1.05])

#===============================================================================
# plot :

plotIce(drg, a_d, name='delta_Fb', direc=out_dir, 
        title=r'$\Delta F_b$', cmap='RdGy',  scale='lin',
        levels=a_d_lvls, tp=True, tpAlpha=0.2,
        extend='neither', show=False, ext='.pdf',
        params=params, plot_pts=None)

plotIce(drg, b_d, name='delta_beta', direc=out_dir,
        title=r'$\beta_{0.01} / \beta_{0.03}$', cmap='RdGy',  scale='lin',
        levels=b_d_lvls, tp=True, tpAlpha=0.2,
        extend='both', show=False, ext='.pdf',
        params=params, plot_pts=None, cb_format='%g',
        u2=bedmodel.B, u2_levels=[-500], u2_color=bc)

plotIce(drg, u_d, name='delta_U', direc=out_dir,
        title=r'$\Vert \mathbf{u}_S \Vert_{0.01} / \Vert \mathbf{u}_S \Vert_{0.03}$', cmap='RdGy',  scale='lin',
        levels=u_d_lvls, tp=True, tpAlpha=0.2,
        extend='both', show=False, ext='.pdf',
        params=params, plot_pts=None, cb_format='%g')



