from   cslvr  import *
import numpy      as np
import fenics_viz as fv
import os


# set the relavent directories :
var_dir = './dump/vars/'                   # directory from gen_vars.py
plt_dir = './dump/images/'                 # directory to save images
out_dir = './dump/nio_small/inversion/'    # directory to load results

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
d3model.set_srf_mesh(fmeshes)
d3model.set_bed_mesh(fmeshes)
#d3model.set_dvd_mesh(fmeshes)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_q_geo(d3model.ghf)
d3model.init_T_surface(fdata)
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)

d3model.init_beta_SIA()

fmeshes.close()
fdata.close()

bedmodel = D2Model(d3model.bedmesh, out_dir)
srfmodel = D2Model(d3model.srfmesh, out_dir)

bedmodel.assign_submesh_variable(bedmodel.mask,  d3model.mask)
bedmodel.assign_submesh_variable(bedmodel.beta,  d3model.beta)
srfmodel.assign_submesh_variable(srfmodel.S,     d3model.S)
srfmodel.assign_submesh_variable(srfmodel.B,     d3model.B)
srfmodel.assign_submesh_variable(srfmodel.U3,    d3model.U3)
srfmodel.assign_submesh_variable(srfmodel.U_ob,  d3model.U_ob) 
srfmodel.assign_submesh_variable(srfmodel.T,     d3model.T_surface)

mask     = bedmodel.mask
beta_sia = bedmodel.beta
S        = srfmodel.S
B        = srfmodel.B
U_ob     = srfmodel.U_ob
T        = srfmodel.T

#calculate the thickness :
H_v      = S.vector().get_local() - B.vector().get_local()
H        = Function(srfmodel.Q)
H.vector().set_local(H_v)

#===============================================================================
# plot the data :
tp_kwargs     = {'linestyle'      : '-',
                 'lw'             : 0.1,
                 'color'          : 'k',
                 'alpha'          : 0.8}

quiver_kwargs = {'pivot'          : 'middle',
                 'color'          : '0.0',
                 'scale'          : 100,
                 'alpha'          : 1.0,
                 'width'          : 0.001,
                 'headwidth'      : 3.0, 
                 'headlength'     : 3.0, 
                 'headaxislength' : 3.0}

plt_kwargs  =  {'direc'              : plt_dir, 
                'coords'             : None,
                'cells'              : None,
                'figsize'            : (5.6,8),
                'cmap'               : 'viridis',
                'scale'              : 'lin',
                'numLvls'            : 10,
                'levels'             : None,#U_lvls,
                'levels_2'           : None,
                'umin'               : None,
                'umax'               : None,
                'plot_tp'            : False,
                'tp_kwargs'          : tp_kwargs,
                'show'               : False,
                'hide_x_tick_labels' : True,
                'hide_y_tick_labels' : True,
                'xlabel'             : "",#r'$x$',
                'ylabel'             : "",#r'$y$',
                'equal_axes'         : True,
                'title'              : r'$S |^{\mathrm{CSLVR}}$',
                'hide_axis'          : True,
                'colorbar_loc'       : 'right',
                'contour_type'       : 'filled',
                'extend'             : 'neither',
                'ext'                : '.pdf',
                'normalize_vec'      : True,
                'plot_quiver'        : False,
                'quiver_kwargs'      : quiver_kwargs,
                'res'                : 150,
                'cb'                 : True,
                'cb_format'          : '%g'}

plt_kwargs['name']   = 'S'
plt_kwargs['title']  =  r'$S |^{\mathrm{CSLVR}}$'
fv.plot_variable(u=S, **plt_kwargs)

plt_kwargs['name']   = 'B'
plt_kwargs['title']  =  r'$B |^{\mathrm{CSLVR}}$'
fv.plot_variable(u=B, **plt_kwargs)

plt_kwargs['name']   = 'H'
plt_kwargs['title']  =  r'$H |^{\mathrm{CSLVR}}$'
fv.plot_variable(u=H, **plt_kwargs)

plt_kwargs['name']    = 'mask'
plt_kwargs['title']   =  ''#r'$\mathrm{mask} |^{\mathrm{CSLVR}}$'
#plt_kwargs['cmap']    = 'RdGy'
plt_kwargs['levels']  = np.array([-0.5, 0.5, 1.5, 2.5, 3])
fv.plot_variable(u=mask, **plt_kwargs)

T_lvls = np.array([T.vector().min(), 242, 244, 246, 248, 250, 252, 254, 256, 
                   T.vector().max()])
plt_kwargs['levels']  = T_lvls
plt_kwargs['scale']   = 'lin'
plt_kwargs['plot_tp'] = False
plt_kwargs['name']    = 'T'
plt_kwargs['title']   =  r'$T |_S^{\mathrm{CSLVR}}$'
fv.plot_variable(u=T, **plt_kwargs)

U_lvls = np.array([U_ob.vector().min(), 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3,
                   U_ob.vector().max()])
plt_kwargs['name']        = 'U_ob'
plt_kwargs['title']       = r'$\underline{u}_{\mathrm{ob}} |_S^{\mathrm{CSLVR}}$'
plt_kwargs['levels']      = U_lvls
plt_kwargs['scale']       = 'lin'
plt_kwargs['cmap']        = 'viridis'
#plt_kwargs['plot_quiver'] = True
plt_kwargs['plot_tp']     = False
fv.plot_variable(u=U_ob, **plt_kwargs)

beta_lvls = np.array([beta_sia.vector().min(), 2e1, 1e2, 5e2, 1e3, 5e3, 1e4,
                      2e4, beta_sia.vector().max()])
plt_kwargs['name']        = 'beta_sia'
plt_kwargs['title']       = r'$\beta_{\mathrm{SIA}} |^{\mathrm{CSLVR}}$'
plt_kwargs['levels']      = beta_lvls
plt_kwargs['scale']       = 'lin'
plt_kwargs['cmap']        = 'viridis'
plt_kwargs['plot_quiver'] = False
fv.plot_variable(u=beta_sia, **plt_kwargs)

bedmach    = DataFactory.get_bedmachine(thklim=1.0)

nio_params = {'llcrnrlat'    :  78.5,
              'llcrnrlon'    : -27.0,
              'urcrnrlat'    :  79.6,
              'urcrnrlon'    : -17.0,
              'scale_color'  : 'k',
              'scale_length' : 40.0,
              'scale_loc'    : 3,
              'figsize'      : (8,8),
              'lat_interval' : 0.5,
              'lon_interval' : 5,
              'plot_grid'    : True,
              'plot_scale'   : True,
              'axes_color'   : 'k'}

plt_params = {'direc'            : plt_dir,
              'coords'           : None,
              'cells'            : None,
              'u2'               : None,
              'u2_levels'        : None,
              'u2_color'         : 'k',
              'u2_linewidth'     : 1.0,
              'cmap'             : 'viridis',
              'scale'            : 'lin',
              'umin'             : None,
              'umax'             : None,
              'numLvls'          : 12,
              'drawGridLabels'   : True,
              'levels_2'         : None,
              'tp'               : True,
              'tpAlpha'          : 0.5,
              'contour_type'     : 'filled',
              'params'           : nio_params,
              'extend'           : 'neither',
              'show'             : False,
              'ext'              : '.pdf',
              'res'              : 150,
              'cb'               : True,
              'cb_format'        : '%g',
              'zoom_box'         : False,
              'zoom_box_kwargs'  : None,
              'plot_pts'         : None,
              'plot_texts'       : None,
              'plot_continent'   : False,
              'cont_plot_params' : None,
              'drawcoastlines'   : True,
              'box_params'       : None}

S_lvls = np.array([S.vector().min(), 10, 50, 100, 200, 400, 600, 800, 1000,
                   S.vector().max()])
plotIce(bedmach,
        u      = S, 
        name   = 'S_nio',
        levels = S_lvls,
        title  = r'$S |^{\mathrm{CSLVR}}$',
        **plt_params)

B_lvls = np.array([B.vector().min(), -500, -250, -100, 0, 100, 250, 500, 750,
                   1000, B.vector().max()])
plotIce(bedmach,
        u      = B, 
        name   = 'B_nio',
        levels = B_lvls,
        title  = r'$B |^{\mathrm{CSLVR}}$',
        **plt_params)

H_lvls = np.array([H.vector().min(), 50, 100, 250, 500, 750, 1000, 1250,
                   H.vector().max()])
plotIce(bedmach,
        u      = H, 
        name   = 'H_nio',
        levels = H_lvls,
        title  = r'$H |^{\mathrm{CSLVR}}$',
        **plt_params)


T_lvls = np.array([T.vector().min(), 251, 252, 253, 254, 255, T.vector().max()])
plotIce(bedmach,
        u      = T, 
        name   = 'T_nio',
        levels = T_lvls,
        title  = r'$T |_S^{\mathrm{CSLVR}}$',
        **plt_params)

U_lvls = np.array([U_ob.vector().min(), 1e1, 1e2, 2.5e2, 4e2, 5e2, 7.5e2, 1e3, 
                   1.2e3, 2e3, U_ob.vector().max()])
plotIce(bedmach,
        u      = U_ob, 
        name   = 'U_ob_nio',
        levels = U_lvls,
        title  = r'$\underline{u}_{\mathrm{ob}} |_S^{\mathrm{CSLVR}}$',
        **plt_params)

beta_lvls = np.array([beta_sia.vector().min(), 2e1, 1e2, 5e2, 1e3, 5e3, 1e4,
                      beta_sia.vector().max()])
plotIce(bedmach,
        u      = beta_sia, 
        name   = 'beta_sia_nio',
        levels = beta_lvls,
        title  = r'$\beta_{\mathrm{SIA}} |^{\mathrm{CSLVR}}$',
        **plt_params)

plt_params['tp'] = True
plotIce(bedmach,
        u      = mask, 
        name   = 'mask_nio',
        levels = np.array([-0.5, 0.5, 1.5, 2.5, 3]),
        title  = r'',
        **plt_params)





