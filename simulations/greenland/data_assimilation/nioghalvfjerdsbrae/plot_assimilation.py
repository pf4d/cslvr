from cslvr      import *
from fenics_viz import plot_variable

# directories for saving data :
mdl_odr = 'BP'
reg_typ = 'TV_Tik_hybrid'#'TV'#'Tikhonov'#
opt_met = 'l_bfgs_b'#'ipopt'#
alpha   = '1.0E-03'

var_dir = './dump/vars/'
out_dir = './dump/results/' + mdl_odr +'/'+ opt_met +'/'\
                                   + reg_typ +'/'+ 'alpha_' + alpha + '/'
#out_dir = './dump/results/BP/tmc/'
out_dir = './dump/results/' + mdl_odr + '/u_opt/'#'/tmc_opt/'
plt_dir = './dump/images/tmc/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')
#f_opt   = HDF5File(mpi_comm_world(), out_dir + 'momentum.h5',  'r')
f_opt   = HDF5File(mpi_comm_world(), out_dir + 'u_opt.h5',     'r')
#f_opt   = HDF5File(mpi_comm_world(), out_dir + 'tmc.h5',       'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
d3model.set_srf_mesh(fmeshes)
d3model.set_bed_mesh(fmeshes)

# initialize the optimized variables :
d3model.assign_variable(d3model.beta, f_opt)
d3model.assign_variable(d3model.U3,   f_opt)
d3model.init_U_mag(d3model.U3)
#d3model.assign_variable(d3model.p,    f_opt)

# create 2D models :
bedmodel = D2Model(d3model.bedmesh, out_dir)
srfmodel = D2Model(d3model.srfmesh, out_dir)

#d3model.generate_submesh_to_mesh_map(sub_model=srfmodel)
#d3model.assign_to_submesh_variable(u=d3model.beta,  u_sub=bedmodel.beta)
#d3model.assign_to_submesh_variable(u=d3model.U3,    u_sub=srfmodel.U3)
#d3model.assign_to_submesh_variable(u=d3model.U_mag, u_sub=srfmodel.U_mag)

d3model.assign_submesh_variable(bedmodel.beta,  d3model.beta)
d3model.assign_submesh_variable(srfmodel.U3,    d3model.U3)
d3model.assign_submesh_variable(srfmodel.U_mag, d3model.U_mag)
d3model.assign_submesh_variable(srfmodel.p,     d3model.p)

# figure out some nice-looking contour levels :
U_min    = srfmodel.U_mag.vector().min()
U_max    = srfmodel.U_mag.vector().max()
beta_min = bedmodel.beta.vector().min()
beta_max = bedmodel.beta.vector().max()

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

#U_lvls = np.array([U_min, 1e0, 5e0, 1e1, 5e1, 1e2, U_max])
U_lvls = np.array([U_min, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, U_max])
#U_lvls = np.array([U_min, 1e1, 1e2, 2.5e2, 4e2, 5e2, 7.5e2, 1e3,
#                   1.2e3, 2e3, U_max])
plt_kwargs['name']        = 'U'
plt_kwargs['title']       = r'$\underline{u} |_S^{\mathrm{CSLVR}}$'
plt_kwargs['levels']      = U_lvls
#plt_kwargs['scale']       = 'log'
plt_kwargs['cmap']        = 'viridis'
#plt_kwargs['plot_quiver'] = True
plt_kwargs['plot_tp']     = False
plot_variable(u=srfmodel.U3, **plt_kwargs)

beta_lvls = np.array([beta_min, 1e-3, 1e1, 1e2, 1e3, 5e3, 1e4, beta_max])
plt_kwargs['name']        = 'beta'
plt_kwargs['title']       = r'$\beta^* |_S^{\mathrm{CSLVR}}$'
plt_kwargs['levels']      = beta_lvls
#plt_kwargs['scale']       = 'lin'
plt_kwargs['cmap']        = 'viridis'
plt_kwargs['plot_tp']     = False
plot_variable(u=bedmodel.beta, **plt_kwargs)

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

#U_lvls = np.array([U_min, 1e1, 1e2, 2.5e2, 4e2, 5e2, 7.5e2, 1e3,
#                   1.2e3, 2e3, U_max])
plotIce(bedmach,
        u      = srfmodel.U_mag,
        name   = 'U_nio',
        levels = U_lvls,
        title  = r'$\underline{u}^* |_S^{\mathrm{CSLVR}}$',
        **plt_params)

beta_lvls = np.array([beta_min, 1e-4, 5e-4, 1e-3, 5e-3, 1e1, 1e2, 1e3, 5e3, beta_max])
plt_params['scale'] = 'lin'
plotIce(bedmach,
        u      = bedmodel.beta,
        name   = 'beta_nio',
        levels = beta_lvls,
        title  = r'$\beta^* |_S^{\mathrm{CSLVR}}$',
        **plt_params)


