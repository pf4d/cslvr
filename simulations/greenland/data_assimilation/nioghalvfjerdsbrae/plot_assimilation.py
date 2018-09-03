from cslvr      import *
from fenics_viz import plot_variable

# directories for saving data :
mdl_odr = 'BP'
reg_typ = 'Tikhonov'#'TV'#
opt_met = 'ipopt'#'l_bfgs_b'#
alpha   = '1.0E-03'

var_dir = './dump/vars/'
out_dir = './dump/results/' + mdl_odr +'/'+ opt_met +'/'\
                                   + reg_typ +'/'+ 'alpha_' + alpha + '/'
out_dir = './dump/results/BP/tmc/'                                  
plt_dir = './dump/images/tmc/'

# create HDF5 files for saving and loading data :
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'submeshes.h5', 'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',     'r')
#f_opt   = HDF5File(mpi_comm_world(), out_dir + 'u_opt.h5',     'r')
f_opt   = HDF5File(mpi_comm_world(), out_dir + 'tmc.h5',       'r')

# create 3D model for stokes solves :
d3model = D3Model(fdata, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)
#d3model.set_srf_mesh(fmeshes)
#d3model.set_bed_mesh(fmeshes)

d3model.form_bed_mesh()
d3model.form_srf_mesh()

# initialize the optimized variables :
d3model.init_beta(f_opt)
d3model.init_U(f_opt)

# create 2D models :
bedmodel = D2Model(d3model.bedmesh, out_dir)
srfmodel = D2Model(d3model.srfmesh, out_dir)

d3model.generate_submesh_to_mesh_map(sub_model=srfmodel)

d3model.assign_to_submesh_variable(u=d3model.beta,  u_sub=bedmodel.beta)
d3model.assign_to_submesh_variable(u=d3model.U3,    u_sub=srfmodel.U3)
d3model.assign_to_submesh_variable(u=d3model.U_mag, u_sub=srfmodel.U_mag)

# figure out some nice-looking contour levels :
U_min  = srfmodel.U_mag.vector().min()
U_max  = srfmodel.U_mag.vector().max()
U_lvls = np.array([U_min, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, U_max])

tp_kwargs     = {'linestyle'      : '-',
                 'lw'             : 0.1,
                 'color'          : 'k',
                 'alpha'          : 0.5}

quiver_kwargs = {'pivot'          : 'middle',
                 'color'          : 'k',
                 'scale'          : 100,
                 'alpha'          : 0.5,
                 'width'          : 0.001,
                 'headwidth'      : 3.0, 
                 'headlength'     : 3.0, 
                 'headaxislength' : 3.0}

# these functions allow the plotting of an arbitrary FEniCS function or 
# vector that reside on a two-dimensional mesh (hence the D2Model
# instantiations above.
plot_variable(u                   = srfmodel.U3,
              name                = 'U_%s' % opt_met,
              coords              = None,
              cells               = None,
              direc               = plt_dir, 
              figsize             = (5,7),
              cmap                = 'viridis',
              scale               = 'lin',
              numLvls             = 10,
              levels              = U_lvls,
              levels_2            = None,
              umin                = None,
              umax                = None,
              plot_tp             = False,
              tp_kwargs           = tp_kwargs,
              show                = False,
              hide_x_tick_labels  = True,#False,
              hide_y_tick_labels  = True,#False,
              xlabel              = '',#r'$x$',
              ylabel              = '',#r'$y$',
              equal_axes          = True,
              title               = r'$\underline{u} |_S^{\mathrm{CSLVR}}$',
              hide_axis           = False,
              colorbar_loc        = 'right',
              contour_type        = 'filled',
              extend              = 'neither',
              ext                 = '.pdf',
              normalize_vec       = True,
              plot_quiver         = True,
              quiver_kwargs       = quiver_kwargs,
              res                 = 150,
              cb                  = True,
              cb_format           = '%g')

