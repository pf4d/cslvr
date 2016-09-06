from cslvr  import *
from scipy  import random

# out directories for saving data and images :
#reg_typ = 'TV'
reg_typ = 'Tikhonov'
#reg_typ = 'TV_Tik_hybrid'
out_dir = './ISMIP_HOM_C_inverse_' + reg_typ + '_results/'
plt_dir = '../../images/data_assimilation/ISMIP_HOM_C/' + reg_typ + '/'

# constants used :
a     = 0.5 * pi / 180
L     = 20000
bmax  = 1000

# a generic box mesh that will be fit to geometry below :
p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 15, 15, 5)

# this is a 3D model :
model = D3Model(mesh, out_dir = out_dir + 'true/', use_periodic = True)

# expressions for the surface and basal topography, and friction :
surface = Expression('- x[0] * tan(a)', a=a, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(a) - 1000.0', a=a, 
                     element=model.Q.ufl_element())
beta    = Expression('bmax/2 + bmax/2 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     bmax=bmax, L=L, element=model.Q.ufl_element())

# calculate the boundaries for integration :
model.calculate_boundaries()

# deform the mesh to the desired geometry :
model.deform_mesh_to_geometry(surface, bed)

# create the bed and surface meshes, for plotting purposes : 
model.form_bed_mesh()
model.form_srf_mesh()

# create 2D models, again for plotting only :
bedmodel = D2Model(model.bedmesh, out_dir)
srfmodel = D2Model(model.srfmesh, out_dir)

# initialize important variables :
model.init_beta(beta)                          # traction
model.init_A(1e-16)                            # isothermal rate-factor

# create the first-order momentum object and solve :
mom = MomentumDukowiczBP(model)
mom.solve(annotate=False)

# add noise with a signal-to-noise ratio of 100 :
snr   = 100.0
u     = Function(model.Q)
v     = Function(model.Q)
assign(u, model.U3.sub(0))
assign(v, model.U3.sub(1))
u_o   = u.vector().array()
v_o   = v.vector().array()
n     = len(u_o)
sig   = model.get_norm(as_vector([u, v]), 'linf')[1] / snr
print_min_max(sig, 'sigma')
print_min_max(snr, 'SNR')
  
u_error = sig * random.randn(n)
v_error = sig * random.randn(n)
u_ob    = u_o + u_error
v_ob    = v_o + v_error

# init the 'observed' velocity :
model.init_U_ob(u_ob, v_ob)
u_ob_ex = model.vert_extrude(model.u_ob, 'down')
v_ob_ex = model.vert_extrude(model.v_ob, 'down')
model.init_U_ob(u_ob_ex, v_ob_ex)

# assign variables to the submesh for plotting :
bedmodel.assign_submesh_variable(bedmodel.beta, model.beta)
srfmodel.assign_submesh_variable(srfmodel.U_ob, model.U_ob)
srfmodel.assign_submesh_variable(srfmodel.u_ob, model.u_ob)
srfmodel.assign_submesh_variable(srfmodel.v_ob, model.v_ob)
srfmodel.assign_submesh_variable(srfmodel.U3,   model.U3)
srfmodel.init_U_mag(srfmodel.U3)

# zero out the vertical velocity for comparison :
srfmodel.init_w(Constant(0.0))

# plotting :
beta_min  = bedmodel.beta.vector().min()
beta_max  = bedmodel.beta.vector().max()
beta_lvls = array([beta_min, 100, 200, 300, 400, 500, 600, 
                   700, 800, 900, beta_max])
plot_variable(u = bedmodel.beta, name = 'beta_true', direc = plt_dir,
              cmap = 'viridis', figsize = (6,5), levels = beta_lvls, tp = True,
              show = False, cb_format='%i', hide_ax_tick_labels=True)

U_min  = srfmodel.U_mag.vector().min()
U_max  = srfmodel.U_mag.vector().max()
U_lvls = array([U_min, 170, 180, 200, 220, 240, 260, 280,  U_max])
plot_variable(u = srfmodel.U3, name = 'U_true', direc = plt_dir,
              cmap = 'viridis', figsize = (6,5), levels = U_lvls, tp = True,
              show = False, cb_format='%i', hide_ax_tick_labels=True)

U_ob_min  = srfmodel.U_ob.vector().min()
U_ob_max  = srfmodel.U_ob.vector().max()
U_ob_lvls = array([U_ob_min, 170, 180, 200, 220, 240, 260, 280,  U_ob_max])
U_ob = as_vector([srfmodel.u_ob, srfmodel.v_ob])
plot_variable(u = U_ob, name = 'U_ob', direc = plt_dir,
              cmap = 'viridis', figsize = (6,5), levels = U_ob_lvls, tp = True,
              show = False, cb_format='%i', hide_ax_tick_labels=True)

# calculate the initial tractin field from the SIA approximation :
model.init_beta_SIA()

# model.beta has been reassigned, so let's plot it :
bedmodel.assign_submesh_variable(bedmodel.beta, model.beta)
beta_min  = bedmodel.beta.vector().min()
beta_max  = bedmodel.beta.vector().max()
beta_lvls = array([250, 275, 300, 325, 350, 375, 400, 425, 450, beta_max])
plot_variable(u = bedmodel.beta, name = 'beta_SIA', direc = plt_dir,
              cmap = 'viridis', figsize = (6,5), levels = beta_lvls, tp = True,
              show = False, cb_format='%i', hide_ax_tick_labels=True,
              extend = 'min')

model.set_out_dir(out_dir + 'inversion/')

# post-adjoint-iteration callback function :
def adj_post_cb_ftn():
  """
  this is called when the optimization is done.  Here all we do is plot, but 
  you may want to calculate other variables of interest to :
  """
  bedmodel.assign_submesh_variable(bedmodel.beta, model.beta)
  srfmodel.assign_submesh_variable(srfmodel.U3,   model.U3)
  srfmodel.init_U_mag(srfmodel.U3)
  srfmodel.init_w(Constant(0.0))

  # plot beta optimal :
  beta_min  = bedmodel.beta.vector().min()
  beta_max  = bedmodel.beta.vector().max()
  beta_lvls = array([beta_min, 100, 200, 300, 400, 500, 600, 700, 
                     800, 900, beta_max])
  plot_variable(u = bedmodel.beta, name = 'beta_opt', direc = plt_dir,
                cmap = 'viridis', figsize = (6,5), levels = beta_lvls,
                tp = True, show = False, cb_format='%i',
                hide_ax_tick_labels=True)
  
  # plot u optimal :
  U_min  = srfmodel.U_mag.vector().min()
  U_max  = srfmodel.U_mag.vector().max()
  U_lvls = array([U_min, 170, 180, 200, 220, 240, 260, 280,  U_max])
  plot_variable(u = srfmodel.U3, name = 'U_opt', direc = plt_dir,
                cmap = 'viridis', figsize = (6,5), levels = U_lvls, tp = True,
                show = False, cb_format='%i', hide_ax_tick_labels=True)
  
  # or we could save the 3D optimized velocity and beta fields for 
  # viewing with paraview, like this :
  model.save_xdmf(model.U3,   'U_opt')
  model.save_xdmf(model.beta, 'beta_opt')

# after every completed adjoining, save the state of these functions :
adj_save_vars = [model.beta, model.U3]

# form the cost functional :
mom.form_obj_ftn(integral=model.GAMMA_U_GND, kind='log_L2_hybrid', 
                 g1=1, g2=1e5)

# form the regularization functional :
if reg_typ == 'TV':
  mom.form_reg_ftn(model.beta, integral=model.GAMMA_B_GND, kind='TV', 
                   alpha=100.0)
elif reg_typ == 'Tikhonov':
  mom.form_reg_ftn(model.beta, integral=model.GAMMA_B_GND, kind='Tikhonov', 
                   alpha=500.0)
elif reg_typ == 'TV_Tik_hybrid':
  mom.form_reg_ftn(model.beta, integral=model.GAMMA_B_GND,
                   kind='TV_Tik_hybrid', alpha_tik=250, alpha_tv=50)

# solving the incomplete adjoint is more efficient :
mom.linearize_viscosity()

# optimize for beta :
mom.optimize_U_ob(control           = model.beta,
                  bounds            = (1e-5, 1e7),
                  method            = 'ipopt',
                  max_iter          = 100,
                  adj_save_vars     = adj_save_vars,
                  adj_callback      = None,
                  post_adj_callback = adj_post_cb_ftn)



