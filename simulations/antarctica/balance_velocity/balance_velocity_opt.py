from cslvr          import *
from dolfin_adjoint import *

mesh_H = 10
kappa  = 0.0
method = 'GLS'

# set plot directory :
plt_dir = '../../../images/balance_velocity/antarctica/'

# load the data :
f = HDF5File(mpi_comm_world(), 'dump/vars/state_%iH.h5' % mesh_H, 'r')

# the balance velocity uses a 2D-model :
model = D2Model(f, out_dir = 'results/', order=1)
    
# set the calculated subdomains :
model.set_subdomains(f)

# use the projection of the dataset 'bedmap1' for plotting :
bm1  = DataFactory.get_bedmap1()

model.init_S(f)
model.init_B(f)
model.init_B_err(f)
model.init_adot(f)
model.init_mask(f)
model.init_U_ob(f,f)
model.init_U_mask(f)

shf_dofs = model.mask.vector().array() == 2
gnd_dofs = model.mask.vector().array() == 1

Bmax = Function(model.Q)
Bmin = Function(model.Q)

Bmax.vector()[:] = model.B.vector()[:] + model.B_err.vector()[:]#1e-323
Bmin.vector()[:] = model.B.vector()[:] - model.B_err.vector()[:]#1e-323

#Bmax.vector()[gnd_dofs] += model.B_err.vector()[gnd_dofs]
#Bmin.vector()[gnd_dofs] -= model.B_err.vector()[gnd_dofs]

Fbmax = Function(model.Q)
Fbmin = Function(model.Q)

Fbmax.vector()[gnd_dofs] = 0
Fbmin.vector()[gnd_dofs] = 0

Fbmax.vector()[shf_dofs] = + 1e4
Fbmin.vector()[shf_dofs] = - 1e4

model.save_xdmf(Fbmax, 'Fbmax')
model.save_xdmf(model.mask, 'mask')

# the imposed direction of flow :
d = (model.u_ob, model.v_ob)
#d = (-model.S.dx(0), -model.S.dx(1))

## plot the observed surface speed :
#U_max  = model.U_ob.vector().max()
#U_min  = model.U_ob.vector().min()
#U_lvls = array([U_min, 2, 10, 20, 50, 100, 200, 500, 1000, U_max])
#plotIce(bm1, model.U_ob, name='U_ob', direc=plt_dir, drawGridLabels=False,
#       title=r'$\Vert \mathbf{u}_{ob} \Vert$', cmap='viridis',
#       show=False, levels=U_lvls, tp=False, cb_format='%.1e')

bv = BalanceVelocity(model, kappa=kappa, stabilization_method=method)
bv.solve_direction_of_flow(d)
#bv.solve(annotate=False)
#model.save_xdmf(model.Ubar, 'Ubar')
#
#U_max  = model.Ubar.vector().max()
#U_min  = model.Ubar.vector().min()
#U_lvls = array([U_min, 2, 10, 20, 50, 100, 200, 500, 1000, U_max])
#
#name = 'Ubar_%iH_kappa_%i_%s' % (mesh_H, kappa, method)
#tit  = r'$\bar{u}_{%i}$' % kappa
#plotIce(bm1, model.Ubar, name=name, direc=plt_dir,
#       title=tit, cmap='viridis', drawGridLabels=False,
#       show=False, levels=U_lvls, tp=False, cb_format='%.1e')
#
## calculate the misfit
#misfit = Function(model.Q)
#Ubar_v = model.Ubar.vector().array()
#U_ob_v = model.U_ob.vector().array()
#m_v    = U_ob_v - Ubar_v
#model.assign_variable(misfit, m_v)
#
#m_max  = misfit.vector().max()
#m_min  = misfit.vector().min()
##m_lvls = array([m_min, -5e2, -1e2, -1e1, -1, 1, 1e1, 1e2, 5e2, m_max])
#m_lvls = array([m_min, -50, -10, -5, -1, 1, 5, 10, 50, m_max])
# 
#name = 'misfit_%iH_kappa_%i_%s' % (mesh_H, kappa, method)
#tit  = r'$M_{%i}$' % kappa
#plotIce(bm1, misfit, name=name, direc=plt_dir,
#       title=tit, cmap='RdGy', drawGridLabels=False,
#       show=False, levels=m_lvls, tp=False, cb_format='%.1e')

ubar_opt_save_vars = [model.B, model.Ubar]

#J_integral = [model.GAMMA_U_GND, model.GAMMA_U_FLT]
#R_integral = [model.GAMMA_S_GND, model.GAMMA_S_FLT]
#
#controls   = [model.B,      model.Fb]
#bounds     = [(Bmin, Bmax), (Fbmin, Fbmax)]

J_integral = [model.GAMMA_U_FLT]
R_integral = [model.GAMMA_U_FLT]

controls   = [model.Fb]
bounds     = [(Fbmin, Fbmax)]

# form the cost functional :
bv.form_obj_ftn(integral=J_integral, kind='abs')

# form the regularization functional :
bv.form_reg_ftn(model.Fb, integral=R_integral, kind='Tikhonov', alpha=1e-2)

kwargs = {'control'             : controls,
          'bounds'              : bounds,
          'method'              : 'ipopt',
          'max_iter'            : 100,
          'adj_save_vars'       : ubar_opt_save_vars,
          'adj_callback'        : None,
          'post_adj_callback'   : None}
                                    
# perform the optimization :
bv.optimize_ubar(**kwargs)

model.save_xdmf(model.B,    'B_opt')
model.save_xdmf(model.Fb,   'Fb_opt')
model.save_xdmf(model.Ubar, 'Ubar_opt')
model.save_xdmf(model.U_ob, 'U_ob')

