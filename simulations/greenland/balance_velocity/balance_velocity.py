from cslvr import *

mesh_H = 5

# set plot directory :
plt_dir = '../../../images/balance_velocity/greenland/'

# load the data :
f = HDF5File(mpi_comm_world(), 'dump/vars/state.h5', 'r')

# the balance velocity uses a 2D-model :
model = D2Model(f, out_dir = 'results/')

# set the calculated subdomains :
model.set_subdomains(f)

# use the projection of the dataset 'searise' for plotting :
searise = DataFactory.get_searise()

model.init_S(f)
model.init_B(f)
model.init_adot(f)
model.init_mask(f)
model.init_U_ob(f,f)
model.init_U_mask(f)
      
kappas  = [0,5,10]
methods = ['SUPG', 'SSM', 'GLS']

# the imposed direction of flow :
#d = (model.u_ob, model.v_ob)
d = (-model.S.dx(0), -model.S.dx(1))

# plot the observed surface speed :
U_max  = model.U_ob.vector().max()
U_min  = model.U_ob.vector().min()
U_lvls = array([U_min, 2, 10, 20, 50, 100, 200, 500, 1000, U_max])
plotIce(searise, model.U_ob, name='U_ob', direc=plt_dir,
       title=r'$\Vert \mathbf{u}_{ob} \Vert$', cmap='viridis',
       show=False, levels=U_lvls, tp=False, cb_format='%.1e')

for kappa in kappas:

  for method in methods:

    bv = BalanceVelocity(model, kappa=kappa, stabilization_method=method)
    bv.solve_direction_of_flow(d)
    bv.solve()

    U_max  = model.Ubar.vector().max()
    U_min  = model.Ubar.vector().min()
    U_lvls = array([U_min, 2, 10, 20, 50, 100, 200, 500, 1000, U_max])
    
    name = 'Ubar_%iH_kappa_%i_%s' % (mesh_H, kappa, method)
    tit  = r'$\bar{u}_{%i}$' % kappa
    plotIce(searise, model.Ubar, name=name, direc=plt_dir,
           title=tit, cmap='viridis',
           show=False, levels=U_lvls, tp=False, cb_format='%.1e')
   
    # calculate the misfit
    misfit = Function(model.Q)
    Ubar_v = model.Ubar.vector().array()
    U_ob_v = model.U_ob.vector().array()
    m_v    = U_ob_v - Ubar_v
    model.assign_variable(misfit, m_v)
   
    m_max  = misfit.vector().max()
    m_min  = misfit.vector().min()
    m_lvls = array([m_min, -50, -10, -5, -1, 1, 5, 10, 50, m_max])
     
    name = 'misfit_%iH_kappa_%i_%s' % (mesh_H, kappa, method)
    tit  = r'$M_{%i}$' % kappa
    plotIce(searise, misfit, name=name, direc=plt_dir,
           title=tit, cmap='RdGy',
           show=False, levels=m_lvls, tp=False, cb_format='%.1e')



