from cslvr          import *
from dolfin_adjoint import *

mesh_H = 10
kappa  = 0.0
method = 'GLS'

# load the data :
f = HDF5File(mpi_comm_world(), 'dump/vars/state_%iH.h5' % mesh_H, 'r')

# the balance velocity uses a 2D-model :
model = D2Model(f, out_dir = 'results/', order=1)
    
# set the calculated subdomains :
model.set_subdomains(f)

model.init_S(f)
model.init_B(f)
model.init_B_err(f)
model.init_S_ring(f)
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

B_ringmax = Function(model.Q)
B_ringmin = Function(model.Q)

#B_ringmax.vector()[gnd_dofs] = 0
#B_ringmin.vector()[gnd_dofs] = 0

#B_ringmax.vector()[shf_dofs] = + 1e4
#B_ringmin.vector()[shf_dofs] = - 1e4

B_ringmax.vector()[:] = + 1e4
B_ringmin.vector()[:] = - 1e4

# the imposed direction of flow :
d = (model.u_ob, model.v_ob)
#d = (-model.S.dx(0), -model.S.dx(1))

bv = BalanceVelocity(model, kappa=kappa, stabilization_method=method)
bv.solve_direction_of_flow(d)
#bv.solve(annotate=False)
#model.save_xdmf(model.Ubar, 'Ubar')

ubar_opt_save_vars = [model.B, model.Ubar]

#J_integral = [model.GAMMA_U_GND, model.GAMMA_U_FLT]
#R_integral = [model.GAMMA_S_GND, model.GAMMA_S_FLT]
#
#controls   = [model.B,      model.B_ring]
#bounds     = [(Bmin, Bmax), (B_ringmin, B_ringmax)]

J_measure  = model.dOmega_u
R_measures = model.dOmega#[model.dOmega_w, model.dOmega_g]
controls   = model.B#[model.B_ring,       model.B]
bounds     = (Bmin, Bmax)#[(B_ringmin, B_ringmax), (Bmin, Bmax)]

# form the cost functional :
J = bv.form_obj_ftn(u        = bv.get_unknown(),
                    u_ob     = model.U_ob,
                    integral = model.dOmega,
                    kind     = 'l2')

# form the regularization functional :
R_B_ring = bv.form_reg_ftn(c        = model.B_ring,
                       integral = model.dOmega_w,
                       kind     = 'Tikhonov')
R_B  = bv.form_reg_ftn(c        = model.B,
                       integral = model.dOmega_g,
                       kind     = 'Tikhonov')

# this is the objective functional :
I = J# + 1e1*R_B_ring + 1e1*R_B

# perform the optimization :
bv.optimize(u                 = bv.get_unknown(),
            u_ob              = model.U_ob,
            I                 = I,
            control           = controls,
            J_measure         = J_measure,
            R_measure         = R_measures,
            bounds            = bounds,
            method            = 'ipopt',
            max_iter          = 100,
            adj_save_vars     = ubar_opt_save_vars,
            adj_callback      = None,
            post_adj_callback = None)

model.save_xdmf(model.B,    'B_opt')
model.save_xdmf(model.B_ring,   'B_ring_opt')
model.save_xdmf(model.Ubar, 'Ubar_opt')
model.save_xdmf(model.U_ob, 'U_ob')



