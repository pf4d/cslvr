from cslvr  import *
from pylab  import *
import sys
import os

#===============================================================================
# get the data from the model output on the bed :

# set the relavent directories :
base_dir = 'dump/jakob_small/inversion_Wc_0.03/01/'
in_dir   = base_dir
out_dir  = base_dir
var_dir  = 'dump/vars_jakobshavn_small/'

# create HDF5 files for saving and loading data :
finv    = HDF5File(mpi_comm_world(), in_dir  + 'inverted_01.h5',       'r')
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',             'r')
fout    = HDF5File(mpi_comm_world(), out_dir + 'linear_model_vars.h5', 'w')

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model :
d3model = D3Model(mesh, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_T_surface(fdata)
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
d3model.init_beta(finv)
d3model.init_U(finv)
d3model.init_T(finv)
d3model.init_W(finv)
d3model.init_theta(finv)
d3model.init_Mb(finv)

d3model.init_tau_ii(fstress)
d3model.init_tau_ij(fstress)
d3model.init_tau_ik(fstress)
d3model.init_tau_ji(fstress)
d3model.init_tau_jj(fstress)
d3model.init_tau_jk(fstress)
d3model.init_tau_ki(fstress)
d3model.init_tau_kj(fstress)
d3model.init_tau_kk(fstress)


#===============================================================================
# form surface meshes :
d3model.form_bed_mesh()
d3model.form_srf_mesh()

# create 2D model for balance velocity :
d2model = D2Model(d3model.bedmesh, out_dir)

# 2D model gets balance-velocity appropriate variables initialized :
d2model.assign_submesh_variable(d2model.S,         d3model.S)
d2model.assign_submesh_variable(d2model.B,         d3model.B)
d2model.assign_submesh_variable(d2model.T_surface, d3model.T_surface)
d2model.assign_submesh_variable(d2model.adot,      d3model.adot)
d2model.assign_submesh_variable(d2model.u_ob,      d3model.u_ob)
d2model.assign_submesh_variable(d2model.v_ob,      d3model.v_ob)
d2model.assign_submesh_variable(d2model.U_ob,      d3model.U_ob)
d2model.assign_submesh_variable(d2model.beta,      d3model.beta)
d2model.assign_submesh_variable(d2model.U3,        d3model.U3)
d2model.assign_submesh_variable(d2model.U_mag,     d3model.U_mag)
d2model.assign_submesh_variable(d2model.T,         d3model.T)
d2model.assign_submesh_variable(d2model.W,         d3model.W)
d2model.assign_submesh_variable(d2model.Mb,        d3model.Mb)
d2model.assign_submesh_variable(d2model.tau_ii,    d3model.tau_ii)
d2model.assign_submesh_variable(d2model.tau_ij,    d3model.tau_ij)
d2model.assign_submesh_variable(d2model.tau_ik,    d3model.tau_ik)
d2model.assign_submesh_variable(d2model.tau_ji,    d3model.tau_ji)
d2model.assign_submesh_variable(d2model.tau_jj,    d3model.tau_jj)
d2model.assign_submesh_variable(d2model.tau_jk,    d3model.tau_jk)

# create a new 2D model for surface variables :
srfmodel = D2Model(d3model.srfmesh, out_dir)

# put the velocity on it :
d2model.assign_submesh_variable(srfmodel.U3,        d3model.U3)
d2model.assign_submesh_variable(srfmodel.U_mag,     d3model.U_mag)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.Ubar5 = d2model.Ubar
d2model.Ubar5.rename('Ubar5', '')

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=10.0)
bv.solve(annotate=False)

d2model.Ubar10 = d2model.Ubar
d2model.Ubar10.rename('Ubar10', '')

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=20.0)
bv.solve(annotate=False)

d2model.Ubar20 = d2model.Ubar
d2model.Ubar20.rename('Ubar20', '')

# collect the data :
Q      = d2model.Q
u,v,w  = d2model.U3.split(True)
S      = d2model.S 
B      = d2model.B
adot   = d2model.adot
qgeo   = d2model.q_geo
beta   = d2model.beta
Mb     = d2model.Mb
Tb     = d2model.T
Ts     = d2model.T_surface
Ubar5  = d2model.Ubar5
Ubar10 = d2model.Ubar10
Ubar20 = d2model.Ubar20
U_ob   = d2model.U_ob
tau_ii = d2model.tau_ii
tau_ij = d2model.tau_ij
tau_ik = d2model.tau_ik
tau_ji = d2model.tau_ji
tau_jj = d2model.tau_jj
tau_jk = d2model.tau_jk
mask   = d2model.mask

dSdx   = project(S.dx(0), Q)
dSdy   = project(S.dx(1), Q)

dBdx   = project(B.dx(0), Q)
dBdy   = project(B.dx(1), Q)
                        
dHdx   = project((S - B).dx(0), Q)
dHdy   = project((S - B).dx(1), Q)

# vectors :
beta_v   = beta.vector().array()
S_v      = S.vector().array()
B_v      = B.vector().array()
adot_v   = adot.vector().array()
qgeo_v   = qgeo.vector().array()
Mb_v     = Mb.vector().array()
Tb_v     = Tb.vector().array()
Ts_v     = Ts.vector().array()
u_v      = u.vector().array()
v_v      = v.vector().array()
w_v      = w.vector().array()
Ubar5_v  = Ubar5.vector().array()
Ubar10_v = Ubar10.vector().array()
Ubar20_v = Ubar20.vector().array()
U_ob_v   = U_ob.vector().array()
tau_ii_v = tau_ii.vector().array()
tau_ij_v = tau_ij.vector().array()
tau_ik_v = tau_ik.vector().array()
tau_ji_v = tau_ji.vector().array()
tau_jj_v = tau_jj.vector().array()
tau_jk_v = tau_jk.vector().array()
mask_v   = mask.vector().array()

H_v    = S_v - B_v
dHdx_v = dHdx.vector().array()
dHdy_v = dHdy.vector().array()
gradH  = sqrt(dHdx_v**2 + dHdy_v**2 + 1e-16)
U_mag  = sqrt(u_v**2 + v_v**2 + 1e-16)
dSdx_v = dSdx.vector().array()
dSdy_v = dSdy.vector().array()
gradS  = sqrt(dSdx_v**2 + dSdy_v**2 + 1e-16)
dBdx_v = dBdx.vector().array()
dBdy_v = dBdy.vector().array()
gradB  = sqrt(dBdx_v**2 + dBdy_v**2 + 1e-16)
D      = zeros(len(B_v))
D[B_v < 0] = B_v[B_v < 0]

taux = -917.0 * 9.8 * H_v * dSdx_v
tauy = -917.0 * 9.8 * H_v * dSdy_v
tau_mag = sqrt(taux**2 + tauy**2 + 1e-16)

uhat = u_v / U_mag
vhat = v_v / U_mag

dBdi = dBdx_v * uhat + dBdy_v * vhat
dBdj = dBdx_v * vhat - dBdy_v * uhat

dSdi = dSdx_v * uhat + dSdy_v * vhat
dSdj = dSdx_v * vhat - dSdy_v * uhat

dHdi = dHdx_v * uhat + dHdy_v * vhat
dHdj = dHdx_v * vhat - dHdy_v * uhat

Ubar_avg = (Ubar5_v + Ubar10_v + Ubar20_v) / 3.0

dBdi_f = Function(Q)
dBdj_f = Function(Q)
dBdi_f.vector().set_local(dBdi)
dBdi_f.vector().apply('insert')
dBdj_f.vector().set_local(dBdj)
dBdj_f.vector().apply('insert')

dSdi_f = Function(Q)
dSdj_f = Function(Q)
dSdi_f.vector().set_local(dSdi)
dSdi_f.vector().apply('insert')
dSdj_f.vector().set_local(dSdj)
dSdj_f.vector().apply('insert')

v0   = S_v
v1   = Ts_v
v2   = gradS
v3   = D
v4   = gradB
v5   = H_v
v6   = qgeo_v
v7   = adot_v
v8   = Tb_v
v9   = Mb_v
v10  = u_v
v11  = v_v
v12  = w_v
v13  = log(Ubar5_v + DOLFIN_EPS)
v14  = log(Ubar10_v + DOLFIN_EPS)
v15  = log(Ubar20_v + DOLFIN_EPS)
v16  = log(U_mag + DOLFIN_EPS)
v17  = tau_ii_v
v18  = tau_ij_v
v19  = tau_ik_v
v20  = tau_ji_v
v21  = tau_jj_v
v22  = tau_jk_v
v23  = ini_i
v24  = ini_j
v25  = dBdi
v26  = dBdj
v27  = dSdi
v28  = dSdj
v29  = gradH
v30  = dHdi
v31  = dHdj

names = [r'$S$', 
         r'$T_S$', 
         r'$\Vert \nabla S \Vert$', 
         r'$D$',
         r'$\Vert \nabla B \Vert$', 
         r'$H$',
         r'$q_{geo}$',
         r'$\dot{a}$',
         r'$T_B$', 
         r'$M_B$', 
         r'$u$', 
         r'$v$', 
         r'$w$', 
         r'$\ln\left( \Vert \bar{\mathbf{u}}_{5} \Vert \right)$',
         r'$\ln\left( \Vert \bar{\mathbf{u}}_{10} \Vert \right)$',
         r'$\ln\left( \Vert \bar{\mathbf{u}}_{20} \Vert \right)$',
         r'$\ln\left( \Vert \mathbf{u}_B \Vert \right)$',
         r'$\tau_{ii}$',
         r'$\tau_{ij}$',
         r'$\tau_{ik}$',
         r'$\tau_{ji}$',
         r'$\tau_{jj}$',
         r'$\tau_{jk}$',
         r'ini$_i$',
         r'ini$_j$',
         r'$\partial_i B$',
         r'$\partial_j B$',
         r'$\partial_i S$',
         r'$\partial_j S$',
         r'$\Vert \nabla H \Vert$',
         r'$\partial_i H$',
         r'$\partial_j H$']

V = [v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,
     v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31]

d2model.save_hdf5(d2model.N_ii, f=fout)
