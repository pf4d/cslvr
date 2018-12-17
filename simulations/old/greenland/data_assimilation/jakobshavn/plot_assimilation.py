from varglas        import *
from fenics         import *
import sys


# set the relavent directories :
dir_b   = 'dump/jakob_small/'
var_dir = 'dump/vars_jakobshavn_small/'       # directory from gen_vars.py
in_dir  = dir_b + '01/hdf5/'                  # input dir
out_dir = 'plot/01/'                          # base directory to save

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',       'r')
fthermo = HDF5File(mpi_comm_world(), in_dir  + 'thermo_01.h5',   'r')
finv    = HDF5File(mpi_comm_world(), in_dir  + 'inverted_01.h5', 'r')
fstress = HDF5File(mpi_comm_world(), in_dir  + 'stress_01.h5',   'r')

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_T_surface(fdata)
d3model.init_S_ring(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
d3model.init_beta(finv)
d3model.init_U(finv)
d3model.init_T(fthermo)
d3model.init_W(fthermo)
d3model.init_theta(fthermo)
d3model.init_Mb(fthermo)
d3model.init_tau_id(fstress)
d3model.init_tau_jd(fstress)
d3model.init_tau_ii(fstress)
d3model.init_tau_ij(fstress)
d3model.init_tau_iz(fstress)
d3model.init_tau_ji(fstress)
d3model.init_tau_jj(fstress)
d3model.init_tau_jz(fstress)


#===============================================================================
# retrieve the bed mesh :
bedmesh = d3model.get_bed_mesh()
srfmesh = d3model.get_surface_mesh()

# create 2D model for balance velocity :
d2model = D2Model(bedmesh, out_dir)

# 2D model gets balance-velocity appropriate variables initialized :
d2model.assign_submesh_variable(d2model.S,         d3model.S)
d2model.assign_submesh_variable(d2model.B,         d3model.B)
d2model.assign_submesh_variable(d2model.T_surface, d3model.T_surface)
d2model.assign_submesh_variable(d2model.S_ring,      d3model.S_ring)
d2model.assign_submesh_variable(d2model.u_ob,      d3model.u_ob)
d2model.assign_submesh_variable(d2model.v_ob,      d3model.v_ob)
d2model.assign_submesh_variable(d2model.U_ob,      d3model.U_ob)
d2model.assign_submesh_variable(d2model.beta,      d3model.beta)
d2model.assign_submesh_variable(d2model.u,        d3model.u)
d2model.assign_submesh_variable(d2model.u_mag,     d3model.u_mag)
d2model.assign_submesh_variable(d2model.T,         d3model.T)
d2model.assign_submesh_variable(d2model.W,         d3model.W)
d2model.assign_submesh_variable(d2model.Mb,        d3model.Mb)
d2model.assign_submesh_variable(d2model.tau_id,    d3model.tau_id)
d2model.assign_submesh_variable(d2model.tau_jd,    d3model.tau_jd)
d2model.assign_submesh_variable(d2model.tau_ii,    d3model.tau_ii)
d2model.assign_submesh_variable(d2model.tau_ij,    d3model.tau_ij)
d2model.assign_submesh_variable(d2model.tau_iz,    d3model.tau_iz)
d2model.assign_submesh_variable(d2model.tau_ji,    d3model.tau_ji)
d2model.assign_submesh_variable(d2model.tau_jj,    d3model.tau_jj)
d2model.assign_submesh_variable(d2model.tau_jz,    d3model.tau_jz)

# create a new 2D model for surface variables :
srfmodel = D2Model(srfmesh, out_dir)

# put the velocity on it :
d2model.assign_submesh_variable(srfmodel.u,        d3model.u)
d2model.assign_submesh_variable(srfmodel.u_mag,     d3model.u_mag)


#===============================================================================
# collect the raw data :
drg  = DataFactory.get_rignot()

cmap = 'RdGy'

#===============================================================================
# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=10.0)
bv.solve(annotate=False)

d2model.Ubar10 = d2model.Ubar
d2model.Ubar10.rename('Ubar10', '')

plotIce(drg, d2model.Ubar10, name='Ubar_10', direc=out_dir,
        title=r'$\Vert \mathbf{\bar{u}}_{10} \Vert$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=20.0)
bv.solve(annotate=False)

d2model.Ubar20 = d2model.Ubar
d2model.Ubar20.rename('Ubar20', '')

plotIce(drg, d2model.Ubar20, name='Ubar_20', direc=out_dir,
        title=r'$\Vert \mathbf{\bar{u}}_{20} \Vert$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.Ubar5 = d2model.Ubar
d2model.Ubar5.rename('Ubar5', '')

plotIce(drg, d2model.Ubar5, name='Ubar_5', direc=out_dir,
        title=r'$\Vert \mathbf{\bar{u}}_5 \Vert$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)


##===============================================================================
# plot :
#
#plotIce(drg, d2model.tau_id, name='tau_id', direc=out_dir,
#        title=r'$\tau_{id}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-3e5, umax=3e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_jd, name='tau_jd', direc=out_dir,
#        title=r'$\tau_{jd}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-3e5, umax=3e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_ii, name='tau_ii', direc=out_dir,
#        title=r'$\tau_{ii}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-1e5, umax=1e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_iz, name='tau_iz', direc=out_dir,
#        title=r'$\tau_{iz}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-3e5, umax=3e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_ij, name='tau_ij', direc=out_dir,
#        title=r'$\tau_{ij}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-1e5, umax=1e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_ji, name='tau_ji', direc=out_dir,
#        title=r'$\tau_{ji}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-1e5, umax=1e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_jj, name='tau_jj', direc=out_dir,
#        title=r'$\tau_{jj}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-1e5, umax=1e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.tau_jz, name='tau_jz', direc=out_dir,
#        title=r'$\tau_{jz}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=-3e5, umax=3e5,
#        numLvls=13, tp=False, tpAlpha=0.5, extend='both', show=False)
#
#plotIce(drg, d2model.T_surface, name='T_S', direc=out_dir,
#        title='$T_S$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=None, umax=None,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#plotIce(drg, d2model.S_ring, name='S_ring', direc=out_dir,
#        title='$\dot{a}$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=None, umax=None,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#plotIce(drg, d2model.W, name='W', direc=out_dir,
#        title='$W$', basin='jakobshavn',
#        cmap=cmap,  scale='lin', umin=None, umax=0.15,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#plotIce(drg, d2model.Mb, name='Mb', direc=out_dir,
#        title='$M_B$', basin='jakobshavn',
#        cmap=cmap,  scale='log', umin=0.03, umax=4,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#plotIce(drg, d2model.U_ob, name='U_ob', direc=out_dir,
#        title=r'$\Vert \mathbf{u}_{ob} \Vert$', basin='jakobshavn',
#        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#plotIce(drg, srfmodel.u_mag, name='U', direc=out_dir,
#        title=r'$\Vert \mathbf{u}_S \Vert$', basin='jakobshavn',
#        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#plotIce(drg, d2model.beta, name='beta_opt', direc=out_dir,
#        title=r'$\beta$', basin='jakobshavn',
#        cmap=cmap,  scale='log', umin=1e-4, umax=1e4,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#d2model.init_beta_stats(mdl='U', use_temp=False, mode='steady')
#
#plotIce(drg, d2model.beta, name='betahat_U', direc=out_dir,
#        title=r'$\hat{\beta_{\mathbf{u_B}}}$', basin='jakobshavn',
#        cmap=cmap,  scale='log', umin=1e-4, umax=1e4,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)
#
#d2model.init_beta_stats(mdl='stress', use_temp=False, mode='steady')
#
#plotIce(drg, d2model.beta, name='betahat_tau', direc=out_dir,
#        title=r'$\hat{\beta_{\tau}}$', basin='jakobshavn',
#        cmap=cmap,  scale='log', umin=1e-4, umax=1e4,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)



