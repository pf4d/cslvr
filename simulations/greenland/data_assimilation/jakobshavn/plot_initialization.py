from varglas        import *
from fenics         import *
import sys


# set the relavent directories :
dir_b   = 'dump/jakob_small/'
var_dir = 'dump/vars_jakobshavn_small/'       # directory from gen_vars.py
in_dir  = dir_b + 'initialization/hdf5/'      # input dir
out_dir = 'plot/initialization/'              # base directory to save

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',        'r')
fthermo = HDF5File(mpi_comm_world(), in_dir  + 'thermo_ini.h5',   'r')

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
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
d3model.init_beta(fthermo)
d3model.init_T(fthermo)
d3model.init_U(fthermo)
d3model.init_W(fthermo)
d3model.init_theta(fthermo)
d3model.init_Mb(fthermo)


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

# create a new 2D model for surface variables :
srfmodel = D2Model(srfmesh, out_dir)

# put the velocity on it :
d2model.assign_submesh_variable(srfmodel.U3,        d3model.U3)
d2model.assign_submesh_variable(srfmodel.U_mag,     d3model.U_mag)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

#===============================================================================
# collect the raw data :
drg  = DataFactory.get_rignot()

cmap = 'RdGy'

plotIce(drg, d2model.W, name='W', direc=out_dir,
        title='$W$', basin='jakobshavn',
        cmap=cmap,  scale='lin', umin=None, umax=0.1,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

plotIce(drg, d2model.Mb, name='Mb', direc=out_dir,
        title='$M_B$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=0.03, umax=2,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

plotIce(drg, srfmodel.U_mag, name='U', direc=out_dir,
        title=r'$\Vert \mathbf{u}_S \Vert$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

plotIce(drg, d2model.beta, name='beta_SIA', direc=out_dir,
        title=r'$\beta_{SIA}$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=5, umax=3e3,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

d2model.init_beta_stats(mdl='U', use_temp=False, mode='steady')

plotIce(drg, d2model.beta, name='betahat', direc=out_dir,
        title=r'$\hat{\beta}$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=5, umax=3e3,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)



