from varglas        import *
from fenics         import *
import sys

i = sys.argv[1]


# set the relavent directories :
dir_b   = 'dump/jakob_small/'
var_dir = 'dump/vars_jakobshavn_small/'    # directory from gen_vars.py
in_dir  = dir_b + i + '/hdf5/'             # input dir
out_dir = 'plot/' + i + '/'                # base directory to save

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)


#===============================================================================
# retrieve the bed mesh :
bedmesh = d3model.get_bed_mesh()
srfmesh = d3model.get_surface_mesh()

# create 2D model for balance velocity :
d2model = D2Model(bedmesh, out_dir)


#===============================================================================
# create HDF5 files for saving and loading data :
fthermo = HDF5File(mpi_comm_world(), in_dir  + 'thermo_' + i + '.h5',   'r')
finv    = HDF5File(mpi_comm_world(), in_dir  + 'inverted_' + i + '.h5', 'r')

# initialize the variables :
d3model.init_beta(finv)
d3model.init_U(finv)
d3model.init_T(fthermo)
d3model.init_W(fthermo)
d3model.init_Mb(fthermo)

# 2D model gets balance-velocity appropriate variables initialized :
d2model.assign_submesh_variable(d2model.beta,      d3model.beta)
d2model.assign_submesh_variable(d2model.T,         d3model.T)
d2model.assign_submesh_variable(d2model.W,         d3model.W)
d2model.assign_submesh_variable(d2model.Mb,        d3model.Mb)

# create a new 2D model for surface variables :
srfmodel = D2Model(srfmesh, out_dir)

# put the velocity on it :
d2model.assign_submesh_variable(srfmodel.U_mag,    d3model.U_mag)


#===============================================================================
# collect the raw data :
drg  = DataFactory.get_rignot()

cmap = 'RdGy'

#===============================================================================
# plot :

plotIce(drg, d2model.W, name='W', direc=out_dir,
        title='$W$', basin='jakobshavn',
        cmap=cmap,  scale='lin', umin=None, umax=0.15,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

plotIce(drg, d2model.Mb, name='Mb', direc=out_dir,
        title='$M_B$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=0.03, umax=4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

plotIce(drg, srfmodel.U_mag, name='U', direc=out_dir,
        title=r'$\Vert \mathbf{u}_S \Vert$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=30.0, umax=1e4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)

plotIce(drg, d2model.beta, name='beta_opt', direc=out_dir,
        title=r'$\beta$', basin='jakobshavn',
        cmap=cmap,  scale='log', umin=1e-4, umax=1e4,
        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=False)



