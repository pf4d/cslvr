from varglas import *
from varglas.helper import plotIce

thklim = 10.0
in_dir = 'dump/balance_velocity/'

# collect the raw data :
bamber  = DataFactory.get_bamber(thklim)

# load a mesh :
#mesh  = MeshFactory.get_greenland_2D_1H()
mesh  = Mesh('dump/meshes/greenland_2D_1H_mesh.xml.gz')

# create data objects to use with varglas :
dbm   = DataInput(bamber,  mesh=mesh)

model = D2Model(mesh)

model.init_Ubar(in_dir + 'Ubar.xml')

plotIce(dbm, model.Ubar, name='Ubar', direc=in_dir + 'plot/', 
        title=r'$\bar{\mathbf{u}}$', cmap='gist_yarg',
        scale='log', numLvls=12, tp=False, tpAlpha=0.5,
        umin=1.0, umax=4000, show=False)

#def plotIce(di, u, name, direc, title='', cmap='gist_yarg',  scale='lin',
#            umin=None, umax=None, numLvls=12, tp=False, tpAlpha=0.5,
#            extend='neither', show=True):


