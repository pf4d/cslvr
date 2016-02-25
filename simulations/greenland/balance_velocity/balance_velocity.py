from cslvr import *

thklim = 10.0

# collect the raw data :
searise = DataFactory.get_searise(thklim)
bamber  = DataFactory.get_bamber(thklim)
rignot  = DataFactory.get_rignot()

# load a mesh :
mesh  = MeshFactory.get_greenland_2D_1H()

# create data objects to use with varglas :
dsr   = DataInput(searise, mesh=mesh)
dbm   = DataInput(bamber,  mesh=mesh)
drg   = DataInput(rignot,  mesh=mesh)

# the mesh is in Bamber coordinates, so transform :
drg.change_projection(dbm)

#plotIce(dsr, 'adot', name='', direc='.', title=r'$\dot{a}$', cmap='gist_yarg',
#        scale='lin', numLvls=12, tp=False, tpAlpha=0.5)

B     = dbm.get_expression("B",     near=False)
S     = dbm.get_expression("S",     near=False)
adot  = dsr.get_expression("adot",  near=False)

model = D2Model(mesh, out_dir = 'results/')

model.init_S(S)
model.init_B(B)
model.init_adot(adot)

bv = BalanceVelocity(model, kappa=5.0)
bv.solve(annotate=False)

model.save_xdmf(model.Ubar, 'Ubar')

plotIce(dbm, model.Ubar, name='Ubar', direc='results/',
       title=r'$\bar{\mathbf{u}}$', cmap='gist_yarg',
       umin=1.5, umax=4000,
       scale='log', numLvls=12, tp=False, tpAlpha=0.5)

#do = DataOutput(out_dir)
#do.write_matlab(bm1, model.Ubar, 'Ubar_5', val=0.0)



