from varglas        import *
from varglas.helper import plotIce


# collect the raw data :
searise = DataFactory.get_searise()
bamber  = DataFactory.get_bamber()
rignot  = DataFactory.get_rignot()

bamber['Bo'][bamber['Bo'] == -9999] = 0.0


# create data objects to use with varglas :
dsr   = DataInput(searise, gen_space=False)
dbm   = DataInput(bamber,  gen_space=False)
drg   = DataInput(rignot,  gen_space=False)

#plotIce(dsr, 'adot', name='adot', direc='greenland/', title='$\dot{a}$',
#        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=True)
#
#plotIce(dsr, 'T', name='T', direc='greenland/', title='$T$',
#        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=True)
#
#plotIce(dsr, 'Tn', name='Tn', direc='greenland/', title='$T_n$',
#        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
#        numLvls=12, tp=False, tpAlpha=0.5, extend='neither', show=True)

plotIce(dbm, 'Bo', name='B', direc='greenland/', title='$B$',
        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
        numLvls=24, tp=False, tpAlpha=0.5, extend='neither', show=True)
#
#plotIce(dbm, 'mask', name='mask', direc='greenland/', title='$M$',
#        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
#        numLvls=4, tp=False, tpAlpha=0.5, extend='neither', show=True)
