from cslvr        import *
from cslvr.helper import plotIce


# collect the raw data :
searise = DataFactory.get_searise()
bamber  = DataFactory.get_bamber()
rignot  = DataFactory.get_rignot()
measure = DataFactory.get_gre_measures()

bamber['Bo'][bamber['Bo'] == -9999] = 0.0

M = bamber['mask_orig']
m1 = M == 1
m2 = M == 2
mask = logical_or(m1,m2)


# create data objects to use with varglas :
dsr   = DataInput(searise, gen_space=False)
dbm   = DataInput(bamber,  gen_space=False)
drg   = DataInput(rignot,  gen_space=False)

dbm.data['M'] = mask

plotIce(dbm, 'M', name='M', direc='greenland/', title='$M$',
        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
        numLvls=3, tp=False, tpAlpha=0.5, extend='neither', show=False)

#plotIce(dsr, 'S_ring', name='S_ring', direc='greenland/', title='$\dot{a}$',
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
#
#plotIce(dbm, 'Bo', name='B', direc='greenland/', title='$B$',
#        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
#        numLvls=24, tp=False, tpAlpha=0.5, extend='neither', show=True)
#
#plotIce(dbm, 'mask', name='mask', direc='greenland/', title='$M$',
#        cmap='gist_yarg',  scale='lin', umin=None, umax=None,
#        numLvls=4, tp=False, tpAlpha=0.5, extend='neither', show=True)
