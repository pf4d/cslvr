from pylab    import *
from varglas  import *
from varglas.energy           import EnergyHybrid
from varglas.mass             import MassHybrid
from varglas.momentumhybrid   import MomentumHybrid
from fenics   import *

#parameters['form_compiler']['quadrature_degree'] = 2
#parameters['form_compiler']['precision']         = 30
#parameters['form_compiler']['optimize']          = True
#parameters['form_compiler']['representation']    = 'quadrature'

thklim = 1.0

# collect the raw data :
searise = DataFactory.get_searise(thklim)
bamber  = DataFactory.get_bamber(thklim)

# load a mesh :
#mesh  = MeshFactory.get_greenland_2D_1H()
mesh  = Mesh('dump/meshes/greenland_2D_mesh.xml.gz')

# create data objects to use with varglas :
dsr   = DataInput(searise, mesh=mesh)
dbm   = DataInput(bamber,  mesh=mesh)

# create a mask for the continent :
M             = dbm.data['mask_orig']
m1            = M == 1
m2            = M == 2
mask          = logical_or(m1,m2)
dbm.data['M'] = mask

# enforce max depth in regions off continent :
B  = dbm.data['Bo']
Bm = logical_and(B < -500, mask != 1)
dbm.data['Bo'][Bm] = -500

#B  = dbm.data['Bo']
#Bm = B[B != -9999].min()
#dbm.set_data_min('Bo', boundary=Bm, val=Bm)
 
# interpolate from the dbm grid to the dsr grid and make areas of ocean very
# high negative accumulation to simulate a `calving frount' :
dbm.interpolate_to_di(dsr, fn='M', fo='M')
dsr.data['adot'][dsr.data['M'] == 0] = -10.0

# get the data :
B     = dbm.get_expression("Bo",    near=False)
adot  = dsr.get_expression("adot",  near=False)
lat   = dsr.get_expression("lat",   near=False)
lon   = dsr.get_expression("lon",   near=False)

# create a 2D model :
model = D2Model(mesh, out_dir = 'dump/results/')

model.init_B(B)
model.init_S(model.B.vector() + thklim)
model.init_mask(1.0)
model.init_adot(adot)
model.init_lat(lat)
model.init_lon(lon)
model.init_beta(1e9)
model.init_H_bounds(thklim, 1e4)
model.init_H_H0(thklim)
model.init_q_geo(model.ghf)
model.assign_variable(model.eps_reg, 1e-5)

#===============================================================================
# initialize transient experiment physics :

#bv  = BalanceVelocity(model, kappa=5.0)
nrg = EnergyHybrid(model, transient=True)
mom = MomentumHybrid(model, isothermal=False)
mas = MassHybrid(model, isothermal=False)

# initialize temperature from lapse rate :
nrg.solve_surface_climate()
model.init_T_T0(model.T_surface)

model.save_xdmf(model.T_surface, 'T_surface')
model.save_xdmf(model.S,         'S')

def cb_ftn():
  #bv.solve(annotate=False)
  model.save_xdmf(model.S,  'S')
  model.save_xdmf(model.H,  'H')
  model.save_xdmf(model.Ts, 'Ts')
  model.save_xdmf(model.Tb, 'Tb')
  model.save_xdmf(model.Mb, 'Mb')
  model.save_xdmf(model.U3, 'Us')
  #model.save_xdmf(model.beta, 'beta')
  nrg.solve_surface_climate()

model.transient_solve(mom, nrg, mas,
                      t_start=0.0, t_end=35000.0, time_step=25.0,
                      adaptive=True, annotate=False, callback=cb_ftn)



