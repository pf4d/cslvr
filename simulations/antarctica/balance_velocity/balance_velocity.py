from varglas import *

thklim = 10.0

# collect the raw data :
bm1   = DataFactory.get_bedmap1(thklim)
bm2   = DataFactory.get_bedmap2(thklim)

# load a mesh :
mesh  = MeshFactory.get_antarctica_2D_medium()

# create data objects to use with varglas :
d1    = DataInput(bm1, mesh=mesh)
d2    = DataInput(bm2, mesh=mesh)

# get projections for use with FEniCS :
S     = d2.get_expression("S",     near=True)
B     = d2.get_expression("B",     near=True)
adot  = d1.get_expression("acca",  near=True)

model = D2Model(out_dir = 'results/')
model.set_mesh(mesh)
model.generate_function_spaces()

model.init_S(S)
model.init_B(B)
model.init_adot(adot)

bv = BalanceVelocity(model, kappa=5.0)
bv.solve(annotate=False)

model.save_pvd(model.Ubar, 'Ubar')
model.save_xml(model.Ubar, 'Ubar')

#do = DataOutput(out_dir)
#do.write_matlab(bm1, model.Ubar, 'Ubar_5', val=0.0)



