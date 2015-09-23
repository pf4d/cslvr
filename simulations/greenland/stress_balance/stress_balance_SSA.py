from varglas import D2Model, SSA_Balance
from fenics  import *

in_dir  = 'dump/bed_vars/'
out_dir = 'dump/output/SSA/'

mesh  = Mesh(in_dir + 'submesh.xdmf')
Q     = FunctionSpace(mesh, 'CG', 1)

model = D2Model(out_dir = out_dir)
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = False)

model.init_S(in_dir + 'S_s.xml')
model.init_B(in_dir + 'B_s.xml')
model.init_beta(in_dir + 'beta_s.xml')
model.init_component_Ubar(in_dir + 'ubar_s.xml',
                          in_dir + 'vbar_s.xml',
                          0.0)
model.init_etabar(in_dir + 'etabar_s.xml')

F = SSA_Balance(model)
F.solve()

model.save_pvd(model.tau_id, 'tau_id')
model.save_pvd(model.tau_jd, 'tau_jd')
model.save_pvd(model.tau_ib, 'tau_ib')
model.save_pvd(model.tau_jb, 'tau_jb')
model.save_pvd(model.tau_ii, 'tau_ii')
model.save_pvd(model.tau_ij, 'tau_ij')
model.save_pvd(model.tau_ji, 'tau_ji')
model.save_pvd(model.tau_jj, 'tau_jj')
