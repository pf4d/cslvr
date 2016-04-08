from cslvr    import *
from fenics   import *

l         = 10000.0
d         = 1500.0
b         = 1000.0
p1        = Point(-l+d, 0.0)
pm        = Point( b,   0.0)
p2        = Point( l-d, 1.0)
nx        = 1000
nx_b      = int(nx * (l/2 - b)/l)  # same resolution for basin mesh
nz        = 20
cont_mesh = RectangleMesh(p1, p2, nx,   nz)
basn_mesh = RectangleMesh(pm, p2, nx_b, nz)
out_dir   = 'basin_results/'

cont_model = LatModel(cont_mesh, out_dir = out_dir)
cont_model.generate_function_spaces(use_periodic = False)

basn_model = LatModel(basn_mesh, out_dir = out_dir)
basn_model.generate_function_spaces(use_periodic = False)

S = Expression('1000*cos(pi*x[0]/(2*l)) - 150', l=l,
               element = cont_model.Q.ufl_element())
B = Expression('25*cos(200*pi*x[0]/(2*l)) - 150', l=l,
               element = cont_model.Q.ufl_element())
class LatMask(Expression):
  def eval(self,values,x):
    if x[0] > 7000:
      values[0] = 1
    else:
      values[0] = 0
latMask = LatMask(element = cont_model.Q.ufl_element())

cont_model.deform_mesh_to_geometry(S, B)
basn_model.deform_mesh_to_geometry(S, B)

basn_model.calculate_boundaries(lat_mask=latMask, mark_divide=True)


fin    = HDF5File(mpi_comm_world(), 'continent_results/state.h5', 'r')
cont_model.init_U(fin)
cont_model.init_p(fin)
basn_model.assign_submesh_variable(basn_model.U3, cont_model.U3)
basn_model.assign_submesh_variable(basn_model.p,  cont_model.p)

basn_model.init_mask(1.0)  # all grounded
basn_model.init_beta(100.0)
basn_model.init_Tp(268.0)
basn_model.init_E(1.0)
basn_model.form_energy_dependent_rate_factor()

mom = MomentumDukowiczPlaneStrain(basn_model, use_lat_bcs=True)
mom.solve()

basn_model.save_xdmf(basn_model.p,  'p')
basn_model.save_xdmf(basn_model.U3, 'U')

