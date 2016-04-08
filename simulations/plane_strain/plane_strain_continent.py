from cslvr    import *
from fenics   import *

l       = 400000.0
d       = 0.0
H       = 2000
p1      = Point(-l+d, 0.0)
p2      = Point( l-d, 1.0)
nx      = 1000
nz      = 20
mesh    = RectangleMesh(p1, p2, nx, nz)
out_dir = 'continent_results/'

model = LatModel(mesh, out_dir = out_dir)
model.generate_function_spaces(use_periodic = False)

S = Expression('H/2*cos(pi*x[0]/l) + 200 + H/2', l=l, H=H,
               element = model.Q.ufl_element())
B = Expression('25*cos(200*pi*x[0]/(2*l)) - 800', l=l,
               element = model.Q.ufl_element())

# surface temperature :
class SurfaceTemperature(Expression):
  Tmin = 238.15
  #St   = 1.67e-5
  St   = 5e-5
  def eval(self,values,x):
    values[0] = self.Tmin + self.St*abs(x[0])
T_s = SurfaceTemperature(element=model.Q.ufl_element())

model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=None)

model.init_mask(1.0)  # all grounded
model.init_beta(20.0)
model.init_T(T_s)
model.init_T_surface(T_s)
model.init_E(1.0)
model.init_Wc(0.03)
model.init_k_0(1e-3)
model.init_q_geo(model.ghf)
model.solve_hydrostatic_pressure()
model.form_energy_dependent_rate_factor()

mom = MomentumDukowiczPlaneStrain(model)
nrg = Enthalpy(model, mom)

#mom.solve()
#mom.calc_q_fric()
#nrg.solve()
#
#model.save_xdmf(model.p,  'p')
#model.save_xdmf(model.U3, 'U')
#model.save_xdmf(model.T,  'T')
#model.save_xdmf(model.W,  'W')
#
#fout    = HDF5File(mpi_comm_world(), out_dir + 'state.h5', 'w')
#model.save_hdf5(model.U3, f=fout)
#model.save_hdf5(model.p , f=fout)
#
#sys.exit(0)

# thermo-solve callback function :
def tmc_cb_ftn():
  nrg.calc_PE()#avg=True)
  nrg.calc_internal_water()
  nrg.calc_integrated_strain_heat()
  nrg.solve_basal_melt_rate()

# after every completed adjoining, save the state of these functions :
tmc_save_vars = [model.T,
                 model.W,
                 model.Fb,
                 model.Mb,
                 model.q_fric,
                 model.alpha,
                 model.PE,
                 model.W_int,
                 model.Q_int,
                 model.U3,
                 model.p,
                 model.beta,
                 model.theta]

# form the objective functional for water-flux optimization :
nrg.form_cost_ftn(kind='L2')

wop_kwargs = {'max_iter'            : 200,
              'bounds'              : (0.0, 100.0),
              'method'              : 'ipopt',
              'adj_callback'        : None}
                                    
tmc_kwargs = {'momentum'            : mom,
              'energy'              : nrg,
              'wop_kwargs'          : wop_kwargs,
              'callback'            : tmc_cb_ftn, 
              'atol'                : 1e2,
              'rtol'                : 1e0,
              'max_iter'            : 10,
              'itr_tmc_save_vars'   : tmc_save_vars,
              'post_tmc_save_vars'  : tmc_save_vars,
              'starting_i'          : 1}
                                    
# thermo_solve :
model.thermo_solve(**tmc_kwargs)

