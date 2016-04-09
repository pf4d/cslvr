from cslvr    import *
from fenics   import *

l       = 200000.0
d       = 0.0
Hmax    = 4000
S0      = 100
B0      = -200
Bmin    = -400
Tmin    = 228.15
betaMax = 8000.0
betaMin = 1.0
sig     = l/4.0
St      = 6.5 / 1000.0
p1      = Point(-l+d, 0.0)
p2      = Point( l-d, 1.0)
nx      = 1000
nz      = 20
mesh    = RectangleMesh(p1, p2, nx, nz)
out_dir = 'continent_results/'

model = LatModel(mesh, out_dir = out_dir, use_periodic = False)

#def gauss(x, sig):
#  return exp(-((x/(2*sig))**2))
#
#class Surface(Expression):
#  def eval(self,values,x):
#    values[0] = S0 + (Hmax + Bmin - S0)*gauss(x[0], sig)
#S = Surface(element = model.Q.ufl_element())
#
#class Bed(Expression):
#  def eval(self,values,x):
#    values[0] = B0 + (Bmin - B0)*gauss(x[0], sig)
#B = Bed(element = model.Q.ufl_element())
#
#class Beta(Expression):
#  def eval(self, values, x):
#    values[0] = betaMin + (betaMax - betaMin) * gauss(x[0], sig)
#b = Beta(element = model.Q.ufl_element())

T = Expression('Tmin + St*(Hmax + B0 - S0 - x[1])',
               Tmin=Tmin, Hmax=Hmax, B0=B0, S0=S0, St=St,
               element = model.Q.ufl_element())
S = Expression('(Hmax+B0-S0)/2*cos(pi*x[0]/l) + (Hmax+B0+S0)/2',
               Hmax=Hmax, B0=B0, S0=S0, l=l,
               element = model.Q.ufl_element())
B = Expression('10*cos(200*pi*x[0]/(2*l)) + B0', l=l, B0=B0,
               element = model.Q.ufl_element())
#b = Expression('betaMax - sqrt(pow(x[0],2)) * (betaMax - betaMin)/l',
#               betaMax=betaMax, betaMin=betaMin, l=l,
#               element = model.Q.ufl_element())
b = Expression('(bMax - bMin)/2.0*cos(pi*x[0]/l) + (bMax + bMin)/2.0',
               bMax=betaMax, bMin=betaMin, l=l,
               element = model.Q.ufl_element())

model.deform_mesh_to_geometry(S, B)
model.calculate_boundaries(mask=None)

model.init_mask(1.0)  # all grounded
model.init_beta(b)
model.init_T(T)
model.init_T_surface(T)
model.init_E(1.0)
model.init_Wc(0.03)
model.init_k_0(1e-3)
model.init_q_geo(model.ghf)
model.solve_hydrostatic_pressure()
model.form_energy_dependent_rate_factor()

mom = MomentumDukowiczPlaneStrain(model)
nrg = Enthalpy(model, mom, energy_flux_mode = 'Fb')

#mom.solve()
#mom.calc_q_fric()
#nrg.derive_temperate_zone()
##nrg.update_thermal_parameters()
#nrg.solve()
#
#model.save_xdmf(model.beta, 'beta')
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

wop_kwargs = {'max_iter'            : 15,
              'bounds'              : (0.0, 100.0),
              'method'              : 'ipopt',
              'adj_callback'        : None}
                                    
tmc_kwargs = {'momentum'            : mom,
              'energy'              : nrg,
              'wop_kwargs'          : wop_kwargs,
              'callback'            : tmc_cb_ftn, 
              'atol'                : 1e2,
              'rtol'                : 1e0,
              'max_iter'            : 1,
              'itr_tmc_save_vars'   : None,#tmc_save_vars,
              'post_tmc_save_vars'  : tmc_save_vars,
              'starting_i'          : 1}
                                    
# thermo_solve :
model.thermo_solve(**tmc_kwargs)

model.save_xdmf(model.beta, 'beta')
model.save_xdmf(model.p,  'p')
model.save_xdmf(model.Q_int, 'Q_int')
model.save_xdmf(model.U3, 'U')
model.save_xdmf(model.T,  'T')
model.save_xdmf(model.W,  'W')
model.save_xdmf(model.Fb,  'Fb')
model.save_xdmf(model.Mb,  'Mb')
