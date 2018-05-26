from cslvr import *

msh_dir = './meshes/'
out_dir = './A/'
mdl_odr = 'BP'

thklim  = 1.0
L       = 800000.
xmin    = -L
xmax    =  L
ymin    = -L
ymax    =  L

mesh    = Mesh(msh_dir + 'cylinder_mesh.xml.gz')

# width and origin of the domain for deforming x coord :
width_x  = xmax - xmin
offset_x = xmin

# width and origin of the domain for deforming y coord :
width_y  = ymax - ymin
offset_y = ymin
for x in mesh.coordinates():
  # transform x :
  x[0]  = x[0]  * width_x
  # transform y :
  x[1]  = x[1]  * width_y

# initialize the model :
model = D3Model(mesh, out_dir=out_dir, use_periodic=False)
model.deform_mesh_to_geometry(S=thklim, B=0.0)
model.calculate_boundaries()

# surface mass balance : 
class Adot(Expression):
  Rel = 450000
  s   = 1e-5
  def eval(self,values,x):
    #values[0] = 0.3
    values[0] = min(0.5,self.s*(self.Rel-sqrt(x[0]**2 + x[1]**2)))
adot = Adot(element=model.Q.ufl_element())

# surface temperature :
class SurfaceTemperature(Expression):
  Tmin = 238.15
  St   = 1.67e-5
  def eval(self,values,x):
    values[0] = self.Tmin + self.St*sqrt(x[0]**2 + x[1]**2)
T_s = SurfaceTemperature(element=model.Q.ufl_element())

# initialize the 3D model vars :
model.init_adot(adot)
model.init_T_surface(T_s)
model.init_T(T_s)
model.init_q_geo(model.ghf)
model.init_time_step(1e-6)
model.init_E(1.0)
model.init_W(0.0)
model.init_Wc(0.03)
model.init_k_0(1e-3)
model.init_beta(1e9)
#model.init_beta_stats()
model.init_A(1e-16)
#model.solve_hydrostatic_pressure()
#model.form_energy_dependent_rate_factor()

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
  mom = MomentumBP(model)
elif mdl_odr == 'BP_duk':
  mom = MomentumDukowiczBP(model)
elif mdl_odr == 'RS':
  mom = MomentumDukowiczStokesReduced(model)
elif mdl_odr == 'FS_duk':
  mom = MomentumDukowiczStokes(model)
elif mdl_odr == 'FS_stab':
  mom = MomentumNitscheStokes(model, stabilized=True)
elif mdl_odr == 'FS_th':
  mom = MomentumNitscheStokes(model, stabilized=False)

#nrg = Enthalpy(model, mom,
#               transient  = True,
#               use_lat_bc = False)
mas = FreeSurface(model,
                  thklim              = 1.0,
                  use_shock_capturing = False,
                  lump_mass_matrix    = False)

U_file  = XDMFFile(out_dir + 'U.xdmf')
S_file  = XDMFFile(out_dir + 'S.xdmf')
#T_file  = XDMFFile(out_dir + 'T.xdmf')
def cb_ftn():
  #nrg.solve()
  model.save_xdmf(model.U3, 'U3', U_file)
  #model.save_xdmf(model.T,  'T',  T_file)
  model.save_xdmf(model.S,  'S',  S_file)

model.transient_solve(mom, mas,
                      t_start    = 0.0,
                      t_end      = 1000.0,
                      time_step  = 100.0,
                      tmc_kwargs = None,
                      adaptive   = True,
                      annotate   = False,
                      callback   = cb_ftn)



