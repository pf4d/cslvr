"""

MISMIP+ model intercomparison:

Written by Evan Cummings in his free time during July, 2018.

"""
from cslvr import *

# directories for loading or saving data :
mdl_odr = 'BP'     # the order of the momentum model
name    = 'first'      # the name of the sim
plt_dir = './images/' + mdl_odr + '/' + name + '/'
out_dir = './results/' + mdl_odr + '/'

#===============================================================================
# define the geometry of the simulation :
Lx     =  640000.0    # [m] domain length (along ice flow)
Ly     =  80000.0     # [m] domain width (across ice flow)
dx     =  10000.0     # [m] element diameter 
nx     =  int(Lx/dx)  # [--] number of x-coordinate divisions
ny     =  int(Ly/dx)  # [--] number of y-coordinate divisions
nz     =  5           # [--] number of z-coordinate divisions
B0     = -150.0       # [m] bedrock topography at x = 0
B2     = -728.8       # [m] second bedrock topography coefficient
B4     =  343.91      # [m] third bedrock topography coefficient
B6     = -50.57       # [m] second bedrock topography coefficient
xbar   =  300000.0    # [m] characteristic along-flow length scale of bedrock
fc     =  4000.0      # [m] characteristic width of channel walls
dc     =  500.0       # [m] depth of the trough compared to its walls
wc     =  24000.0     # [m] half width of the trough
zd     = -720.0       # [m] maximum depth of the bedrock topography
thklim =  10.0        # [m] thickness limit
rhow   =  1028.0      # [kg m^-3] density of seawater
rhoi   =  910.0       # [kg m^-3] density of glacier ice
g      =  9.81        # [m s^2] gravitational acceleration
spy    =  31556926.0  # [s a^-1] seconds per year
Hini   =  100.0       # [m] initial ice thickness
Tm     =  273.15      # [K] melting temperature of ice
A      =  2e-17       # [Pa^{-n} s^{-1}] flow 
beta   =  1e8         # [Pa m^{-1/n} a^{-1/n}] friction coefficient
adot   =  0.3         # [m a^{-a}] surface-mass balance
tf     =  200.0       # [a] final time
dt     =  100.0         # [a] time step
dt_sav =  1.0         # [a] time interval to save data
cfl    =  0.5         # [--] CFL coefficient

thklim  = 1.0          # [m] thickness limit
L       = 800000.0     # [m] mesh radius
Rel     = 450000       # [m] radial distance at which adot becomes negative
s       = 1e-5         # [a^{-1}] accumulation/ablation coefficient
Tmin    = 238.15       # [K] minimum temperature (located at divide)
St      = 1.67e-5      # [K m^{-1}] lapse rate

# create a genreic box mesh, we'll fit it to geometry below :
p1    = Point(0.0, 0.0, 0.0)          # origin
p2    = Point(Lx,  Ly,  1)            # x, y, z corner 
mesh  = BoxMesh(p1, p2, nx, ny, nz)   # a box to fill the void 

# initialize the model :
model    = D3Model(mesh, out_dir=out_dir, use_periodic=False)

# form the 2D upper-surface mesh :
model.form_srf_mesh()

# form a 2D model using the upper-surface mesh :
srfmodel = D2Model(model.srfmesh, 
                   out_dir      = out_dir,
                   use_periodic = False,
                   kind         = 'submesh')

# generate the map between the 3D and 2D models : 
model.generate_submesh_to_mesh_map(sub_model=srfmodel)

# the MISMIP+ experiment lower topography :
class Bed(Expression):
  def eval(self, values, x):
    xt        = x[0] / xbar
    Bx        = B0 + B2*xt**2 + B4*xt**4 + B6*xt**6
    By        = + dc / (1 + exp(-2*(x[1] - Ly/2 - wc) / fc)) \
                + dc / (1 + exp( 2*(x[1] - Ly/2 + wc) / fc))
    values[0] = max(Bx + By, zd)
bed = Bed(element=model.Q.ufl_element())

# the MISMIP+ experiment upper surface :
class UpperSurface(Expression):
  def eval(self, values, x):
    S  = bed(x) + Hini                        # upper surface without flotation
    ls = Hini + rhow / rhoi * bed(x)          # grounded ice level-set flotation
    if ls <= 0:  S = Hini * (1 - rhoi / rhow) # correct for flotation
    values[0] = S
S = UpperSurface(element=model.Q.ufl_element())

# the MISMIP+ experiment lower surface :
class LowerSurface(Expression):
  def eval(self, values, x):
    values[0] = S(x) - Hini
B = LowerSurface(element=model.Q.ufl_element())

# the MISMIP+ ice-shelf mask :
class Mask(Expression):
  def eval(self, values, x):
    ls = Hini + rhow / rhoi * bed(x)          # grounded ice level-set flotation
    if   ls <= 0: values[0] = 2               # floating ice
    elif ls > 0:  values[0] = 1               # grounded ice
mask = Mask(element=model.Q.ufl_element())

# define where the ice is in contact with the ocean :
ice_front_x = Lx * np.ones(1000)
ice_front_y = np.linspace(0.0, Ly, 1000)
ice_front   = np.vstack((ice_front_x, ice_front_y)).T

# deform the mesh to match our desired geometry :
model.deform_mesh_to_geometry(S, B)

# mark the exterior facets and interior cells appropriately :
model.calculate_boundaries(mask        = mask,
                           adot        = adot,
                           U_mask      = None,
                           mark_divide = True,
                           contour     = ice_front)

# initialize the 3D model variables :
#model.init_T_surface(Tm)        # upper surface temperature
#model.init_T(Tm)                # initial 3D temperature
#model.init_q_geo(model.ghf)     # geothermal heat flux
#model.init_E(1.0)               # flow enhancement (1.0 == no enhancement)
#model.init_W(0.0)               # water content (0.0 == no water)
#model.init_k_0(1e-3)            # water "non-advective" coefficient (enthalpy)
model.init_beta(beta)           # basal fricition coefficient
#model.init_beta_stats()     
model.init_A(A)                 # constant flow-rate factor
#model.solve_hydrostatic_pressure()
#model.form_energy_dependent_rate_factor()

# update the 2D model variables that we'll need to compute the mass balance :
model.assign_to_submesh_variable(u = model.S,      u_sub = srfmodel.S)
model.assign_to_submesh_variable(u = model.B,      u_sub = srfmodel.B)
model.assign_to_submesh_variable(u = model.adot,   u_sub = srfmodel.adot)

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
  mom = MomentumBP(model, use_pressure_bc=True)
elif mdl_odr == 'BP_duk':
  mom = MomentumDukowiczBP(model, use_pressure_bc=True)
elif mdl_odr == 'RS':
  mom = MomentumDukowiczStokesReduced(model, use_pressure_bc=True)
elif mdl_odr == 'FS_duk':
  mom = MomentumDukowiczStokes(model, use_pressure_bc=True)
elif mdl_odr == 'FS_stab':
  mom = MomentumNitscheStokes(model, use_pressure_bc=True, stabilized=True)
elif mdl_odr == 'FS_th':
  mom = MomentumNitscheStokes(model, use_pressure_bc=True, stabilized=False)

#nrg = Enthalpy(model, mom,
#               transient  = True,
#               use_lat_bc = False)
mass = FreeSurface(srfmodel,
                   thklim              = thklim,
                   lump_mass_matrix    = False)


# create a function to be called at the end of each iteration :
U_file  = XDMFFile(out_dir + 'U.xdmf')
S_file  = XDMFFile(out_dir + 'S.xdmf')
def cb_ftn():
  model.save_xdmf(model.U3, 'U3', U_file)
  model.save_xdmf(model.S,  'S',  S_file)

# run the transient simulation :
model.transient_solve(mom, mass,
                      t_start    = 0.0,
                      t_end      = tf,
                      time_step  = dt,
                      tmc_kwargs = None,
                      adaptive   = False,
                      annotate   = False,
                      callback   = cb_ftn)



