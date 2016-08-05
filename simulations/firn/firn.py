from fenics          import *
from dolfin_adjoint  import *
from cslvr           import *
import sys
import pylab as p
    
n     =  100                   # num of z-positions
zs    =  0.0                   # surface start .................. m
zb    = -100.0                 # depth .......................... m

mesh  = IntervalMesh(100, zb, zs)      # interval from bed to surface

model = D1Model(mesh, out_dir = 'results')

model.refine_mesh(divs=2, i=1/4.0, k=1/5.)
model.refine_mesh(divs=2, i=1/4.0, k=1/5.)
model.refine_mesh(divs=2, i=1/4.0, k=1/5.)
#model.refine_mesh(divs=2, i=1/4.0, k=1/5.)
#model.refine_mesh(divs=2, i=1/4.0, k=1/5.)

model.calculate_boundaries()

#===============================================================================
# model variables :
rhos  = 360.                   # initial density at surface ..... kg/m^3
rhoin = model.rhoi(0)          # initial density at surface ..... kg/m^3
rin   = 0.0005                 # initial grain radius ........... m^2
adot  = 0.01                   # accumulation rate .............. m/a
Tavg  = 273.15 - 15.0          # average temperature ............ degrees K

dt1   = 10.0*model.spy(0)      # time-step ...................... s
dt2   = 0.1/365.0*model.spy(0) # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = 1001                   # end-time ....................... a
tf    = tf*model.spy(0)        # end-time ....................... s
bp    = True                   # plot or not .................... bool
tm    = 1000.0 * model.spy(0)

#===============================================================================
# enthalpy BC :
code      = 'c*(Tavg + 10*(sin(2*omega*t) + 5*sin(4*omega*t)))'
theta_exp = Expression(code, Tavg=Tavg, omega=pi/model.spy(0), 
                       t=t0, c=model.ci(0))

T_i = theta_exp(zs) / model.ci(0)

# surface density :
rho_exp = Expression('rhon', rhon=rhos)
#rho_exp = Constant(rhos)

# velocity of surface (-acc / rhos) [m/s] :
code    = '- rhoi/rhos * adot / spy'
w_exp   = Expression(code, rhoi=model.rhoi(0), adot=adot, 
                     spy=model.spy(0), rhos=rhos)

# grain radius of surface [m^2] :
r_exp   = Expression('r_s', r_s=rin)

model.init_theta_surface(theta_exp)
model.init_rho_surface(rho_exp)
model.init_w_surface(w_exp)
model.init_r_surface(r_exp)
model.init_sigma_surface(0.0)

model.init_T(T_i)
model.init_rho(rhoin)
model.init_r(rin)
model.init_adot(adot)
model.init_time_step(dt1)

# load initialization data :
#model.set_ini_conv(ex)

plot_cfg = {  'on'       : bp,
              'zMin'     : -20,
              'zMax'     : 2,
              'wMin'     : -0.5,
              'wMax'     : 0.5,
              'uMin'     : -4e-3,
              'uMax'     : 4e-4,
              'rhoMin'   : 0.0,
              'rhoMax'   : 1000,
              'rMin'     : 0.0,
              'rMax'     : 3.0,
              'Tmin'     : -50.0,
              'Tmax'     : 5.0,
              'ageMin'   : 0.0,
              'ageMax'   : 100,
              'WMin'     : -0.05, 
              'WMax'     : 0.25,
              'enthalpy' : True,
              'density'  : True,
              'velocity' : True,
              'age'      : False  }

mom = MomentumFirn(model)
nrg = EnergyFirn(model, mom)
plt = FirnPlot(model, plot_cfg)

def cb():
  model.update_height_history()
  theta_exp.t = model.t
  rho_exp.t   = model.t
  w_exp.t     = model.t
  #bdotNew     = (w_exp.adot * model.rhoi(0)) / model.spy(0)
  #model.assign_variable(model.bdot, bdotNew)
  plt.update()

dt_list = [dt1,dt2]
#dt_list = None 

model.transient_solve(mom, nrg, t0, tm, tf, dt1, dt_list=dt_list, callback=cb)
  
p.ioff()
p.show()

# plot the surface height trend :
#plt.plot_height(model.times, model.ht, model.origHt)



