from fenics          import *
from dolfin_adjoint  import *
from varglas         import FirnPlot, D1Model, MomentumFirn
from varglas.energy  import EnergyFirn
import sys
    
n     =  100                   # num of z-positions
zs    =  0.0                   # surface start .................. m
zb    = -100.0                 # depth .......................... m

mesh  = IntervalMesh(100, zb, zs)      # interval from bed to surface

model = D1Model(out_dir = 'results')
model.set_mesh(mesh)

#model.refine_mesh(divs=3, i=1/3., k=1/20.)
model.refine_mesh(divs=2, i=1/3.,  k=1/4.)
#model.refine_mesh(divs=2, i=1/3.,  k=1/4.)
#model.refine_mesh(divs=2, i=1/3.,  k=1/4.)
#model.refine_mesh(divs=2, i=1/3.,  k=1/4.)

model.generate_function_spaces()
model.calculate_boundaries()

#===============================================================================
# model variables :
rhos  = 360.                   # initial density at surface ..... kg/m^3
rhoin = 717.                   # initial density at surface ..... kg/m^3
rin   = 0.0005**2              # initial grain radius ........... m^2
adot  = 0.1                    # accumulation rate .............. m/a
Tavg  = 273.15 - 15.0          # average temperature ............ degrees K

dt1   = 10.0*model.spy(0)      # time-step ...................... s
dt2   = 0.5/365.0*model.spy(0) # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*model.spy(0) # end-time ....................... s
bp    = int(sys.argv[2])       # plot or not .................... bool
tm    = 500.0 * model.spy(0)

#===============================================================================
# enthalpy BC :
#code  = 'cp*(Tavg + 5*(sin(2*omega*t) + 5*sin(4*omega*t)))'
#H_exp = Expression(code, cp=cpi, Tavg=Tavg, omega=pi/spy, t=t0)
code  = 'c*(Tavg + 5*(sin(2*omega*t) + 5*sin(4*omega*t)))'
H_exp = Expression(code, Tavg=Tavg, omega=pi/model.spy(0), t=t0, c=model.ci(0))

# surface density :
rho_exp = Expression('rhon', rhon=rhos)
#rho_exp = Constant(rhos)

# velocity of surface (-acc / rhos) [m/s] :
code    = '- rhoi/rhos * adot / spy'
w_exp   = Expression(code, rhoi=model.rhoi(0), adot=adot, 
                     spy=model.spy(0), rhos=rhos)

# grain radius of surface [cm^2] :
r_exp   = Expression('r_s', r_s=rin)

model.init_H_surface(H_exp)
model.init_rho_surface(rho_exp)
model.init_w_surface(w_exp)
model.init_r_surface(r_exp)
model.init_sigma_surface(0.0)

model.init_T(Tavg)
model.init_rho(rhoin)
model.init_r(rin)
model.init_adot(adot)
model.init_time_step(dt1)

# load initialization data :
#model.set_ini_conv(ex)

plot_cfg = {  'on'       : bp,
              'zMin'     : -100,
              'zMax'     : 20.0,
              'wMin'     : -30,
              'wMax'     : 5,
              'uMin'     : -1500,
              'uMax'     : 300,
              'rhoMin'   : 0.0,
              'rhoMax'   : 1000,
              'rMin'     : 0.0,
              'rMax'     : 3.0,
              'Tmin'     : -50.0,
              'Tmax'     : 5.0,
              'ageMin'   : 0.0,
              'ageMax'   : 100,
              'omegaMin' : -0.01, 
              'omegaMax' : 0.10,
              'enthalpy' : True,
              'density'  : True,
              'velocity' : True,
              'age'      : False  }

mom = MomentumFirn(model)
nrg = EnergyFirn(model)
plt = FirnPlot(model, plot_cfg)


def cb():
  model.H_surface.t    = model.t
  model.H_surface.c    = model.cp[0]
  model.rho_surface.t  = model.t
  model.w_surface.t    = model.t
  model.w_surface.rhos = model.rhop[0]
  bdotNew    = (model.w_surface.adot * model.rhoi(0)) / model.spy(0)
  model.assign_variable(model.bdot, bdotNew)
  plt.update()

model.transient_solve(mom, nrg, t0, tm, tf, dt, dt_list=[dt1,dt2], callback=cb)
  
#plt.ioff()
#plt.show()

# plot the surface height trend :
plt.plot_height(F.times, model.ht, model.origHt)



