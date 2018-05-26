from cslvr   import *
from mshr    import *

out_dir = './A_hybrid/'

thklim  = 1.0
r       = 800000.0
res     = 25

mesh  = generate_mesh(Circle(Point(0,0), r), res)
model = D2Model(mesh, out_dir=out_dir, use_periodic=False, kind='hybrid')

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

model.init_S(thklim)
model.init_B(0.0)
model.init_adot(adot)
model.init_beta(1e9)
model.init_T_surface(T_s)
model.init_T_T0(268.0)
model.init_H_H0(thklim)
model.init_H_bounds(thklim, 1e4)
model.init_q_geo(model.ghf)
#model.init_beta_stats()

model.eps_reg.assign(1e-10)

mom = MomentumHybrid(model, isothermal=False)
nrg = EnergyHybrid(model, mom, transient=True)
mas = MassHybrid(model, mom, thklim=thklim, isothermal=False)

U_file  = XDMFFile(out_dir + 'U.xdmf')
S_file  = XDMFFile(out_dir + 'S.xdmf')
Tb_file = XDMFFile(out_dir + 'Tb.xdmf')
Ts_file = XDMFFile(out_dir + 'Ts.xdmf')
def cb_ftn():
  nrg.solve()
  model.save_xdmf(model.U3, 'U3', U_file)
  model.save_xdmf(model.Tb, 'Tb', Tb_file)
  model.save_xdmf(model.Ts, 'Ts', Ts_file)
  model.save_xdmf(model.S,  'S',  S_file)

model.transient_solve(mom, mas,
                      t_start    = 0.0,
                      t_end      = 100.0,
                      time_step  = 10.0,
                      tmc_kwargs = None,
                      adaptive   = True,
                      annotate   = False,
                      callback   = cb_ftn)


