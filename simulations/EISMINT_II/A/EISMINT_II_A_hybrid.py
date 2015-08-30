from fenics  import *
from varglas import *
from varglas.energy import EnergyHybrid
from varglas.mass   import MassTransportHybrid

#set_log_active(False)

parameters['form_compiler']['precision']         = 30
parameters['form_compiler']['optimize']          = True
parameters['form_compiler']['cpp_optimize']      = True
parameters['form_compiler']['representation']    = 'quadrature'

mesh = MeshFactory.get_circle()

out_dir = './A_hybrid/'
thklim  = 1.0
L       = 800000.
xmin    = -L
xmax    =  L
ymin    = -L
ymax    =  L

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

model = D2Model(out_dir = out_dir)
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = False)

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
model.init_T_T0_(268.0)
model.init_H(thklim)
model.init_H_bounds(thklim, 1e4)
model.init_q_geo(model.ghf)
#model.init_beta_stats()

model.eps_reg.assign(1e-10)

mom = MomentumHybrid(model, isothermal=False)
nrg = EnergyHybrid(model, transient=True)
mas = MassTransportHybrid(model, thklim=thklim, isothermal=False)

U_file  = File(out_dir + 'U.pvd')
S_file  = File(out_dir + 'S.pvd')
Tb_file = File(out_dir + 'Tb.pvd')
Ts_file = File(out_dir + 'Ts.pvd')
def cb_ftn():
  model.save_pvd(model.U3, 'U3', U_file)
  model.save_pvd(model.Ts, 'Ts', Tb_file)
  model.save_pvd(model.Tb, 'Tb', Ts_file)
  model.save_pvd(model.S,  'S',  S_file)

model.transient_solve(mom, nrg, mas, t_start=0.0, t_end=200000.0, 
                      time_step = 10.0, annotate=False, callback=cb_ftn)



