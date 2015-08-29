from fenics          import *
#from varglas.d2model import D2Model
from varglas import *

set_log_active(False)

parameters['form_compiler']['precision']         = 30
parameters['form_compiler']['optimize']          = True
parameters['form_compiler']['cpp_optimize']      = True
parameters['form_compiler']['representation']    = 'quadrature'

mesh = Mesh('meshes/circle_mesh.xml')

thklim = 1.0
L      = 800000.
xmin   = -L
xmax   =  L
ymin   = -L
ymax   =  L

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

#config['mode']                         = 'transient'
#config['t_start']                      = 0.0
#config['t_end']                        = 200000.0
#config['time_step']                    = 10.0
#config['free_surface']['on']           = True
#config['free_surface']['thklim']       = thklim
#config['balance_velocity']['on']       = True

model = D2Model(out_dir = './A/')
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
model.init_H(thklim)
model.init_H_bounds(thklim, 1e4)
model.init_q_geo(model.ghf)
#model.init_beta_stats()

model.eps_reg.assign(1e-10)




