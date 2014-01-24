import sys
import os
src_directory = '../../../../'
sys.path.append(src_directory)

import src.physical_constants as pc
from data.data_factory   import DataFactory
from pylab               import *
from gmshpy              import *
from scipy.interpolate   import RectBivariateSpline
from src.utilities       import DataInput

#===============================================================================
# data preparation :

thklim = 200.0

# collect the raw data :
#measures = DataFactory.get_ant_measures()
#bedmap1  = DataFactory.get_bedmap1(thklim=thklim)
#bedmap2  = DataFactory.get_bedmap2(thklim=thklim)

dbv = DataInput("../results/", ("v_mag.mat", ), gen_space=False)
dbv.set_data_min('v_mag', 0.0,   0.0)
dbv.set_data_max('v_mag', 800.0, 800.0)

#db2  = DataInput(None, bedmap2, gen_space=False)

# might want to refine off of thickness :
#H   = db2.data['H'].copy().T

# ensure that there are no values less than 1 for taking log :
#vel = dsr.data['U_ob'].copy().T
vel  = dbv.data['v_mag'].copy().T
vel += 1

# invert the sections where the velocity is low to refine at divide :
data                = log(vel)
#mv                  = data.max()
#k                   = 1
#data[2*data < k*mv] = -data[2*data < k*mv]  + k*mv

# plot to check :
#imshow(data.T[::-1,::-1])
#colorbar()
#show()

# x- and y-values for creating nearest-neighbor spline interpolation :
xs     = dbv.x
ys     = dbv.y
spline = RectBivariateSpline(xs, ys, data, kx=1, ky=1)

#===============================================================================
# GMSHing stuff :

#load the mesh into a GModel
m = GModel.current()
m.load("3dmesh.geo")

#boolean options are stored as floating point numbers in gmsh => 0. is False
GmshSetOption("Mesh", "CharacteristicLengthFromPoints", 0.)
GmshSetOption("Mesh", "CharacteristicLengthExtendFromBoundary", 0.)
GmshSetOption("Mesh", "Smoothing", 100.)

#the callback can be a member function
class attractor:
  """
  Create an attractor object which refines with min and max cell radius <lmin>,
  <lmax> over data field <field>.  The <f_max> parameter specifies a max value
  for which to apply the minimum cell size such that if <field>_i is less than 
  <f_max>, the cell size in this region will be <lmax>.  If <hard_cut> is true,
  the values of <field> above <lmax> will be set to <lmin>, otherwise regular
  interpolation based on <field> is performed.  If <inv> = True the object 
  refines on the inverse of the data field <field>.
  """
  def __init__(self, field, f_max, lmin, lmax, hard_cut=False, inv=True):
    self.field    = field
    self.lmin     = lmin
    self.lmax     = lmax
    self.f_max    = f_max
    self.hard_cut = hard_cut
    self.inv      = inv
    self.c        = (self.lmax - self.lmin) / self.field.max()
  
  def op(self, x, y, z, entity):
    v = spline(x,y)[0][0]
    if v > self.f_max:
      if self.hard_cut:
        return self.lmin
      else:
        if self.inv:
          lc = self.lmax - self.c * v
        else:
          lc = self.lmin + self.c * v
        return lc
    else:
      return self.lmax

class min_field:
  """
  Return the minimum of a list of attactor operator fields <f_list>.
  """
  def __init__(self, f_list):
    self.f_list = f_list

  def op(self, x, y, z, entity):
    l = []
    for f in self.f_list:
      l.append(f(x,y,z,entity))
    return min(l)

lmin = 10000
lmax = 100000

a1   = attractor(data, 0.0, lmin,     lmax, inv=True)
a2   = attractor(data, 0.0, lmin+1e4, lmax, inv=False)
m1   = min_field([a1.op, a2.op])
aid1 = m.getFields().addPythonField(a1.op)
aid2 = m.getFields().addPythonField(a2.op)
mid  = m.getFields().addPythonField(m1.op)
m.getFields().setBackgroundFieldId(mid)

#launch the GUI
#FlGui.instance().run()

#instead of starting the GUI, we could generate the mesh and save it
m.mesh(3) # 2 is the dimension
m.save("mesh_low.msh")

