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

thklim = 200.0

# collect the raw data :
bamber  = DataFactory.get_bamber(thklim = thklim)
dbv     = DataInput("../results/", ("Ubmag_measures.mat", "Ubmag.mat"), 
                    gen_space=False)
dbv.set_data_min('Ubmag', 1.0, 1.0)

dbm = DataInput(None, bamber, gen_space=False)

H   = dbm.data['H_n'].T
vel = dbv.data['Ubmag'].T
vel[vel > 500] = 500.0

vx,vy = gradient(vel)

data = log(vel)
#data = H
xs   = dbm.x
ys   = dbm.y

spline = RectBivariateSpline(xs, ys, data, kx=1, ky=1)


#load the mesh into a GModel
m = GModel.current()
m.load("mesh.geo")

#boolean options are stored as floating point numbers in gmsh => 0. is False
GmshSetOption("Mesh", "CharacteristicLengthFromPoints", 0.)
GmshSetOption("Mesh", "CharacteristicLengthExtendFromBoundary", 0.)
GmshSetOption("Mesh", "Smoothing", 100.)

#the callback can be a member function
class attractor:
  
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

  def __init__(self, f_list):
    self.f_list = f_list

  def op(self, x, y, z, entity):
    l = []
    for f in self.f_list:
      l.append(f(x,y,z,entity))
    return min(l)

#lmax      = 50000
#lmin      = 2000
#lmin_max  = 5000
#lmin_min  = 500
#num_cuts  = 5
#fmax_max  = 8.0
#a_list    = []
#lc_list   = linspace(lmin_max, lmin_min, num_cuts)
#fmax_list = linspace(0, fmax_max, num_cuts)
#for fmax, l_min in zip(fmax_list, lc_list):
#  a = attractor(data, fmax, l_min, lmax, hard_cut=False)
#  a_list.append(a.op)
#  m.getFields().addPythonField(a.op)
#
#m1  = min_field(a_list)
#mid = m.getFields().addPythonField(m1.op)
#m.getFields().setBackgroundFieldId(mid)

a1 = attractor(data, 0.0, 1000, 50000, inv=True)
#a2 = attractor(data, 5.0,  500, 25000, hard_cut=False)
#m1 = min_field([a1.op, a2.op])
#
aid1 = m.getFields().addPythonField(a1.op)
#aid2 = m.getFields().addPythonField(a2.op)
#mid  = m.getFields().addPythonField(m1.op)
#
#m.getFields().setBackgroundFieldId(mid)
m.getFields().setBackgroundFieldId(aid1)

#m.extrude(0,0,1)

#launch the GUI
FlGui.instance().run()

#instead of starting the GUI, we could generate the mesh and save it
#m.mesh(2) # 2 is the dimension
#m.save("square.msh")
