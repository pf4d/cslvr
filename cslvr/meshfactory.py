import inspect
import os
import sys
from fenics import Mesh

class MeshFactory(object):
    
  global home 
  filename = inspect.getframeinfo(inspect.currentframe()).filename
  home     = os.path.dirname(os.path.abspath(filename)) + '/../meshes'
 
  @staticmethod
  def get_greenland_detailed():
    global home
    return Mesh(home + '/greenland/greenland_detailed_mesh.xml')
 

  @staticmethod
  def get_greenland_medium():
    global home
    return Mesh(home + '/greenland/greenland_medium_mesh.xml')


  @staticmethod
  def get_greenland_coarse():
    global home
    return Mesh(home + '/greenland/greenland_coarse_mesh.xml')
  
  
  @staticmethod
  def get_greenland_2D_1H():
    global home
    return Mesh(home + '/greenland/greenland_2D_1H_mesh.xml')
 
  
  @staticmethod
  def get_greenland_2D_5H():
    global home
    return Mesh(home + '/greenland/greenland_2D_5H_mesh.xml')
 
  
  @staticmethod
  def get_greenland_2D_5H_sealevel():
    global home
    return Mesh(home + '/greenland/greenland_2D_5H_sealevel.xml')
 
  
  @staticmethod
  def get_greenland_3D_5H():
    global home
    return Mesh(home + '/greenland/greenland_3D_5H_mesh.xml')


  @staticmethod
  def get_antarctica_coarse():
    global home
    return Mesh(home + '/antarctica/antarctica_50H_5l.xml')

  
  @staticmethod
  def get_antarctica_2D_medium():
    global home
    return Mesh(home + '/antarctica/antarctica_2D_medium_mesh.xml')

  
  @staticmethod
  def get_antarctica_3D_50H():
    global home
    return Mesh(home + '/antarctica/antarctica_3D_50H_mesh.xml')
 
  
  @staticmethod
  def get_antarctica_3D_100H():
    global home
    return Mesh(home + '/antarctica/antarctica_3D_100H_mesh.xml')
 
  
  @staticmethod
  def get_antarctica_3D_gradS_detailed():
    global home
    return Mesh(home + '/antarctica/antarctica_3D_gradS_mesh_detailed.xml')
  
  @staticmethod
  def get_antarctica_3D_10k():
    global home
    return Mesh(home + '/antarctica/antarctica_3D_10k.xml')

  
  @staticmethod
  def get_antarctica_3D_gradS_crude():
    global home
    return Mesh(home + '/antarctica/antarctica_3D_gradS_mesh_crude.xml')


  @staticmethod
  def get_ronne_3D_10H():
    global home
    return Mesh(home + '/antarctica/antarctica_ronne_shelf.xml')


  @staticmethod
  def get_ronne_3D_50H():
    global home
    return Mesh(home + '/antarctica/antarctica_ronne_shelf_crude.xml')


  @staticmethod
  def get_circle():
    global home
    return Mesh(home + '/test/circle_mesh.xml.gz')



