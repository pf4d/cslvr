import inspect
import os
import sys
from fenics import Mesh

class MeshFactory(object):
 
  @staticmethod
  def get_greenland_detailed():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/greenland/greenland_detailed_mesh.xml')
    return mesh
 

  @staticmethod
  def get_greenland_medium():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/greenland/greenland_medium_mesh.xml')
    return mesh


  @staticmethod
  def get_greenland_coarse():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/greenland/greenland_coarse_mesh.xml')
    return mesh
  
  
  @staticmethod
  def get_greenland_2D_1H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/greenland/greenland_2D_1H_mesh.xml')
    return mesh
 
  
  @staticmethod
  def get_greenland_2D_5H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/greenland/greenland_2D_5H_mesh.xml')
    return mesh
 
  
  @staticmethod
  def get_greenland_3D_5H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/greenland/greenland_3D_5H_mesh.xml')
    return mesh


  @staticmethod
  def get_antarctica_coarse():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_50H_5l.xml')
    return mesh

  
  @staticmethod
  def get_antarctica_3D_50H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_3D_50H_mesh.xml')
    return mesh
 
  
  @staticmethod
  def get_antarctica_3D_100H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_3D_100H_mesh.xml')
    return mesh
 
  
  @staticmethod
  def get_antarctica_3D_gradS_detailed():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_3D_gradS_mesh_detailed.xml')
    return mesh
 
  
  @staticmethod
  def get_antarctica_3D_gradS_crude():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_3D_gradS_mesh_crude.xml')
    return mesh


  @staticmethod
  def get_ronne_3D_10H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_ronne_shelf.xml')
    return mesh


  @staticmethod
  def get_ronne_3D_50H():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/antarctica/antarctica_ronne_shelf_crude.xml')
    return mesh


  @staticmethod
  def get_circle():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    mesh     = Mesh(home + '/test/circle.xml')
    return mesh



