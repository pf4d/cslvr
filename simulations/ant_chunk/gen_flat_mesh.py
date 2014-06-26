# -*- coding: utf-8 -*-
"""
Generates a mesh for a rectangular region of Antarctica defined by two 
GPS coordinates. 
"""
from varglas.data.data_factory import DataFactory
from varglas.utilities import DataInput, MeshGenerator, MeshRefiner
from numpy import *

"""
Creates a rectangular mesh for a region defined by two corners (lon0,lat0) and 
(lon1,lat1).  

Inputs :
(lon0,lat0) : One corner of rectangular mesh
(lon1,lat1) : Other corner of rectangular mesh
mesh_name   : Name of the generated mesh file
refine      : Set to true to refine mesh based on thickness
refine_rad  : The maximum radius of cells in the mesh in terms of ice 
              thicknesses              

Example usage: 
lon0, lat0 = (45.0, -87.82335879855461)
lon1, lat1 = (-63.52084947747111, -88.62120337777857)
"""

def create_rect_mesh(lon0,lat0,lon1,lat1,mesh_name,refine = True,refine_rad = 1) :   
  bm2 = DataFactory.get_bedmap2()
  bedmap2 = DataInput(None,bm2)

  # Convert (lon,lat) to (x,y) using the data input object's projection
  x0, y0 = bedmap2.get_xy(lon0,lat0)
  x1, y1 = bedmap2.get_xy(lon1,lat1)
  
  # Manually create a mesh contour array that defines the points in the 
  # rectangular region 
  cont = array([
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1]
  ])
  
  # Create a mesh file in the current directory 
  m = MeshGenerator(bedmap2, mesh_name, '')
  # Manually set the countour instead of calculating it automatically
  m.set_contour(cont)
  # Write the contour points to the mesh file
  m.write_gmsh_contour(100000, boundary_extend = False)
  # Extrude the flat mesh 10,000 meters in the z dimension. The height of the 
  # mesh can be automatically scaled to the proper height by the model object
  m.extrude(10000, 10)
  
  # We're finished with the flat mesh!
  m.close_file()
  
  # Refine the mesh based on thickenss
  if refine :
    ref_bm = MeshRefiner(bedmap2, 'H', gmsh_file_name = mesh_name) 
    # Refine the mesh based on the refinement radius
    a, aid = ref_bm.add_static_attractor(refine_rad)
    ref_bm.set_background_field(aid)
    # Write out the file
    ref_bm.finish(gui=False, out_file_name = mesh_name)
    
  # Convert the generated .msh file to an xml file
  m.convert_msh_to_xml(mesh_name, mesh_name)



