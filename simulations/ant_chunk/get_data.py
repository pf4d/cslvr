""" Script for projecting data onto a mesh for a rectangular region of Antarctica"""

import varglas.model              as model
from varglas.data.data_factory    import DataFactory
from varglas.utilities            import DataInput
from fenics                       import *

thklim = 200.0

# Get the Antarctica data sets
bedmap1   = DataFactory.get_bedmap1(thklim=thklim)
bedmap2   = DataFactory.get_bedmap2(thklim=thklim)
measures  = DataFactory.get_ant_measures(res=900)

mesh = Mesh("ant_flat_mesh.xml")        

db1 = DataInput(None, bedmap1, mesh = mesh)
db2 = DataInput(None, bedmap2, mesh = mesh)
dm = DataInput(None, measures, mesh = mesh)

# Project surface, bed, and thickness onto a flat mesh
# Ice thickness
H_proj = db2.get_projection('H')
File('data/H.xml') << H_proj
File('data/H.pvd') << H_proj
# Surface 
S_proj = db2.get_projection('S')
File('data/Surface.xml') << S_proj
File('data/Surface.pvd') << S_proj
# Bed 
B_proj = db2.get_projection('B')
File('data/Bed.xml') << B_proj
File('data/Bed.pvd') << B_proj

# Then create a new deformed mesh
model = model.Model()
# Set the mesh to the non-deformed anisotropic mesh
model.set_mesh(mesh)

# Deform it to match the surface and bed geometry
Surface = db2.get_spline_expression("S")
Bed = db2.get_spline_expression("B")
model.set_geometry(Surface, Bed)
model.deform_mesh_to_geometry()

# Get the boundaries of the mesh
x_min = model.mesh.coordinates()[:,0].min()
x_max = model.mesh.coordinates()[:,0].max()
y_min = model.mesh.coordinates()[:,1].min()
y_max = model.mesh.coordinates()[:,1].max()

# Write out the deformed mesh
File("ant_deformed_mesh.xml") << model.mesh

# Then project the rest of the data onto the function space of the deformed mesh
# with periodic boundaries enabled

# Surface temperature
Ts_proj = db1.get_projection('srfTemp')
File('data/Ts_deformed_per.xml') << Ts_proj 
File('data/Ts_deformed_per.pvd') << Ts_proj 
# Basal heat flux
Qgeo_proj = db1.get_projection('q_geo')
File('data/Qgeo_deformed_per.xml') << Qgeo_proj
File('data/Qgeo_deformed_per.pvd') << Qgeo_proj
# Accumulation
adot_proj = db1.get_projection('adot')
File('data/adot_deformed_per.xml') << adot_proj
File('data/adot_deformed_per.pvd') << adot_proj
# Magnitude of Ice velocity
Umag_proj = dm.get_projection('U_ob')
File('data/Umag_deformed_per.xml') << Umag_proj
File('data/Umag_deformed_per.pvd') << Umag_proj