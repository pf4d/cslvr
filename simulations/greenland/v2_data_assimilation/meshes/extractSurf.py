from dolfin import *

"""
mesh = UnitCubeMesh(10,10,10)
V    = FunctionSpace(mesh, "CG", 1)
u    = Function(V)
u.vector()[:] = 1.5

bmesh   = BoundaryMesh(mesh, "exterior")
#File('bmesh.xml') << bmesh
mapping = bmesh.entity_map(2) 
part_of_boundary = CellFunction("size_t", bmesh, 0)
for cell in cells(bmesh):
  if Facet(mesh, mapping[cell.index()]).normal().z() < 0:
    part_of_boundary[cell] = 1

submesh_of_boundary = SubMesh(bmesh, part_of_boundary, 1)
Vb = FunctionSpace(submesh_of_boundary, "CG", 1)
ub = Function(Vb)
ub.interpolate(u)
File("ub.pvd") << ub
"""

f = HDF5File('u.h5', 'r')

mesh = Mesh()
f.read(mesh, 'mesh')

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
f.read(u, 'beta2')

bmesh   = BoundaryMesh(mesh, "exterior")
mapping = bmesh.entity_map(2)
part_of_boundary = CellFunction("size_t", bmesh, 0)

for cell in cells(bmesh):
  if Facet(mesh, mapping[cell.index()]).normal().z() < 0:
    part_of_boundary[cell] = 1

submesh_of_boundary = SubMesh(bmesh, part_of_boundary, 1)
Vb = FunctionSpace(submesh_of_boundary, "CG", 1)
ub = Function(Vb)
ub.interpolate(u)
File("beta2_2d.pvd") << ub

