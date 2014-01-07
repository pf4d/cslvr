
"""

from dolfin import *
from pylab  import *

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

"""
    
from dolfin import *
from pylab  import *

mesh = UnitCubeMesh(10,10,10)          # original mesh
mesh.coordinates()[:,0] -= .5          # shift x-coords
mesh.coordinates()[:,1] -= .5          # shift y-coords
V    = FunctionSpace(mesh, "CG", 1)
u    = Function(V)

# apply expression over cube for clearer results :
u_i  = Expression('sqrt(pow(x[0],2) + pow(x[1], 2))')
u.interpolate(u_i)

bmesh  = BoundaryMesh(mesh, "exterior")   # surface boundary mesh

cellmap = bmesh.entity_map(2)
vertmap = bmesh.entity_map(0)
pb      = CellFunction("size_t", bmesh, 0)
for c in cells(bmesh):
  if Facet(mesh, cellmap[c.index()]).normal().z() < 0:
    pb[c] = 1

submesh = SubMesh(bmesh, pb, 1)           # subset of surface mesh

Vb = FunctionSpace(bmesh,   "CG", 1)      # surface function space
Vs = FunctionSpace(submesh, "CG", 1)      # submesh function space

ub = Function(Vb)                         # boundary function
us = Function(Vs)                         # desired function
un = Function(Vs)

us.interpolate(u)

#m    = V.dofmap().vertex_to_dof_map(mesh)        # mesh dofmap
#b    = Vb.dofmap().vertex_to_dof_map(bmesh)      # bmesh dofmap
#s    = Vs.dofmap().vertex_to_dof_map(submesh)    # submesh dofmap
m    = vertex_to_dof_map(V)       # mesh dofmap
b    = vertex_to_dof_map(Vb)      # bmesh dofmap
s    = vertex_to_dof_map(Vs)      # submesh dofmap

#mi   = V.dofmap().dof_to_vertex_map(mesh)        # mesh dofmap
#bi   = Vb.dofmap().dof_to_vertex_map(bmesh)      # bmesh dofmap
#si   = Vs.dofmap().dof_to_vertex_map(submesh)    # submesh dofmap
mi   = dof_to_vertex_map(V)       # mesh dofmap
bi   = dof_to_vertex_map(Vb)      # bmesh dofmap
si   = dof_to_vertex_map(Vs)      # submesh dofmap

u_a  = u.vector().array()                 # array form of original ftn
ub_a = ub.vector().array()                # array form of boundary ftn
us_a = us.vector().array()                # array form of desired ftn
un_a = un.vector().array()                # array form of desired ftn

for v in vertices(bmesh):
  i       = v.index()
  #ub_a[i] = u_a[mi[vertmap[i]]]
  ub_a[i] = u_a[bi[i]]

ub.vector().set_local(ub_a)
File("ub_new.pvd") << ub

un.vector().set_local(ub_a[b][s])
File("us.pvd") << un



