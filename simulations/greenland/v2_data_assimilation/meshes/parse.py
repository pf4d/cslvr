import re
from pylab  import *
from dolfin import *

#===============================================================================
# parse the mesh file :
grid    = open('mesh.xml', 'r')
newGrid = open('bed.xml',  'w')

sv    = '^.*index="([0-9]+)"\sx="(-?[0-9]+\.[0-9]+e?[+-]?[0-9]+)"' + \
                           '\sy="(-?[0-9]+\.[0-9]+e?[+-]?[0-9]+)"' + \
                           '\sz="(-?[0-9]+\.[0-9]+e?[+-]?[0-9]+)"'
pv    = re.compile(sv)

sc    = '^.*tetrahedron\sindex="([0-9]+)"' + \
                       '\sv0="([0-9]+)"' + \
                       '\sv1="([0-9]+)"' + \
                       '\sv2="([0-9]+)"' + \
                       '\sv2="([0-9]+)"'
pc    = re.compile(sc)

ci    = []  # cell index
vi    = []  # vertex index
vx    = []  # vertex x coordinate
vy    = []  # vertex y coordinate
vz    = []  # vertex z coordinate

print "<::::::::PARSING THE 3D XML FILE::::::::>"
for l in grid.readlines():
  mrv = pv.match(l)
  mrc = pc.match(l)
  if mrv != None:
    vi.append(int(mrv.group(1)))
    vx.append(float(mrv.group(2)))
    vy.append(float(mrv.group(3)))
    vz.append(float(mrv.group(4)))
  if mrc != None:
    v0 = int(mrc.group(1))
    v1 = int(mrc.group(2))
    v2 = int(mrc.group(3))
    v3 = int(mrc.group(4))
    v4 = int(mrc.group(5))
    ci.append([v0, v1, v2, v3, v4])
  if mrc == None and mrv == None:
    print "DATA NOT MATCHED :\t", l[:-1]
print "::::END OF GRID FILE::::"
grid.close()

# convert to array (more efficient) :
ci = array(ci)  # cell index
vi = array(vi)  # vertex index
vx = array(vx)  # vertex x coordinate
vy = array(vy)  # vertex y coordinate
vz = array(vz)  # vertex z coordinate

#===============================================================================
# formulate triangles :
# throw out parts that are not on the bed :
bed = where(vz == 0)[0]
n   = len(bed)
vi  = vi[bed] 
vx  = vx[bed] 
vy  = vy[bed] 
vz  = vz[bed] 
ta  = []                # triangles, values are indices to vertices
ti  = []                # cell index, need to make this cell the same as 3D

# iterate though each cell and find any cell with exactly 3 matching vertices 
# on the surface.  Then append a list of each vertex index that matches to the 
# array of triangles.
print "<::::::::FORMULATING TRIANGLES::::::::>"
for c in ci:
  k = in1d(vi, c[1:])
  if sum(k) == 3:
    ta.append(append(c[0], vi[k]))
print "::::FINISHED::::"

ta = array(ta)
m  = len(ta)

#===============================================================================
# create the new 2D .xml file :
print "<::::::::WRITING NEW 2D XML::::::::>"
newGrid.write('<?xml version="1.0" encoding="UTF-8"?>\n\n' + \
              '<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n' + \
              '  <mesh celltype="triangle" dim="2">\n' + \
              '    <vertices size="%i">\n' % n)

for i,x,y,z in zip(vi,vx,vy,vz):
  newGrid.write('      <vertex index="%i" x="%f" y="%f" z="%f"/>\n' 
                % (i,x,y,z))

newGrid.write('    </vertices>\n' + \
              '    <cells size="%i">\n' % m)
for i,v0,v1,v2 in ta:
  newGrid.write('      <triangle index="%i" v0="%i" v1="%i" v2="%i"/>\n' 
                % (i,v0,v1,v2))

newGrid.write('    </cells>\n' + \
              '  </mesh>\n' + \
              '</dolfin>')
print "::::FINISHED::::"

newGrid.close()
  
  
mesh   = Mesh('mesh.xml')
bed    = Mesh('bed.xml')
Q      = FunctionSpace(mesh, 'CG', 1)  
Qb     = FunctionSpace(bed,  'CG', 1)
B2     = Function(Q)
bedB2  = Function(Qb)
df     = Q.dofmap()
dfmap  = df.vertex_to_dof_map(mesh)

File('../results_sq/beta2_opt.xml') >> B2

B2a    = B2.vector().array()
bedB2a = bedB2.vector().array()

for v in vertices(mesh):
  ind.append(v.index())

  
  def get_nearest(self, fn):
    """
    returns a dolfin Function object with values given by interpolated 
    nearest-neighbor data <fn>.
    """
    #FIXME: get to work with a change of projection.
    # get the dofmap to map from mesh vertex indices to function indicies :
    df    = self.func_space.dofmap()
    dfmap = df.vertex_to_dof_map(self.mesh)
    
    unew  = Function(self.func_space)         # existing dataset projection
    uocom = unew.vector().array()             # mesh indexed main vertex values
    
    d     = float64(self.data[fn])            # original matlab spec dataset

    # get arrays of x-values for specific domain
    xs    = self.x
    ys    = self.y
    
    for v in vertices(self.mesh):
      # mesh vertex x,y coordinate :
      i   = v.index()
      p   = v.point()
      x   = p.x()
      y   = p.y()
      
      # indexes of closest datapoint to specific dataset's x and y domains :
      idx = abs(xs - x).argmin()
      idy = abs(ys - y).argmin()
      
      # data value for closest value :
      dv  = d[idy, idx] 
      if dv > 0:
        dv = 1.0
      uocom[i] = dv
    
    # set the values of the empty function's vertices to the data values :
    unew.vector().set_local(uocom[dfmap])
    return unew
