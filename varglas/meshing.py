"""
Utilities file:

  This contains classes that are used by UM-FEISM to aid in the loading
  of data and preparing data for use in DOLFIN based simulation.
  Specifically the data are projected onto the mesh with appropriate
  basis functions.

"""
import subprocess
import inspect
import os
import Image
from gmshpy            import GModel, GmshSetOption, FlGui
from scipy.interpolate import RectBivariateSpline
from pylab             import array, linspace, ones, meshgrid, figure, \
                              size, hstack, vstack, argmin, zeros, shape, \
                              sqrt, show
from fenics            import Mesh, MeshEditor, Point, File
from pyproj            import transform
from io                import print_text, print_min_max
#from scipy.spatial     import ConvexHull
from shapely.geometry  import Polygon
from shapely.geometry  import Point as shapelyPoint
from shapely.ops       import cascaded_union


class MeshGenerator(object):
  """
  generate a mesh.
  """
  def __init__(self, dd, fn, direc):
    """
    Generate a mesh with DataInput object <dd>, output filename <fn>, and
    output directory <direc>.
    """
    self.color = 'grey_46'
    s    = "::: INITIALIZING MESHGENERATOR :::"
    print_text(s, self.color)
    self.dd         = dd
    self.fn         = fn
    self.direc      = direc
    self.x, self.y  = meshgrid(dd.x, dd.y)
    if not os.path.exists(direc):
      os.makedirs(direc)
    self.f          = open(direc + fn + '.geo', 'w')
    self.fieldList  = []  # list of field indexes created.

  def create_contour(self, var, zero_cntr, skip_pts):
    """
    Create a contour of the data field with index <var> of <dd> provided at
    initialization.  <zero_cntr> is the value of <var> to contour, <skip_pts>
    is the number of points to skip in the contour, needed to prevent overlap.
    """
    s    = "::: creating contour from %s's \"%s\" field with skipping %i " + \
           "point(s) :::"
    print_text(s % (self.dd.name, var, skip_pts) , self.color)

    skip_pts = skip_pts + 1

    # create contour :
    field  = self.dd.data[var]
    fig = figure()
    self.ax = fig.add_subplot(111)
    self.ax.set_aspect('equal')
    self.c = self.ax.contour(self.x, self.y, field, [zero_cntr])

    # Get longest contour:
    cl       = self.c.allsegs[0]
    ind      = 0
    amax     = 0
    amax_ind = 0

    for a in cl:
      if size(a) > amax:
        amax = size(a)
        amax_ind = ind
      ind += 1

    # remove skip points and last point to avoid overlap :
    self.longest_cont = cl[amax_ind]
    s    = "::: contour created, length %s nodes :::"
    print_text(s % shape(self.longest_cont)[0], self.color)
    self.remove_skip_points(skip_pts)

  def remove_skip_points(self, skip_pts):
    """
    remove every other <skip_pts> node from the contour.
    """
    # remove skip points and last point to avoid overlap :
    longest_cont      = self.longest_cont
    self.longest_cont = longest_cont[::skip_pts,:][:-1,:]
    s    = "::: contour points skipped, new length %s nodes :::"
    print_text(s % shape(self.longest_cont)[0], self.color)

  def transform_contour(self, di):
    """
    Transforms the coordinates of the contour to DataInput object <di>'s
    projection coordinates.
    """
    if type(di) == type(self.dd):
      proj = di.proj
      name = di.name
    elif type(di) == dict:
      name = di['dataset']
      proj = di['pyproj_Proj']
    s = "::: transforming contour coordinates from %s to %s :::"
    print_text(s % (name, self.dd.name), self.color)
    x,y    = self.longest_cont.T
    xn, yn = transform(self.dd.proj, proj, x, y)
    self.longest_cont = array([xn, yn]).T

  def set_contour(self,cont_array):
    """ This is an alternative to the create_contour method that allows you to
    manually specify contour points.
    Inputs:
    cont_array : A numpy array of contour points (i.e. array([[1,2],[3,4],...]))
    """
    s = "::: manually setting contour with %s nodes:::"
    print_text(s % shape(cont_array)[0], self.color)
    fig = figure()
    self.ax = fig.add_subplot(111)
    self.ax.set_aspect('equal')
    self.longest_cont = cont_array

  def plot_contour(self):
    """
    Plot the contour created with the "create_contour" method.
    """
    s = "::: plotting contour :::"
    print_text(s, self.color)
    ax = self.ax
    lc = self.longest_cont
    ax.plot(lc[:,0], lc[:,1], 'r-', lw = 3.0)
    ax.set_title("contour")
    show()

  def eliminate_intersections(self, dist=10):
    """
    Eliminate intersecting boundary elements. <dist> is an integer specifiying
    how far forward to look to eliminate intersections.  If any intersections
    are found, this method is called recursively until none are found.
    """
    s    = "::: eliminating intersections :::"
    print_text(s, self.color)

    class Point:
      def __init__(self,x,y):
        self.x = x
        self.y = y

    def ccw(A,B,C):
      return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

    def intersect(A,B,C,D):
      return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    lc   = self.longest_cont

    flag = ones(len(lc))
    intr = False
    for ii in range(len(lc)-1):

      A = Point(*lc[ii])
      B = Point(*lc[ii+1])

      for jj in range(ii, min(ii + dist, len(lc)-1)):

        C = Point(*lc[jj])
        D = Point(*lc[jj+1])

        if intersect(A,B,C,D) and ii!=jj+1 and ii+1!=jj:
          s    = "  - intersection found between node %i and %i"
          print_text(s % (ii+1, jj), 'red')
          flag[ii+1] = 0
          flag[jj]   = 0
          intr       = True

    counter  = 0
    new_cont = zeros((sum(flag),2))
    for ii,fl in enumerate(flag):
      if fl:
        new_cont[counter,:] = lc[ii,:]
        counter += 1

    self.longest_cont = new_cont
    s    = "::: eliminated %i nodes :::"
    print_text(s % sum(flag == 0), self.color)
    # call again if required :
    if intr:
      self.eliminate_intersections(dist)

  def restart(self):
    """
    clear all contents from the .geo file.
    """
    self.f.close
    self.f = open(self.direc + self.fn + '.geo', 'w')
    s = 'Reopened \"' + self.direc + self.fn + '.geo\".'
    print_text(s, self.color)

  def write_gmsh_contour(self, lc=100000, boundary_extend=True):
    """
    write the contour created with create_contour to the .geo file with mesh
    spacing <lc>.  If <boundary_extend> is true, the spacing in the interior
    of the domain will be the same as the distance between nodes on the contour.
    """
    #FIXME: sporadic results when used with ipython, does not stops writing the
    #       file after a certain point.  calling restart() then write again
    #       results in correct .geo file written.  However, running the script
    #       outside of ipython works.
    s    = "::: writing gmsh contour to \"%s%s.geo\" :::"
    print_text(s % (self.direc, self.fn), self.color)
    c   = self.longest_cont
    f   = self.f

    pts = size(c[:,0])

    # write the file to .geo file :
    f.write("// Mesh spacing\n")
    f.write("lc = " + str(lc) + ";\n\n")

    f.write("// Points\n")
    for i in range(pts):
      f.write("Point(" + str(i) + ") = {" + str(c[i,0]) + "," \
              + str(c[i,1]) + ",0,lc};\n")

    f.write("\n// Lines\n")
    for i in range(pts-1):
      f.write("Line(" + str(i) + ") = {" + str(i) + "," + str(i+1) + "};\n")
    f.write("Line(" + str(pts-1) + ") = {" + str(pts-1) + "," \
            + str(0) + "};\n\n")

    f.write("// Line loop\n")
    loop = ""
    loop += "{"
    for i in range(pts-1):
      loop += str(i) + ","
    loop += str(pts-1) + "}"
    f.write("Line Loop(" + str(pts+1) + ") = " + loop + ";\n\n")

    f.write("// Surface\n")
    surf_num = pts+2
    f.write("Plane Surface(" + str(surf_num) + ") = {" + str(pts+1) + "};\n\n")

    if not boundary_extend:
      f.write("Mesh.CharacteristicLengthExtendFromBoundary = 0;\n\n")

    self.surf_num = surf_num
    self.pts      = pts
    self.loop     = loop

  def extrude(self, h, n_layers):
    """
    Extrude the mesh <h> units with <n_layers> number of layers.
    """
    s    = "::: extruding gmsh contour %i layers :::" % n_layers
    print_text(s, self.color)
    f = self.f
    s = str(self.surf_num)
    h = str(h)
    layers = str(n_layers)

    f.write("Extrude {0,0," + h + "}" \
            + "{Surface{" + s + "};" \
            + "Layers{" + layers + "};}\n\n")

  def add_box(self, field, vin, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    add a box to the mesh.  e.g. for Byrd Glacier data:

      add_box(10000, 260000, 620000, -1080000, -710100, 0, 0)

    """
    f  = self.f
    fd = str(field)

    f.write("Field[" + fd + "]      =  Box;\n")
    f.write("Field[" + fd + "].VIn  =  " + float(vin)  + ";\n")
    f.write("Field[" + fd + "].VOut =  lc;\n")
    f.write("Field[" + fd + "].XMax =  " + float(xmax) + ";\n")
    f.write("Field[" + fd + "].XMin =  " + float(xmin) + ";\n")
    f.write("Field[" + fd + "].YMax =  " + float(ymax) + ";\n")
    f.write("Field[" + fd + "].YMin =  " + float(ymin) + ";\n")
    f.write("Field[" + fd + "].ZMax =  " + float(zmax) + ";\n")
    f.write("Field[" + fd + "].ZMin =  " + float(zmin) + ";\n\n")

    self.fieldList.append(field)

  def add_edge_attractor(self, field):
    """
    """
    fd = str(field)
    f  = self.f

    f.write("Field[" + fd + "]              = Attractor;\n")
    f.write("Field[" + fd + "].NodesList    = " + self.loop + ";\n")
    f.write("Field[" + fd + "].NNodesByEdge = 100;\n\n")

  def add_threshold(self, field, ifield, lcMin, lcMax, distMin, distMax):
    """
    """
    fd = str(field)
    f  = self.f

    f.write("Field[" + fd + "]         = Threshold;\n")
    f.write("Field[" + fd + "].IField  = " + str(ifield)  + ";\n")
    f.write("Field[" + fd + "].LcMin   = " + str(lcMin)   + ";\n")
    f.write("Field[" + fd + "].LcMax   = " + str(lcMax)   + ";\n")
    f.write("Field[" + fd + "].DistMin = " + str(distMin) + ";\n")
    f.write("Field[" + fd + "].DistMax = " + str(distMax) + ";\n\n")

    self.fieldList.append(field)

  def finish(self, field):
    """
    figure out background field and close the .geo file.
    """
    f     = self.f
    fd    = str(field)
    flist = self.fieldList

    # get a string of the fields list :
    l = ""
    for i,j in enumerate(flist):
      l += str(j)
      if i != len(flist) - 1:
        l += ', '

    # make the background mesh size the minimum of the fields :
    if len(flist) > 0:
      f.write("Field[" + fd + "]            = Min;\n")
      f.write("Field[" + fd + "].FieldsList = {" + l + "};\n")
      f.write("Background Field    = " + fd + ";\n\n")
    else:
      f.write("Background Field = " + fd + ";\n\n")

    s = 'finished, closing \"' + self.direc + self.fn + '.geo\".'
    print_text(s, self.color)
    f.close()

  def close_file(self):
    """
    close the .geo file down for further editing.
    """
    s    = '::: finished, closing \"' + self.direc + self.fn + '.geo\" :::'
    print_text(s, self.color)
    self.f.close()


  def create_2D_mesh(self, outfile):
    """
    create the 2D mesh to file <outfile>.msh.
    """
    #FIXME: this fails every time, the call in the terminal does work however.
    cmd = 'gmsh ' + '-2 ' + self.direc + self.fn + '.geo'# -2 -o ' \
                  #+ self.direc + outfile + '.msh'
    s = "\nExecuting :\n\n\t", cmd, "\n\n"
    print_text(s, self.color)
    subprocess.call(cmd.split())
  
  def check_dist(self, dist=1.0):
    """
    remove points in contour that are not a linear distance of at least
    <dist> from previous point.
    """
    lin_dist = lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    xycoords = self.longest_cont

    mask = ones(len(xycoords), dtype=bool)

    i = 0
    while(i < len(xycoords)-1):
      p1 = xycoords[i]
      j = i + 1
      while(j < len(xycoords) and \
            lin_dist(p1, xycoords[j]) < dist):
        mask[j] = 0
        j += 1
      i = j

    # fix end of array
    i = -1
    while(len(xycoords) + i >= 0 and (not mask[i] or \
          lin_dist(xycoords[0],xycoords[i]) < dist)):
      mask[i] = 0
      i -= 1

    # print results
    s    = "::: removed %s points closer than %s m to one another :::"% \
            (str(len(mask) - sum(mask)), dist)
    print_text(s, self.color)

    self.longest_cont = xycoords[mask]

  def intersection(self, new_contour):
    """
    Take the geometric intersection of current coordinates with <new_contour>.
    Used primarily to replace the edge with something from a different
    (better) data set.
    """
    contour = self.longest_cont

    p1 = Polygon(zip(contour[:,0],     contour[:,1]))
    p2 = Polygon(zip(new_contour[:,0], new_contour[:,1]))

    intersection = p1.intersection(p2)

    # check if multi-polygon is created. If so, take polygon 
    # with greatest area
    import collections
    if isinstance(intersection, collections.Iterable):
      p3 = max(intersection, key = lambda x: x.area)
    else:
      p3 = intersection

    contour_intersect = zip(p3.exterior.xy[:][0], p3.exterior.xy[:][1])
    self.longest_cont = array(contour_intersect)[1:]
    s    = "::: intersection contour created, length %s nodes :::"
    print_text(s % shape(self.longest_cont)[0], self.color)

  def extend_edge(self, r):
    """
    Extends a 2d contour out from points labeled in self.edge by a distance
    <r> (radius) in all directions.
    """
    xycoords = self.longest_cont

    polygons = []
    for i, v in enumerate(xycoords):
      polygons.append(shapelyPoint(v[0],v[1]).buffer(r))

    # union of our original polygon and convex hull
    p1 = cascaded_union(polygons)
    p3 = cascaded_union(p1)

    xycoords_buf = array(zip(p3.exterior.xy[:][0], p3.exterior.xy[:][1]))
    self.longest_cont = xycoords_buf

  def convert_msh_to_xml(self, mshfile, xmlfile):
    """
    convert <mshfile> .msh file to .xml file <xmlfile> via dolfin-convert.
    """
    msh = self.direc + mshfile + '.msh'
    xml = self.direc + xmlfile + '.xml'

    cmd = 'dolfin-convert ' + msh + ' ' + xml
    s   = "\nExecuting :\n\n\t %s\n\n" % cmd
    print_text(s, self.color)
    subprocess.call(cmd.split())


class linear_attractor(object):
  """
  Create an attractor object which refines with min and max cell radius <l_min>,
  <l_max> over data field <field>.  The <f_max> parameter specifies a max value
  for which to apply the minimum cell size such that if <field>_i is less than
  <f_max>, the cell size in this region will be <l_max>.  If <inv> = True
  the object refines on the inverse of the data field <field>.

               {l_min,     field_i > f_max
    cell_h_i = {l_max,     field_i < f_max
               {field_i,   otherwise

  """
  def __init__(self, spline, field, f_max, l_min, l_max, inv=True):
    """
    Refine the mesh off of data field <field> using spline <spline> with the
    cell radius defined as :

               {l_min,     field_i > f_max
    cell_h_i = {l_max,     field_i < f_max
               {field_i,   otherwise

    If <inv> is True, refine off of the inverse of <field> instead.

    """
    self.spline   = spline
    self.field    = field
    self.l_min    = l_min
    self.l_max    = l_max
    self.f_max    = f_max
    self.inv      = inv

  def op(self, x, y, z, entity):
    """
    """
    l_min = self.l_min
    l_max = self.l_max
    f     = self.field
    v     = self.spline(x,y)[0][0]
    if self.inv:
      if v < self.f_max:
        lc = l_max - (l_max - l_min) / f.max() * v
      else:
        lc = l_min
    else:
      if v < self.f_max:
        lc = l_min + (l_max - l_min) / f.max() * v
      else:
        lc = l_max
    return lc

class static_attractor(object):
  """
  """
  def __init__(self, spline, c, inv=False):
    """
    Refine the mesh off of data field <spline> with the cell radius
    defined as :

    cell_h_i = c * spline(x,y)

    """
    self.spline = spline
    self.c      = c
    self.inv    = inv

  def op(self, x, y, z, entity):
    """
    """
    if not self.inv:
      lc = self.c * self.spline(x,y)[0][0]
    else:
      lc = self.c * 1/self.spline(x,y)[0][0]
    return lc


class min_field(object):
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


class max_field(object):
  """
  Return the minimum of a list of attactor operator fields <f_list>.
  """
  def __init__(self, f_list):
    self.f_list = f_list

  def op(self, x, y, z, entity):
    l = []
    for f in self.f_list:
      l.append(f(x,y,z,entity))
    return max(l)

class MeshRefiner(object):

  def __init__(self, di, fn, gmsh_file_name):
    """
    Creates a 2D or 3D mesh based on contour .geo file <gmsh_file_name>.
    Refinements are done on DataInput object <di> with data field index <fn>.
    """
    self.color = '43'
    s    = "::: initializing MeshRefiner on \"%s.geo\" :::" % gmsh_file_name
    print_text(s, self.color)

    self.field  = di.data[fn].T
    print_min_max(self.field, 'refinement field [m]')

    self.spline = RectBivariateSpline(di.x, di.y, self.field, kx=1, ky=1)

    #load the mesh into a GModel
    self.m = GModel.current()
    self.m.load(gmsh_file_name + '.geo')

    # set some parameters :
    GmshSetOption("Mesh", "CharacteristicLengthFromPoints", 0.0)
    GmshSetOption("Mesh", "CharacteristicLengthExtendFromBoundary", 0.0)
    GmshSetOption("Mesh", "Smoothing", 100.0)

  def add_linear_attractor(self, f_max, l_min, l_max, inv):
    """
    Refine the mesh with the cell radius defined as :

               {l_min,     field_i > f_max
    cell_h_i = {l_max,     field_i < f_max
               {field_i,   otherwise

    If <inv> is True, refine off of the inverse of <field> instead.

    """
    # field, f_max, l_min, l_max, hard_cut=false, inv=true
    a   = linear_attractor(self.spline, self.field, f_max, l_min, l_max,
                           inv=inv)
    aid = self.m.getFields().addPythonField(a.op)
    return a,aid

  def add_static_attractor(self, c=1, inv=False):
    """
    Refine the mesh with the cell radius defined as :

    cell_h_i = c * field_i

    returns a tuple, static_attractor object and id number.

    """
    # field, f_max, l_min, l_max, hard_cut=false, inv=true
    a   = static_attractor(self.spline, c, inv)
    aid = self.m.getFields().addPythonField(a.op)
    return a,aid

  def add_min_field(self, op_list):
    """
    Create a miniumum field of attactor operator lists <op_list>.
    """
    mf  = min_field(op_list)
    mid = self.m.getFields().addPythonField(mf.op)
    return mid

  def set_background_field(self, idn):
    """
    Set the background field to that of field index <idn>.
    """
    self.m.getFields().setBackgroundFieldId(idn)

  def finish(self, gui=True, dim=3, out_file_name='mesh'):
    """
    Finish and create the .msh file.  If <gui> is True, run the gui program,
    Otherwise, create the .msh file with dimension <dim> and filename
    <out_file_name>.msh.
    """
    self.out_file_name = out_file_name

    #launch the GUI
    if gui:
      print_text("::: opening GUI :::", self.color)
      FlGui.instance().run()

    # instead of starting the GUI, we could generate the mesh and save it
    else:
      s    = "::: writing %s.msh :::" % out_file_name
      print_text(s, self.color)
      self.m.mesh(dim)
      self.m.save(out_file_name + ".msh")
  
  def convert_msh_to_xml(self):
    """
    convert <mshfile> .msh file to .xml file <xmlfile> via dolfin-convert.
    """
    msh = self.out_file_name + '.msh'
    xml = self.out_file_name + '.xml'

    cmd = 'dolfin-convert ' + msh + ' ' + xml
    s   = "\nExecuting :\n\n\t %s\n\n" % cmd
    print_text(s, self.color)
    subprocess.call(cmd.split())
    
    cmd = 'gzip -f ' + xml
    s   = "\nExecuting :\n\n\t %s\n\n" % cmd
    print_text(s, self.color)
    subprocess.call(cmd.split())


class MeshExtruder(object):
  """
  Due to extreme bugginess in the gmsh extrusion utilities, this class
  extrudes a 2D mesh footprint in the z direction in an arbitrary number of
  layers.  Its primary purpose is to facilitate mesh generation for the
  ice sheet model VarGlaS.  Method based on HOW TO SUBDIVIDE PYRAMIDS, PRISMS
  AND HEXAHEDRA INTO TETRAHEDRA by Dompierre et al.

  Written by Douglas Brinkerhoff 14.01.25
  """

  indirection_table = {0:[0,1,2,3,4,5],
                       1:[1,2,0,4,5,3],
                       2:[2,0,1,5,3,4],
                       3:[3,5,4,0,2,1],
                       4:[4,3,5,1,0,3],
                       5:[5,4,3,2,1,0]}

  def __init__(self,mesh):
    # Accepts a dolfin mesh of dimension 2
    self.mesh = mesh
    self.n_v2 = mesh.num_vertices()

    # Initialize tetrahedron array for extruded mesh
    self.global_tets = array([-1,-1,-1,-1])

  def extrude_mesh(self,l,z_offset):
    # accepts the number of layers and the length of extrusion

    mesh = self.mesh

    # Extrude vertices
    all_coords = []
    for i in linspace(0,z_offset,l):
      all_coords.append(hstack((mesh.coordinates(),i*ones((self.n_v2,1)))))
    self.global_vertices = vstack(all_coords)

    # Extrude cells (tris to tetrahedra)
    for i in range(l-1):
      for c in self.mesh.cells():
        # Make a prism out of 2 stacked triangles
        vertices = hstack((c+i*self.n_v2,c+(i+1)*self.n_v2))

        # Determine prism orientation
        smallest_vertex_index = argmin(vertices)

        # Map to I-ordering of Dompierre et al.
        mapping = self.indirection_table[smallest_vertex_index]

        # Determine which subdivision scheme to use.
        if min(vertices[mapping][[1,5]]) < min(vertices[mapping][[2,4]]):
          local_tets = vstack((vertices[mapping][[0,1,2,5]],\
                               vertices[mapping][[0,1,5,4]],\
                               vertices[mapping][[0,4,5,3]]))
        else:
          local_tets = vstack((vertices[mapping][[0,1,2,4]],\
                               vertices[mapping][[0,4,2,5]],\
                               vertices[mapping][[0,4,5,3]]))
        # Concatenate local tet to cell array
        self.global_tets = vstack((self.global_tets,local_tets))

    # Eliminate phantom initialization tet
    self.global_tets = self.global_tets[1:,:]

    # Query number of vertices and tets in new mesh
    self.n_verts = self.global_vertices.shape[0]
    self.n_tets = self.global_tets.shape[0]

    # Initialize new dolfin mesh of dimension 3
    self.new_mesh = Mesh()
    m = MeshEditor()
    m.open(self.new_mesh,3,3)
    m.init_vertices(self.n_verts,self.n_verts)
    m.init_cells(self.n_tets,self.n_tets)

    # Copy vertex data into new mesh
    for i,v in enumerate(self.global_vertices):
      m.add_vertex(i,Point(*v))

    # Copy cell data into new mesh
    for j,c in enumerate(self.global_tets):
      m.add_cell(j,*c)

    m.close()

  def write_mesh_to_file(self,filename):
    # Output mesh
    File(filename) << self.new_mesh

class GetBasin(object):
  """
  This class contains functions to return a contour corresponding to the
  perimeter of various basins in Antarctica and Greenland. The libraries of
  basins are drawn from ICESat data, and posted here:

  http://icesat4.gsfc.nasa.gov/cryo_data/ant_grn_drainage_systems.php

  INPUTS:
    di :
      an instance of a DataInput obect (see above) needed for the projection
      function

    edeg_resolution:
      distance between points on boundary

    basin:
      basin number. If left as None, the program will prompt you to pick a basin

  TODO: Now working to extend the domain beyond the present day ice margin for
  the purpose of increasing the stability of dynamic runs. Additionally, there
  appear to be some stability issues when running the MCB algorithm, but these
  are not consistent; some domains work, others do not. The hope is that
  extension of the domain will help here too.

  """
  def __init__(self, di, basin=None, edge_resolution=1000.0):
    """
    """
    self.color  = 'grey_46'
    self.di     = di

    s    = "::: INITIALIZING BASIN GENERATOR :::"
    print_text(s, self.color)

    self.edge_resolution = edge_resolution
    self.plot_coords     = {}

    # Get path of this file, which should be in the src directory
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename)) + "/.."

    if di.cont == "greenland":
      path           = home + "/data/greenland/basins/"
      self.datafile  = path + "GrnDrainageSystems_Ekholm.txt"
      self.imagefile = path + "Grn_Drainage_Systems.png"
    elif di.cont == "antarctica":
      path           = home + "/data/antarctica/basins/"
      self.datafile  = path + "Ant_Full_DrainageSystem_Polygons.txt"
      self.imagefile = path + "Ant_ICESatDSMaps_Fig_1.jpg"
    else:
      s = "Can not find data corresponding to location %s" % di.cont
      print_text(s, 'red', 1)

    if basin == None:
      self.show_and_get_basin()
    else:
      self.basin = basin
    self.retrive_basin_latlong()
    self.convert_to_projection()

  def show_and_get_basin(self):
    """
    """
    print_text(self.imagefile, self.color)
    image = Image.open(self.imagefile)
    image.show()
    self.basin = raw_input("Input the numerical code of the basin.\n")

  def retrive_basin_latlong(self):
    """
    """
    self.llcoords = []
    if self.di.cont == 'antarctica':
      id  = 2
      lat = 0
      lon = 1
    elif self.di.cont == 'greenland':
      id  = 0
      lat = 1
      lon = 2

    f = open(self.datafile)
    for line in f:
      sl = line.split()
      if sl[id] == self.basin:
        self.llcoords.append([sl[lon],sl[lat]])
    self.llcoords = array(self.llcoords)

  def convert_to_projection(self):
    """
    """
    self.xycoords = []
    self.edge     = []
    p = self.llcoords[0,:] # previous point
    self.xycoords.append(self.di.get_xy(p[0],p[1]))
    self.edge.append(True)
    p_p = self.xycoords[-1]
    distance = 0

    lin_dist = lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    for p in self.llcoords:
      p_n = self.di.get_xy(p[0],p[1]) # Current point xy
      delta_X = lin_dist(p_n, p_p)
      distance += delta_X

      if distance > self.edge_resolution:
        if delta_X > 500.:            # edge points are further apart
          self.edge.append(True)
        else:
          self.edge.append(False)
        self.xycoords.append(p_n)
        distance = 0.
        p_p = p_n
      else:
        p_p = p_n

    """
    # remove points at end of array that may overlap
    while(len(self.xycoords) > 0 and \
          lin_dist(self.xycoords[0],self.xycoords[-1]) < self.edge_resolution):
      self.xycoords.pop()
      self.edge.pop()
    """

    self.xycoords = array(self.xycoords)
    self.plot_coords["xycoords"] = self.xycoords

    #self.clean_edge() #clean (very rare) incorrectly identified edge points
    self.edge = array(self.edge)

  def clean_edge(self):
    """
    Remove spurious edge markers. Not very common, but they do happen.
    """
    edge  = self.edge
    check = 5

    def check_n(i, l, n, check_f):
      """
      Return True if for at least <n> points on either side of a given
      index check_f(l[i]) returns True. Array will be assumed to be
      circular, i.e. l[len(l)] will be converted to l[0], and
      l[len(l)+1] will be converted to [1]
      """
      g = lambda i: i%len(l)

      behind = sum([check_f( l[g(i-(j+1))] ) for j in range(n)]) == n
      front  = sum([check_f( l[g(i+j+1)] ) for j in range(n)]) == n

      return front or behind

    # For every edge point make sure that at least <check> points on either
    # side are also edge Points.
    for i in range(len(edge)):
      if edge[i]:
        if not check_n(i, edge, check, lambda v: v):
          edge[i] = False

  def check_dist(self):
    """
    remove points in xycoords that are not a linear distance of at least
    <dist> from previous point.
    """
    lin_dist = lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    edge     = self.edge
    xycoords = self.xycoords
    n        = len(xycoords)

    mask = ones(n, dtype=bool)

    i = 0
    while(i < n-1):
      p1 = xycoords[i]
      j = i + 1
      while(j < n and lin_dist(p1, xycoords[j]) < self.edge_resolution):
        mask[j] = 0
        j += 1
      i = j

    # fix end of array
    i = -1
    while(n + i >= 0 and (not mask[i] or \
          lin_dist(xycoords[0],xycoords[i]) < self.edge_resolution)):
      mask[i] = 0
      i -= 1

    #for i in range(0,n-2):
    #  p1 = xycoords[i]
    #  p2 = xycoords[i+1]
    #  if lin_dist(p1, p2) < self.edge_resolution:
    #    mask[i] = 0

    # print results
    s    = "::: removed %s points closer than %s m to one another :::"% \
            (str(len(mask) - sum(mask)), self.edge_resolution)
    print_text(s, self.color)

    self.xycoords = xycoords[mask]

  def extend_edge(self, r):
    """
    Extends a 2d contour out from points labeled in self.edge by a distance
    <r> (radius) in all directions.
    """
    xycoords = self.xycoords
    edge = self.edge

    polygons = []
    for i, v in enumerate(xycoords):
      if edge[i]:
        polygons.append(shapelyPoint(v[0],v[1]).buffer(r))

    # union of our original polygon and convex hull
    p1 = cascaded_union(polygons)
    p2 = Polygon(zip(xycoords[:,0],xycoords[:,1]))
    p3 = cascaded_union([p1,p2])

    xycoords_buf = array(zip(p3.exterior.xy[:][0], p3.exterior.xy[:][1]))
    self.plot_coords["xycoords_buf"] = xycoords_buf
    self.xycoords = xycoords_buf

  def intersection(self, other):
    """
    Take the geometric intersection of current coordinates with <other>.
    Used primarily to replace the edge with something from a different
    (better) data set.

    NOTE: it's probably better to extend the boundary before taking the
    intersection.
    """
    s    = "::: taking intersection with new contour of length %i :::"
    print_text(s % len(other), self.color)

    xycoords = self.xycoords

    p1 = Polygon( zip(xycoords[:,0], xycoords[:,1]) )
    p2 = Polygon( zip(other[:,0],    other[:,1]   ) )

    intersection = p1.intersection(p2)

    # check if multi-polygon is created. If so, take polygon with greatest
    # area
    import collections
    if isinstance(intersection, collections.Iterable):
      p3 = max(intersection, key = lambda x: x.area)
    else:
      p3 = intersection

    xycoords_intersect = array(zip(p3.exterior.xy[:][0], \
                                   p3.exterior.xy[:][1]))

    self.plot_coords["xycoords_intersect"] = xycoords_intersect
    self.xycoords = xycoords_intersect
    
    s    = "::: intersection createed with length %i :::"
    print_text(s % len(self.xycoords), self.color)

  def plot_xycoords_buf(self, other=None):
    """
    """
    fig = figure()
    ax  = fig.add_subplot(111)
    ax.set_aspect("equal")

    # plot other
    if other != None:
      ax.plot(other[:,0], other[:,1], 'g', lw=3.0)

    # plot buffered coordinates
    if "xycoords_buf" in self.plot_coords:
      xycoords_buf = self.plot_coords["xycoords_buf"]
      ax.plot(xycoords_buf[:,0], xycoords_buf[:,1], 'b', lw=2.5)

    # plot original data
    xycoords = self.plot_coords["xycoords"]
    ax.plot(xycoords[:,0], xycoords[:,1], 'r', lw=2.5)

    #from numpy import ma
    #interior = ma.masked_array(xycoords, array([zip(self.edge,self.edge)]))
    #ax.plot(interior[:,0], interior[:,1], 'k', lw=3.0)

    # plot intersection
    if "xycoords_intersect" in self.plot_coords:
      xycoords_intersect = self.plot_coords["xycoords_intersect"]
      ax.plot(xycoords_intersect[:,0], xycoords_intersect[:,1], 'c', lw=8)

    ax.set_title("boundaries")
    show()

  def get_xy_contour(self):
    """
    """
    return self.xycoords



