"""
Utilities file:

  This contains classes that are used by UM-FEISM to aid in the loading
  of data and preparing data for use in DOLFIN based simulation.
  Specifically the data are projected onto the mesh with appropriate
  basis functions.

"""
import subprocess
from gmshpy            import GModel, GmshSetOption, FlGui
from scipy.interpolate import RectBivariateSpline
from pylab             import array, linspace, ones, meshgrid, figure, show, \
                              size, hstack, vstack, argmin, zeros, shape
from fenics            import Mesh, MeshEditor, Point, File
#from termcolor         import colored, cprint
from pyproj            import transform
from io                import print_text, print_min_max


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
    self.f          = open(direc + fn + '.geo', 'w')
    self.fieldList  = []  # list of field indexes created.

  def create_contour(self, var, zero_cntr, skip_pts):
    """
    Create a contour of the data field with index <var> of <dd> provided at
    initialization.  <zero_cntr> is the value of <var> to contour, <skip_pts>
    is the number of points to skip in the contour, needed to prevent overlap.
    """
    s    = "::: creating contour from %s's \"%s\" field :::"
    print_text(s % (self.dd.name, var) , self.color)
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
    longest_cont      = cl[amax_ind]
    self.longest_cont = longest_cont[::skip_pts,:][:-1,:]
    s    = "::: contour created, length %s nodes :::"
    print_text(s % shape(self.longest_cont)[0], self.color)

  def transform_contour(self, di):
    """
    Transforms the coordinates of the contour to DataInput object <di>'s
    projection coordinates.
    """
    s = "::: transforming contour coordinates from %s to %s :::"
    print_text(s % (di.name, self.dd.name), self.color)
    x,y    = self.longest_cont.T
    xn, yn = transform(self.dd.p, di.p, x, y)
    self.longest_cont = array([xn, yn]).T

  def set_contour(self,cont_array):
    """ This is an alternative to the create_contour method that allows you to
    manually specify contour points.
    Inputs:
    cont_array : A numpy array of contour points (i.e. array([[1,2],[3,4],...]))
    """
    s = "::: manually setting contour :::"
    fig = figure()
    self.ax = fig.add_subplot(111)
    self.ax.set_aspect('equal')
    print_text(s, self.color)
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
    how far forward to look to eliminate intersections.
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

    counter  = 0
    new_cont = zeros((sum(flag),2))
    for ii,fl in enumerate(flag):
      if fl:
        new_cont[counter,:] = lc[ii,:]
        counter += 1

    self.longest_cont = new_cont
    s    = "::: eliminated %i nodes :::"
    print_text(s % sum(flag == 0), self.color)

  def restart(self):
    """
    clear all contents from the .geo file.
    """
    self.f.close
    self.f = open(self.direc + self.fn + '.geo', 'w')
    print 'Reopened \"' + self.direc + self.fn + '.geo\".'

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

    print 'finished, closing \"' + self.direc + self.fn + '.geo\".'
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
    print "\nExecuting :\n\n\t", cmd, "\n\n"
    subprocess.call(cmd.split())


  def convert_msh_to_xml(self, mshfile, xmlfile):
    """
    convert <mshfile> .msh file to .xml file <xmlfile> via dolfin-convert.
    """
    msh = self.direc + mshfile + '.msh'
    xml = self.direc + xmlfile + '.xml'

    cmd = 'dolfin-convert ' + msh + ' ' + xml
    print "\nExecuting :\n\n\t", cmd, "\n\n"
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
    self.color = 'grey_74'
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

