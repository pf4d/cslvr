"""
Utilities file:

  This contains classes that are used by UM-FEISM to aid in the loading
  of data and preparing data for use in DOLFIN based simulation.
  Specifically the data are projected onto the mesh with appropriate
  basis functions.
"""
import inspect
import os
import Image
from scipy.io          import loadmat, savemat
from scipy.interpolate import RectBivariateSpline
from pylab             import array, shape, linspace, ones, isnan, all, zeros, \
                              ndarray, e, nan, sqrt, float64, sin, cos, pi,\
                              figure, show
from fenics            import interpolate, Expression, Function, \
                              vertices, FunctionSpace, RectangleMesh, \
                              MPI, mpi_comm_world, GenericVector, parameters, \
                              File
from pyproj            import Proj, transform
#from termcolor         import colored, cprint
from colored           import fg, attr
from scipy.spatial     import ConvexHull
from shapely.geometry  import Polygon
from shapely.ops       import cascaded_union

class DataInput(object):
  """
  This object brokers the relation between the driver file and a number of
  data sets. It's function is to:

    1) Read the data. Presently it is assumed that all input is Matlab V5.
    2) Filter or process the data. Presently the only filter is to remove
       rows or columns in key data sets that are entirely not a number.
    3) Project the data onto a finite element mesh that is generated based
       on the extents of the input data set.
  """
  def __init__(self, files, direc=None, flip=False, mesh=None, gen_space=True,
               zero_edge=False, bool_data=False, req_dg=False):
    """
    The following data are used to initialize the class :

      direc     : Set the directory containing the input files.
      files     : Tuple of file names.  All files are scanned for rows or
                  columns of nans. Assume all files have the same extents.
      flip      : flip the data over the x-axis?
      mesh      : FEniCS mesh if there is one already created.
      zero_edge : Make edges of domain -0.002?
      bool_data : Convert data to boolean?
      req_dg    : Some field may require DG space?

    Based on thickness extents, create a rectangular mesh object.
    Also define the function space as continious galerkin, order 1.
    """
    self.directory  = direc
    self.data       = {}        # dictionary of converted matlab data
    self.rem_nans   = False
    self.chg_proj   = False     # change to other projection flag
    self.color      = 'light_green'

    first = True  # initialize domain by first file's extents

    if direc == None and type(files) == dict:
      self.name = files.pop('dataset')
    elif direc != None:
      self.name = direc

    s    = "::: creating %s DataInput object :::" % self.name
    print_text(s, self.color)

    # process the data files :
    for fn in files:

      if direc == None and type(files) == dict:
        d_dict = files[fn]


      elif direc != None:
        d_dict = loadmat(direc + fn)
        d_dict['projection']     = d_dict['projection'][0]
        d_dict['standard lat']   = d_dict['standard lat'][0]
        d_dict['standard lon']   = d_dict['standard lon'][0]
        d_dict['lat true scale'] = d_dict['lat true scale'][0]

      d = d_dict["map_data"]

      # initialize extents :
      if first:
        self.ny,self.nx = shape(d_dict['map_data'])
        self.x_min      = float(d_dict['map_western_edge'])
        self.x_max      = float(d_dict['map_eastern_edge'])
        self.y_min      = float(d_dict['map_southern_edge'])
        self.y_max      = float(d_dict['map_northern_edge'])
        self.proj       = str(d_dict['projection'])
        self.lat_0      = str(d_dict['standard lat'])
        self.lon_0      = str(d_dict['standard lon'])
        self.lat_ts     = str(d_dict['lat true scale'])
        self.x          = linspace(self.x_min, self.x_max, self.nx)
        self.y          = linspace(self.y_min, self.y_max, self.ny)
        self.good_x     = array(ones(len(self.x)), dtype=bool)      # no NaNs
        self.good_y     = array(ones(len(self.y)), dtype=bool)      # no NaNs
        first           = False

      # identify, but not remove the NaNs :
      self.identify_nans(d, fn)

      # make edges all zero for interpolation of interior regions :
      if zero_edge:
        d[:,0] = d[:,-1] = d[0,:] = d[-1,:] = -0.002
        d[:,1] = d[:,-2] = d[1,:] = d[-2,:] = -0.002

      # convert to boolean :
      if bool_data: d[d > 0] = 1

      # reflect over the x-axis :
      if flip: d = d[::-1, :]

      # add to the dictionary of arrays :
      self.data[fn.split('.')[0]] = d

    # remove un-needed rows/cols from data:
    if self.rem_nans:
      self.remove_nans()

    if gen_space:
      # define a FEniCS Rectangle over the domain :
      if mesh == None:
        self.mesh = RectangleMesh(self.x_min, self.y_min,
                                  self.x_max, self.y_max,
                                  self.nx,    self.ny)
      else:
        self.mesh = mesh

      # define the function space of the problem :
      self.func_space      = FunctionSpace(self.mesh, "CG", 1)

      # if DG space is needed :
      if req_dg:
        self.func_space_dg = FunctionSpace(self.mesh, "DG", 1)

    # create projection :
    proj =   " +proj="   + self.proj \
           + " +lat_0="  + self.lat_0 \
           + " +lat_ts=" + self.lat_ts \
           + " +lon_0="  + self.lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    self.p = Proj(proj)

  def change_projection(self, di):
    """
    change the projection of this data to that of the <di> DataInput object's
    projection.  The works only if the object was created with the parameter
    create_proj = True.
    """
    self.chg_proj = True
    self.new_p    = di.p

  def get_xy(self,lon,lat):
    """
    Returns the (x,y) flat map coordinates corresponding to a given (lon,lat)
    coordinate pair using the DataInput object's current projection."""
    return self.p(lon,lat)

  def transform_xy(self, di):
    """
    Transforms the coordinates from DataInput object <di> to this object's
    coordinates.  Returns tuple of arrays (x,y).
    """
    # FIXME : need a fast way to convert all the x, y. Currently broken
    s = "::: transforming coordinates from %s to %s :::" % (di.name, self.name)
    print_text(s, self.color)
    xn, yn = transform(di.p, self.p, di.x, di.y)
    return (xn, yn)

  def integrate_field(self, fn_spec, specific, fn_main, r=20, val=0.0):
    """
    Assimilate a field with filename <fn_spec>  from DataInput object
    <specific> into this DataInput's field with filename <fn_main>.  The
    parameter <val> should be set to the specific dataset's value for
    undefined regions, default is 0.0.  <r> is a parameter used to eliminate
    border artifacts from interpolation; increase this value to eliminate edge
    noise.
    """
    s    = "::: integrating %s field from %s :::" % (fn_spec, specific.name)
    print_text(s, self.color)
    # get the dofmap to map from mesh vertex indices to function indicies :
    df    = self.func_space.dofmap()
    dfmap = df.vertex_to_dof_map(self.mesh)

    unew  = self.get_projection(fn_main)      # existing dataset projection
    uocom = unew.compute_vertex_values()      # mesh indexed main vertex values

    uspec = specific.get_projection(fn_spec)  # specific dataset projection
    uscom = uspec.compute_vertex_values()     # mesh indexed spec vertex values

    d     = float64(specific.data[fn_spec])   # original matlab spec dataset

    # get arrays of x-values for specific domain
    xs    = specific.x
    ys    = specific.y
    nx    = specific.nx
    ny    = specific.ny

    for v in vertices(self.mesh):
      # mesh vertex x,y coordinate :
      i   = v.index()
      p   = v.point()
      x   = p.x()
      y   = p.y()

      # indexes of closest datapoint to specific dataset's x and y domains :
      idx = abs(xs - x).argmin()
      idy = abs(ys - y).argmin()

      # data value for closest value and square around the value in question :
      dv  = d[idy, idx]
      db  = d[max(0,idy-r) : min(ny, idy+r),  max(0, idx-r) : min(nx, idx+r)]

      # if the vertex is in the domain of the specific dataset, and the value
      # of the dataset at this point is not abov <val>, set the array value
      # of the main file to this new specific region's value.
      if dv > val:
        #print "found:", x, y, idx, idy, v.index()
        # if the values is not near an edge, make the value equal to the
        # nearest specific region's dataset value, otherwise, use the
        # specific region's projected value :
        if all(db > val):
          uocom[i] = uscom[i]
        else :
          uocom[i] = dv

    # set the values of the projected original dataset equal to the assimilated
    # dataset :
    unew.vector().set_local(uocom[dfmap])
    return unew

  def identify_nans(self, data, fn):
    """
    private method to identify rows and columns of all nans from grids. This
    happens when the data from multiple GIS databases don't quite align on
    whatever the desired grid is.
    """
    good_x = ~all(isnan(data), axis=0) & self.good_x  # good cols
    good_y = ~all(isnan(data), axis=1) & self.good_y  # good rows

    if any(good_x != self.good_x):
      total_nan_x = sum(good_x == False)
      self.rem_nans = True
      s =  "Warning: %d row(s) of \"%s\" are entirely NaN." % (total_nan_x, fn)
      print_text(s, self.color)

    if any(good_y != self.good_y):
      total_nan_y = sum(good_y == False)
      self.rem_nans = True
      s = "Warning: %d col(s) of \"%s\" are entirely NaN." % (total_nan_y, fn)
      print_text(s, self.color)

    self.good_x = good_x
    self.good_y = good_y

  def remove_nans(self):
    """
    remove extra rows/cols from data where NaNs were identified and set the
    extents to those of the good x and y values.
    """
    s = "::: removing NaNs from %s :::" % self.name
    print_text(s, self.color)

    self.x     = self.x[self.good_x]
    self.y     = self.y[self.good_y]
    self.x_min = self.x.min()
    self.x_max = self.x.max()
    self.y_min = self.y.min()
    self.y_max = self.y.max()
    self.nx    = len(self.x)
    self.ny    = len(self.y)

    for i in self.data.keys():
      self.data[i] = self.data[i][self.good_y, :          ]
      self.data[i] = self.data[i][:,           self.good_x]

  def set_data_min(self, fn, boundary, val):
    """
    set the minimum value of a data array with filename <fn> below <boundary>
    to value <val>.
    """
    d                = self.data[fn]
    d[d <= boundary] = val
    self.data[fn]    = d

  def set_data_max(self, fn, boundary, val):
    """
    set the maximum value of a data array with filename <fn> above <boundary>
    to value <val>.
    """
    d                = self.data[fn]
    d[d >= boundary] = val
    self.data[fn]    = d

  def set_data_val(self, fn, old_val, new_val):
    """
    set all values of the matrix with filename <fn> equal to <old_val>
    to <new_val>.
    """
    d                = self.data[fn]
    d[d == old_val]  = new_val
    self.data[fn]    = d

  def get_interpolation(self, fn, near=False, bool_data=False, kx=3, ky=3):
    """
    Return a projection of data with field name <fn> on the functionspace.
    If multiple instances of the DataInput class are present, both initialized
    with identical meshes, the projections returned by this function may be
    used by the same mathematical problem.

    If <bool_data> is True, convert all values > 0 to 1.
    """
    if near:
      t = 'nearest-neighbor'
    else:
      t = 'spline'
    s    = "::: getting %s %s interpolation from %s :::" % (fn, t, self.name)
    print_text(s, self.color)

    interp = self.get_expression(fn, kx=kx, ky=ky,
                                 bool_data=bool_data, near=near)

    return interpolate(interp, self.func_space)

  def get_expression(self, fn, kx=3, ky=3, bool_data=False, near=False):
    """
    Creates a spline-interpolation expression for data <fn>.  Optional
    arguments <kx> and <ky> determine order of approximation in x and y
    directions (default cubic).  If <bool_data> is True, convert to boolean,
    if <near> is True, use nearest-neighbor interpolation.
    """
    if near:
      t = 'nearest-neighbor'
    else:
      t = 'spline'
    s = "::: getting %s %s expression from %s :::" % (fn, t, self.name)
    print_text(s, self.color)

    data = self.data[fn]
    if bool_data: data[data > 0] = 1

    if self.chg_proj:
      new_proj = self.new_p
      old_proj = self.p

    if not near :
      spline = RectBivariateSpline(self.x, self.y, data.T, kx=kx, ky=ky)

    xs       = self.x
    ys       = self.y
    chg_proj = self.chg_proj

    class newExpression(Expression):
      def eval(self, values, x):
        if chg_proj:
          xn, yn = transform(new_proj, old_proj, x[0], x[1])
        else:
          xn, yn = x[0], x[1]
        if not near:
          values[0] = spline(xn, yn)
        else:
          idx       = abs(xs - xn).argmin()
          idy       = abs(ys - yn).argmin()
          values[0] = data[idy, idx]

    return newExpression(element = self.func_space.ufl_element())

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


class DataOutput(object):

  def __init__(self, directory):
    """
    Create object to write data to directory <directory>
    """
    self.directory = directory
    self.color     = 'orange_3'

  def write_dict_of_files(self, d, extension='.pvd'):
    """
    Looking for a dictionary <d> of data to save. The keys are the file
    names, and the values are the data fields to be stored. Also takes an
    optional extension to determine if it is pvd or xml output.
    """
    for fn in d:
      self.write_one_file(fn, d[fn], extension)

  def write_one_file(self, name, data, extension='.pvd'):
    """
    Save a single file of FEniCS Function <data> named <name> to the DataOutput
    instance's directory.  Extension may be '.xml' or '.pvd'.
    """
    s    = "::: writing file %s :::" % (name + extension)
    print_text(s, self.color)
    file_handle = File(self.directory + name + extension)
    file_handle << data

  def write_matlab(self, di, f, filename, val=e, size=None):
    """
    Using the projections that are read in as data files, create Matlab
    version 4 files to output the regular gridded data in a field.  Will accept
    functions in 2D or 3D; if a 3D mesh is used, Ensure that value you want
    projected is located at z=0, the bed.  This can be accomplished by using
    any of the non-deformed flat meshes provided by the MeshFactory class.

    INPUTS:
      di       : a DataInput object, defined in the class above in this file.
      f        : a FEniCS function, to be mapped onto the regular grid that is
                 in di, established from the regular gridded data to start the
                 simulation.
      filename : a file name for the matlab file output (include the
                 extension) values not in mesh are set to <val>.
      val      : value to make values outside of mesh.  Default is 'e'.
    OUTPUT:
      A single file will be written with name, outfile.
    """
    fa   = zeros( (di.ny, di.nx) )
    s    = "::: writing %i x %i matlab matrix file %s.mat :::"
    text = s % (di.ny, di.nx, filename)
    print_text(text, self.color)
    parameters['allow_extrapolation'] = True
    dim = f.geometric_dimension()
    for j,x in enumerate(di.x):
      for i,y in enumerate(di.y):
        try:
          if dim == 3:
            fa[i,j] = f(x,y,0)
          else:
            fa[i,j] = f(x,y)
        except:
          fa[i,j] = val
    print_min_max(fa, filename + 'matrix')
    outfile = self.directory + filename + '.mat'
    savemat(outfile, {'map_data'          : fa,
                      'map_eastern_edge'  : di.x_max,
                      'map_western_edge'  : di.x_min,
                      'map_northern_edge' : di.y_max,
                      'map_southern_edge' : di.y_min,
                      'map_name'          : outfile,
                      'projection'        : di.proj,
                      'standard lat'      : di.lat_0,
                      'standard lon'      : di.lon_0,
                      'lat true scale'    : di.lat_ts})


def print_min_max(u, title, color='yellow'):
  """
  Print the minimum and maximum values of <u>, a Vector, Function, or array.
  """
  if isinstance(u, GenericVector):
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
  elif isinstance(u, Function):
    uMin = MPI.min(mpi_comm_world(), u.vector().min())
    uMax = MPI.max(mpi_comm_world(), u.vector().max())
  elif isinstance(u, ndarray):
    er = "warning, input to print_min_max() is a NumPy array, local min " + \
         " / max only"
    er = ('%s%s' + er + '%s') % (fg('red'), attr(1), attr(0))
    print er
    uMin = u.min()
    uMax = u.max()
  elif isinstance(u, int) or isinstance(u, float):
    uMin = uMax = u
  else:
    if MPI.rank(mpi_comm_world())==0:
      er = "print_min_max function requires a Vector, Function, array," \
           + " int or float, not %s." % type(u)
      er = ('%s%s' + er + '%s') % (fg('red'), attr(1), attr(0))
      print er
    uMin = uMax = nan
  if MPI.rank(mpi_comm_world())==0:
    s    = title + ' <min, max> : <%f, %f>' % (uMin, uMax)
    text = ('%s' + s + '%s') % (fg(color), attr(0))
    print text


def print_text(text, color='white', atrb=0):
  """
  Print text <text> from calling class <cl> to the screen.
  """
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    print text


class GetBasin(object):
  """This class contains functions to return a contour corresponding to the
  perimeter of various basins in Antarctica and Greenland. The libraries of
  basins are drawn from ICESat data, and posted here:

  http://icesat4.gsfc.nasa.gov/cryo_data/ant_grn_drainage_systems.php

  INPUTS:
              di :     an instance of a DataInput obect (see above) needed for
                       projection function

              where : 'Greenland' if contour is to be from Greenland, 
                       False for Antarctica

  TODO: Now working to extend the domain beyond the present day ice margin for
  the purpose of increasing the stability of dynamic runs. Additionally, there
  appear to be some stability issues when running the MCB algorithm, but these
  are not consistent; some domains work, others do not. The hope is that
  extension of the domain will help here too.

  """
  def __init__(self,di,where = 'Greenland',edge_resolution = 1000.):
    """
    INPUTS:
      di :              data input object
      where :           Greenland or Antarctica
      edge_resolution : meters between points that are pulled from the
                        database of domain edges.
    """
    self.extend = False
    self.di = di
    self.where = where
    self.edge_resolution = edge_resolution

    # Get path of this file, which should be in the src directory
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))

    if where == "Greenland":
      path = home+"/data/greenland/basins/"
      self.datafile = path+"GrnDrainageSystems_Ekholm.txt"
      self.imagefile =path+"Grn_Drainage_Systems.png"
    elif where == "Antarctica":
      path = home+"/data/antarctica/basins/"
      self.datafile = path+"Ant_Grounded_DrainageSystem_Polygons.txt"
      self.imagefile =path+"Ant_ICESatDSMaps_Fig_1_sm.jpg"
    else:
      print "Can not find data corresponding to location "+where+"."

    self.show_and_get_basin()
    self.retrive_basin_latlong()
    self.convert_to_projection()

  def show_and_get_basin(self):
    print self.imagefile
    image = Image.open(self.imagefile)
    image.show()
    self.basin = raw_input("Input the numerical code of the basin.\n")

  def retrive_basin_latlong(self):
    self.llcoords = []
    if self.where == "Antarctica":
      id  = 2
      lat = 0
      lon = 1
    else:
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
    self.xycoords = []
    self.edge     = []
    p = self.llcoords[0,:] # previous point
    self.xycoords.append(self.di.get_xy(p[0],p[1]))
    self.edge.append(True)
    p_p = self.xycoords[-1]
    distance = 0

    for p in self.llcoords:
      p_n = self.di.get_xy(p[0],p[1]) # Current point xy
      delta_X = sqrt((p_n[0] - p_p[0])**2 + (p_n[1] - p_p[1])**2)
      distance += delta_X

      if distance > self.edge_resolution:
        # Simple observation that edge points are further apart
        if delta_X > 500.:
          self.edge.append(True)
        else:
          self.edge.append(False)
        self.xycoords.append(p_n)
        distance = 0.
        p_p = p_n
      else:
        p_p = p_n

    self.xycoords = array(self.xycoords)
    self.clean_edge() #clean (very rare) incorrectly identified edge points
    self.edge = array(self.edge)

  def clean_edge(self):
    """
    Remove spurious edge markers. Not very common but do happen.
    """
    edge = self.edge

    def check_n(i, l, n, check_f):
      """
      Return True if for at least <n> points on either side of a given
      index check_f(l[i]) returns True. Array will be assumed to be
      circular, i.e. l[len(l)] will be converted to l[0]
      """
      g = lambda i: i%len(l)

      behind = sum([check_f( l[g(i-(j+1))] ) for j in range(n)]) == n
      front  = sum([check_f( l[g(i+j+1)] ) for j in range(n)]) == n

      return behind or front

    # For every edge point make sure that at least 5 points on either side
    # are also edge Points.
    for i in range(len(edge)):
      if edge[i]:
        if not check_n(i, edge, 5, lambda v: v):
          edge[i] = False

  def extend_edge(self, r):
    """
    Extends a 2d contour out from points labeled in self.edge by a distance
    <r> (radius) in all directions.
    """
    xycoords = self.xycoords
    edge = self.edge

    def points_circle(x, y, r,n=100):
      xo = [x + cos(2*pi/n*i)*r for i in range(0,n+1)]
      yo = [y + sin(2*pi/n*j)*r for j in range(0,n+1)]
      return array(zip(xo,yo))

    # create points in a circle around each edge point
    pts = []
    for i,v  in enumerate(xycoords):
      if edge[i]:
        pts.extend(points_circle(v[0], v[1], r))

    # take convex hull
    pts  = array(pts)
    hull = ConvexHull(pts)

    # union of our original polygon and convex hull
    p1 = Polygon(zip(pts[hull.vertices,0],pts[hull.vertices,1]))
    p2 = Polygon(zip(xycoords[:,0],xycoords[:,1]))
    p3 = cascaded_union([p1,p2])

    self.extend = True
    self.xycoords_buf = array(zip(p3.exterior.xy[:][0],p3.exterior.xy[:][1]))


  def plot_xycoords_buf(self, Show=True):
    fig = figure()
    ax  = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.plot(self.xycoords_buf[:,0], self.xycoords_buf[:,1], 'b', lw=2.5)
    ax.plot(self.xycoords[:,0], self.xycoords[:,1], 'r', lw=2.5)
    from numpy import ma
    interior  = ma.masked_array(self.xycoords,array([zip(self.edge,self.edge)]))
    ax.plot(interior[:,0], interior[:,1], 'k', lw=3.0)
    ax.set_title("boundaries")
    if Show:
      show()

  def get_xy_contour(self):
    if self.extend:
      return self.xycoords_buf
    else:
      return self.xycoords

  def get_edge(self):
    return self.edge



