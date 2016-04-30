from scipy.io          import loadmat, savemat
from scipy.interpolate import RectBivariateSpline, interp2d
from pylab             import array, linspace, ones, isnan, all, zeros, \
                              ndarray, e, nan, float64
from fenics            import interpolate, Expression, Function, \
                              vertices, FunctionSpace, RectangleMesh, \
                              MPI, mpi_comm_world, GenericVector, parameters, \
                              File, Constant
from pyproj            import Proj, transform
from colored           import fg, attr

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
  def __init__(self, mf_obj, flip=False, mesh=None, gen_space=True,
               zero_edge=False, bool_data=False, req_dg=False):
    """
    The following data are used to initialize the class :

      mf_obj    : mesh factory dictionary.
      flip      : flip the data over the x-axis?
      mesh      : FEniCS mesh if there is one already created.
      zero_edge : Make edges of domain -0.002?
      bool_data : Convert data to boolean?
      req_dg    : Some field may require DG space?

    Based on thickness extents, create a rectangular mesh object.
    Also define the function space as continious galerkin, order 1.
    """
    self.data       = {}        # dictionary of converted matlab data
    self.rem_nans   = False     # may change depending on 'identify_nans' call
    self.chg_proj   = False     # change to other projection flag
    self.color      = 'light_green'

    mf_obj = mf_obj.copy()

    self.name       = mf_obj.pop('dataset')
    self.cont       = mf_obj.pop('continent')
    self.proj       = mf_obj.pop('pyproj_Proj')

    # initialize extents :
    self.ny         = mf_obj.pop('ny')
    self.nx         = mf_obj.pop('nx')
    self.x_min      = float(mf_obj.pop('map_western_edge'))
    self.x_max      = float(mf_obj.pop('map_eastern_edge'))
    self.y_min      = float(mf_obj.pop('map_southern_edge'))
    self.y_max      = float(mf_obj.pop('map_northern_edge'))
    self.x          = linspace(self.x_min, self.x_max, self.nx)
    self.y          = linspace(self.y_min, self.y_max, self.ny)
    self.good_x     = array(ones(len(self.x)), dtype=bool)      # no NaNs
    self.good_y     = array(ones(len(self.y)), dtype=bool)      # no NaNs

    s    = "::: creating %s DataInput object :::" % self.name
    print_text(s, self.color)

    # process the data mf_obj :
    for fn in mf_obj:

      # raw data matrix with key fn :
      d = mf_obj[fn]

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

      self.mesh.init(1,2)
      self.dim        = self.mesh.ufl_cell().topological_dimension()
      if self.dim == 3:
        self.num_facets = self.mesh.size_global(2)
        self.num_cells  = self.mesh.size_global(3)
        self.dof        = self.mesh.size_global(0)
      elif self.dim == 2:
        self.num_facets = self.mesh.size_global(1)
        self.num_cells  = self.mesh.size_global(2)
        self.dof        = self.mesh.size_global(0)
      s = "    - using %iD mesh with %i cells, %i facets, %i vertices - " \
          % (self.dim, self.num_cells, self.num_facets, self.dof)
      print_text(s, self.color)
    else:
      s = "    - not using a mesh - "
      print_text(s, self.color)

  def change_projection(self, di):
    """
    change the projection of this data to that of the <di> DataInput object's
    projection.  The works only if the object was created with the parameter
    create_proj = True.
    """
    if type(di) == type(self):
      proj = di.proj
      name = di.name
    elif type(di) == dict:
      name = di['dataset']
      proj = di['pyproj_Proj']

    s    = "::: changing '%s' DataInput object projection to that of '%s' :::" \
           % (self.name, name)
    print_text(s, self.color)

    self.chg_proj = True
    self.new_p    = proj

  def get_xy(self,lon,lat):
    """
    Returns the (x,y) flat map coordinates corresponding to a given (lon,lat)
    coordinate pair using the DataInput object's current projection.
    """
    return self.proj(lon,lat)

  def interpolate_to_di(self, do, fn, fo):
    """
    interpolate the field with name <fn> from this dataInput object to 
    the grid used by the other dataInput object <do>.  The field is saved
    to <do>.data[<fo>].
    """
    s = "::: interpolating %s's '%s' field to %s's grid with key '%s' :::"
    print_text(s % (self.name, fn, do.name, fo) , self.color)
    interp      = interp2d(self.x, self.y, self.data[fn])
    fo_v        = interp(do.x, do.y)
    do.data[fo] = fo_v

  def transform_xy(self, di):
    """
    Transforms the coordinates from DataInput object <di> to this object's
    coordinates.  Returns tuple of arrays (x,y).
    """
    # FIXME : need a fast way to convert all the x, y. Currently broken
    s = "::: transforming coordinates from %s to %s :::" % (di.name, self.name)
    print_text(s, self.color)
    xn, yn = transform(di.proj, self.proj, di.x, di.y)
    return (xn, yn)

  def rescale_field(self, fo, fn, umin, umax, inverse=False):
    """
    Rescale the data field with key <fo> with lower and upper bound <umin>,
    <umax>, creating a new data field with key <fn>.

    If <inverse> == True, scale the data to the inverse of the data <fo>,
    i.e., the smallest values become <umax>, and the largest become <umin>.

    This is useful, for example, when refining a mesh in areas where a 
    velocity field is high.
    """
    if inverse:
      inv_txt = 'inversely'
    elif not inverse:
      inv_txt = ''
    s = "::: rescaling data field '%s' %s with lower and upper " + \
        "bound (%g, %g) to field '%s' :::" 
    print_text(s % (fo, inv_txt, umin, umax, fn), self.color)

    U = self.data[fo]
    if not inverse:
      amin = ( umin/(1.0 + U.max()) - umax/(1.0 + U.min()) ) / (umax - umin)
      amax = umin / ( amin + 1.0/(1.0 + U.min()) )
    elif inverse:
      amin = ( umin/(1.0 + U.min()) - umax/(1.0 + U.max()) ) / (umax - umin)
      amax = umin / ( amin + 1.0/(1.0 + U.max()) )
    
    self.data[fn] = (amin + 1.0/(1.0 + U)) * amax

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
    s    = "::: setting any value of %s's %s field below %.3e to %.3e :::" \
           % (self.name, fn, boundary, val)
    print_text(s, self.color)
    
    d                = self.data[fn]
    d[d <= boundary] = val
    self.data[fn]    = d

  def set_data_max(self, fn, boundary, val):
    """
    set the maximum value of a data array with filename <fn> above <boundary>
    to value <val>.
    """
    s    = "::: setting any value of %s's %s field above %.3e to %.3e :::" \
           % (self.name, fn, boundary, val)
    print_text(s, self.color)
    
    d                = self.data[fn]
    d[d >= boundary] = val
    self.data[fn]    = d

  def set_data_val(self, fn, old_val, new_val):
    """
    set all values of the matrix with filename <fn> equal to <old_val>
    to <new_val>.
    """
    s    = "::: setting all values of %s's %s field equal to %.3e to %.3e :::" \
           % (self.name, fn, old_val, new_val)
    print_text(s, self.color)
    
    d                = self.data[fn]
    d[d == old_val]  = new_val
    self.data[fn]    = d

  def get_interpolation(self, fn, near=False, bool_data=False, order=1):
    """
    Return a projection of data with field name <fn> on the functionspace.
    If multiple instances of the DataInput class are present, both initialized
    with identical meshes, the projections returned by this function may be
    used by the same mathematical problem.

    If <bool_data> is True, convert all values > 0 to 1.
    <order> sets the order of the interpolation, default linear (1).
    """
    if near:
      t = 'nearest-neighbor'
    else:
      t = 'spline'
    s    = "::: getting %s %s interpolation from %s :::" % (fn, t, self.name)
    print_text(s, self.color)

    interp = self.get_expression(fn, kx=order, ky=order,
                                 bool_data=bool_data, near=near)

    return interpolate(interp, self.func_space, annotate=False)

  def get_expression(self, fn, order=1, bool_data=False, near=False):
    """
    Creates a spline-interpolation expression for data <fn>.  Optional
    argument <order> determine order of approximation in x and y
    directions (default linear).  If <bool_data> is True, convert to boolean,
    if <near> is True, use nearest-neighbor interpolation.
    """
    if near:
      t = 'nearest-neighbor'
    else:
      t = '%i-order spline' % order
    s = "::: getting %s %s expression from %s :::" % (fn, t, self.name)
    print_text(s, self.color)

    data = self.data[fn]
    if bool_data: data[data > 0] = 1

    if self.chg_proj:
      new_proj = self.new_p
      old_proj = self.proj

    if not near :
      spline = RectBivariateSpline(self.x, self.y, data.T, kx=order, ky=order)

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

  def write_matlab(self, di, f, filename, val=e):
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
      filename : file name to save.
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
                      'continent'         : di.cont,
                      'nx'                : di.nx,
                      'ny'                : di.ny,
                      'map_eastern_edge'  : di.x_max,
                      'map_western_edge'  : di.x_min,
                      'map_northern_edge' : di.y_max,
                      'map_southern_edge' : di.y_min,
                      'map_name'          : outfile,
                      'projection'        : di.proj.srs})


def print_min_max(u, title, color='97', cls=None):
  """
  Print the minimum and maximum values of <u>, a Vector, Function, or array.
  """
  #if cls is not None:
  #  color = cls.color()
  if isinstance(u, GenericVector):
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, ndarray):
    if u.dtype != float64:
      u = u.astype(float64)
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, Function):# \
    #   or isinstance(u, dolfin.functions.function.Function):
    uMin = MPI.min(mpi_comm_world(), u.vector().min())
    uMax = MPI.max(mpi_comm_world(), u.vector().max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, int) or isinstance(u, float):
    s    = title + ' : %.3e' % u
    print_text(s, color)
  elif isinstance(u, Constant):
    s    = title + ' : %.3e' % u(0)
    print_text(s, color)
  else:
    er = title + ": print_min_max function requires a Vector, Function" \
         + ", array, int or float, not %s." % type(u)
    print_text(er, 'red', 1)


def get_text(text, color='white', atrb=0, cls=None):
  """
  Returns text <text> from calling class <cl> for later printing.
  """
  if cls is not None:
    color = cls.color()
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    return text


def print_text(text, color='white', atrb=0, cls=None):
  """
  Print text <text> from calling class <cl> to the screen.
  """
  if cls is not None:
    color = cls.color()
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    print text



