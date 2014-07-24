"""
Utilities file:

  This contains classes that are used by UM-FEISM to aid in the loading
  of data and preparing data for use in DOLFIN based simulation. 
  Specifically the data are projected onto the mesh with appropriate 
  basis functions.

"""
import subprocess

from scipy.io          import loadmat, savemat
from scipy.interpolate import RectBivariateSpline
from pylab             import array, shape, linspace, ones, isnan, all, zeros, \
                              meshgrid, figure, show, size, hstack, vstack, \
                              argmin
from fenics            import interpolate, project, Expression, Function, \
                              vertices, Mesh, MeshEditor, FunctionSpace, \
                              RectangleMesh, MPI, mpi_comm_world
from data.data_factory import DataFactory
from pyproj            import Proj, transform
from termcolor         import colored, cprint

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
    
    first = True  # initialize domain by first file's extents

    if direc == None and type(files) == dict:
      self.name = files.pop('dataset')
    elif direc != None:
      self.name = direc
       
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: creating %s DataInput object :::" % self.name
      text = colored(s, 'green')
      print text
    
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

  def integrate_field(self, fn_spec, specific, fn_main, r=20, val=0.0):
    """
    Assimilate a field with filename <fn_spec>  from DataInput object 
    <specific> into this DataInput's field with filename <fn_main>.  The
    parameter <val> should be set to the specific dataset's value for 
    undefined regions, default is 0.0.  <r> is a parameter used to eliminate
    border artifacts from interpolation; increase this value to eliminate edge
    noise.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: integrating %s field from %s :::" % (fn_spec, specific.name)
      text = colored(s, 'green')
      print text
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
      print "Warning: %d row(s) of \"%s\" are entirely NaN." % (total_nan_x, fn)

    if any(good_y != self.good_y):
      total_nan_y = sum(good_y == False)
      self.rem_nans = True
      print "Warning: %d col(s) of \"%s\" are entirely NaN." % (total_nan_y, fn)
    
    self.good_x = good_x
    self.good_y = good_y
  
  def remove_nans(self):
    """
    remove extra rows/cols from data where NaNs were identified and set the 
    extents to those of the good x and y values.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: removing NaNs from %s :::" % self.name
      text = colored(s, 'green')
      print text
    
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

  def get_interpolation(self, fn, near=False, kx=3, ky=3):
    """
    return the interpolation of data field <fn>.
    """
    if near:
      interp = self.get_nearest_expression(fn, bool_data=bool_data)
    else:
      interp = self.get_spline_expression(fn,kx=kx,ky=ky)
    
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: getting %s interpolation :::" % fn
      text = colored(s, 'green')
      print text

    proj   = interpolate(interp, self.func_space)
    return proj

  def get_projection(self, fn, dg=False, near=False, 
                     bool_data=False, kx=3, ky=3):
    """
    Return a projection of data with filname <fn> on the functionspace.
    If multiple instances of the DataInput class are present, both initialized 
    with identical meshes, the projections returned by this function may be
    used by the same mathematical problem.

    If <dg> is True, use a discontinuous space, otherwise, continuous.

    If <bool_data> is True, convert all values > 0 to 1.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: getting %s projection :::" % fn
      text = colored(s, 'green')
      print text

    if dg:
      interp = self.get_nearest_expression(fn, bool_data=bool_data)
      proj   = project(interp, self.func_space_dg)
    
    else:
      if near:
        interp = self.get_nearest_expression(fn, bool_data=bool_data)
        proj   = project(interp, self.func_space)
      else:
        interp = self.get_spline_expression(fn,kx=kx,ky=ky,bool_data=bool_data)
        proj   = project(interp, self.func_space)
        
    return proj

  def get_nearest_expression(self, fn, bool_data=False):
    """
    Returns a dolfin expression using a nearest-neighbor interpolant of data 
    <fn>.  If <bool_data> is True, convert to boolean.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: getting %s nearest expression from %s :::" % (fn, self.name)
      text = colored(s, 'green')
      print text
    
    data = self.data[fn]
    if bool_data: data[data > 0] = 1
    
    if self.chg_proj:
      new_proj = self.new_p
      old_proj = self.p

    xs       = self.x
    ys       = self.y
    chg_proj = self.chg_proj

    class newExpression(Expression):
      def eval(self, values, x):
        if chg_proj:
          xn, yn = transform(new_proj, old_proj, x[0], x[1])
        else:
          xn, yn = x[0], x[1]
        idx       = abs(xs - xn).argmin()
        idy       = abs(ys - yn).argmin()
        values[0] = data[idy, idx]

    return newExpression(element = self.func_space.ufl_element())

  def get_spline_expression(self, fn, kx=3, ky=3, bool_data=False):
    """
    Creates a spline-interpolation expression for data <fn>.  Optional 
    arguments <kx> and <ky> determine order of approximation in x and y
    directions (default cubic).  If <bool_data> is True, convert to boolean.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: getting %s spline expression from %s :::" % (fn, self.name)
      text = colored(s, 'green')
      print text

    data = self.data[fn]
    if bool_data: data[data > 0] = 1
    
    if self.chg_proj:
      new_proj = self.new_p
      old_proj = self.p
  
    spline = RectBivariateSpline(self.x, self.y, data.T, kx=kx, ky=ky)
    chg_proj = self.chg_proj
    
    class newExpression(Expression):
      def eval(self, values, x):
        if chg_proj:
          xn, yn = transform(new_proj, old_proj, x[0], x[1])
        else:
          xn, yn = x[0], x[1]
        values[0] = spline(xn, yn)
  
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
    self.directory = directory
      
  def write_dict_of_files(self, d, extension='.pvd'):
    """ 
    Looking for a dictionary d of data to save. The keys are the file 
    names, and the values are the data fields to be stored. Also takes an
    optional extension to determine if it is pvd or xml output.
    """
    for filename in d:
      file_handle = File(self.directory + filename + extension)
      file_handle << d[filename]
  
  def write_one_file(self, name, data, extension='.pvd'):
    """
    Save a single file of FEniCS Function <data> named <name> to the DataOutput 
    instance's directory.  Extension may be '.xml' or '.pvd'.
    """
    file_handle = File(self.directory + name + extension)
    file_handle << data

  def write_matlab(self, di, f, outfile, val=-2e9):
    """ 
    Using the projections that are read in as data files, create Matlab
    version 4 files to output the regular gridded data in a field.

    INPUTS:
      di      : a DataInput object, defined in the class above in this file.
      f       : a FEniCS function, to be mapped onto the regular grid that is in
                di, established from the regular gridded data to start the
                simulation.
      outfile : a file name for the matlab file output (include the
                extension) values not in mesh are set to <val>, default -2e9. 
    
    OUTPUT: 
      A single file will be written with name, outfile.
    """
    fa = zeros( (di.y.size, di.x.size) )
    parameters['allow_extrapolation'] = True
    for j,x in enumerate(di.x):
      for i,y in enumerate(di.y):
        try:
          fa[i,j] = f(x,y)
        except: 
          fa[i,j] = val
    
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


class MeshGenerator(object):
  """
  generate a mesh.
  """ 
  def __init__(self, dd, fn, direc):
    """
    Generate a mesh with DataInput object <dd>, output filename <fn>, and 
    output directory <direc>.
    """
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
    
  def set_contour(self,cont_array):
    """ This is an alternative to the create_contour method that allows you to 
    manually specify contour points.
    Inputs:
    cont_array : A numpy array of contour points (i.e. array([[1,2],[3,4],...])) 
    """
    self.longest_cont = cont_array
    
  def plot_contour(self):
    """
    Plot the contour created with the "create_contour" method.
    """
    ax = self.ax
    lc  = self.longest_cont
    ax.plot(lc[:,0], lc[:,1], 'r-', lw = 3.0)
    show()

  def eliminate_intersections(self, dist=10):
    """
    Eliminate intersecting boundary elements. <dist> is an integer specifiying 
    how far forward to look to eliminate intersections.
    """
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: eliminating intersections :::"
      text = colored(s, 'green')
      print text

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
          flag[ii+1] = 0
          flag[jj] = 0
    
    counter  = 0
    new_cont = zeros((sum(flag),2))
    for ii,fl in enumerate(flag):
      if fl:
        new_cont[counter,:] = lc[ii,:]
        counter += 1
    
    self.longest_cont = new_cont
  
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
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: writing gmsh contour :::"
      text = colored(s, 'green')
      print text
    c   = self.longest_cont
    f   = self.f
    x   = self.x
    y   = self.y

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
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: extruding gmsh contour :::"
      text = colored(s, 'green')
      print text
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
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: closing geo file :::"
      text = colored(s, 'green')
      print text
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
    if MPI.rank(mpi_comm_world())==0:
      s    = "::: initializing MeshRefiner on '%s.geo' :::" % gmsh_file_name
      text = colored(s, 'green')
      print text
    
    from gmshpy import GModel, GmshSetOption

    self.field  = di.data[fn].T
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
      if MPI.rank(mpi_comm_world())==0:
        s    = "::: opening GUI :::"
        text = colored(s, 'green')
        print text
      FlGui.instance().run()

    # instead of starting the GUI, we could generate the mesh and save it
    else:
      if MPI.rank(mpi_comm_world())==0:
        s    = "::: writing %s.msh :::" % out_file_name
        text = colored(s, 'green')
        print text
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

