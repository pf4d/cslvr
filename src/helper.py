r"""
This module handles the refinements to the tetrehedral mesh as well as providing
functions to set nonlinear solver defaults, extract boundaries, generate
expressions from data, and write mesh files.

The mesh refinements are determined by calculating the anisotropic error metric

:Equation:
   .. math::
      e\left(c\right)\propto max_{i\in T} \ \textbf{x}^{T}_{i}\textbf{Mx}_{i}

+-----------------------------------------+---------------------------------+
|Term                                     |Description                      |
+=========================================+=================================+
|.. math::                                |Cellwise error estimate          |
|   e\left(c\right)                       |                                 |
+-----------------------------------------+---------------------------------+
|.. math::                                |A given mesh cell                |
|   T                                     |                                 |
+-----------------------------------------+---------------------------------+
|.. math::                                |An edge in T                     |
|   \textbf{x}_i                          |                                 |
+-----------------------------------------+---------------------------------+
|.. math::                                |Metric tensor                    |
|   \textbf{M} = \textbf{V}^T|\Lambda|    |                                 |
|   \textbf{V}                            |                                 |
+-----------------------------------------+---------------------------------+
|.. math::                                |Eigenvalues of the Hessian       |
|   \textbf{V}                            |matrix we are attempting to      |
|                                         |equidistribute over              |
|.. math::                                |                                 |
|   \Lambda                               |                                 |
+-----------------------------------------+---------------------------------+

The Hessian of the velocity norm is used in calculating error metrics of the 
meshes used in the simulations.
"""

import inspect
import pylab              as pl
import numpy              as np
from pylab                import plt
from fenics               import *
from termcolor            import colored, cprint
from mpl_toolkits.basemap import Basemap
from matplotlib           import colors
from pyproj               import *
from io                   import DataInput

pl.mpl.rcParams['font.family']     = 'serif'
pl.mpl.rcParams['legend.fontsize'] = 'medium'


def raiseNotDefined():
  fileName = inspect.stack()[1][1]
  line     = inspect.stack()[1][2]
  method   = inspect.stack()[1][3]
  
  text = "*** Method not implemented: %s at line %s of %s"
  print text % (method, line, fileName)
  sys.exit(1)

def download_file(url, direc, folder, extract=False):
  """
  download a file with url <url> into directory <direc>/<folder>.  If <extract>
  is True, extract the .zip file into the directory and delete the .zip file.
  """
  import urllib2
  import sys
  import os
  import zipfile
  import tarfile
  
  # make the directory if needed :
  direc = direc + '/' + folder + '/'
  d     = os.path.dirname(direc)
  if not os.path.exists(d):
    os.makedirs(d)

  # url file info :
  fn   = url.split('/')[-1]
  u    = urllib2.urlopen(url)
  f    = open(direc + fn, 'wb')
  meta = u.info()
  fs   = int(meta.getheaders("Content-Length")[0])
  
  s    = "Downloading: %s Bytes: %s" % (fn, fs)
  text = colored(s, 'green')
  print text % (counter, max_iter, inner_error, inner_tol)
  
  fs_dl  = 0
  blk_sz = 8192
  
  # download the file and print status :
  while True:
    buffer = u.read(blk_sz)
    if not buffer:
      break
  
    fs_dl += len(buffer)
    f.write(buffer)
    status = r"%10d  [%3.2f%%]" % (fs_dl, fs_dl * 100. / fs)
    status = status + chr(8)*(len(status)+1)
    sys.stdout.write(status)
    sys.stdout.flush()
  
  f.close()
  
  # extract the zip/tar.gz file if necessary :
  if extract:
    ty = fn.split('.')[-1]
    if ty == 'zip':
      cf = zipfile.ZipFile(direc + fn)
    else:
      cf = tarfile.open(direc + fn, 'r:gz')
    cf.extractall(direc)
    os.remove(direc + fn)


class IsotropicMeshRefiner(object):
  """
  In this class, the cells in the mesh are isotropically refined above a 
  a certain proportion of average error.
  
  :param m_0  : Initial unrefined mesh
  :param U_ex : Representation of the ice sheet as a dolfin expression
  """
  def __init__(self, m_0, U_ex):
    self.mesh = m_0
    self.U_ex = U_ex

  def refine(self, REFINE_RATIO, hmin, hmax):
    """
    Determines which cells that are above a certain error threshold based
    on the diameter of the cells
    
    :param REFINE_RATIO : Long value between 0 and 1, inclusive, to select 
                          an error value in a reverse sorted list.  
                          (i.e. .5 selects the midpoint)
    :param hmin         : Minimum diameter of the cells
    :param hmax         : Maximum diameter of the cells
    """
    print "refining %.2f%% of max error" % (REFINE_RATIO * 100)
    mesh = self.mesh

    V    = FunctionSpace(mesh, "CG", 1)
    Hxx  = TrialFunction(V)
    Hxy  = TrialFunction(V)
    Hyy  = TrialFunction(V)
    phi  = TestFunction(V)
    
    U    = project(self.U_ex, V)
    a_xx = Hxx * phi * dx
    L_xx = - U.dx(0) * phi.dx(0) * dx

    a_xy = Hxy * phi * dx
    L_xy = - U.dx(0) * phi.dx(1) * dx

    a_yy = Hyy * phi * dx
    L_yy = - U.dx(1) * phi.dx(1) * dx

    Hxx  = Function(V)
    Hxy  = Function(V)
    Hyy  = Function(V)
  
    solve(a_xx == L_xx, Hxx)
    solve(a_xy == L_xy, Hxy)
    solve(a_yy == L_yy, Hyy)
    e_list = []
    e_list_calc = []
    
    #Iterate overthecells in the mesh
    for c in cells(mesh):
    
      x = c.midpoint().x()
      y = c.midpoint().y()

      a = Hxx(x,y)
      b = Hxy(x,y)
      d = Hyy(x,y)

      H_local = ([[a,b], [b,d]])
      
      #Calculate the Hessian of the velocity norm
      Hnorm = pl.norm(H_local, 2)
      h     = c.diameter()
      
      # Use the diameter of the cells to set the error :
      if h < hmin:
        error = 0
      elif h > hmax:
        error = 1e20
      else:
        error = h**2 * Hnorm
        e_list_calc.append(error)
      e_list.append(error)
   
    idx   = int(len(e_list_calc) * REFINE_RATIO)
    e_mid = sorted(e_list, reverse=True)[idx]
    print "error midpoint :", e_mid

    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
      ci = c.index()
      cd = c.diameter()
      cell_markers[c] = e_list[ci] > e_mid and cd > hmin and cd < hmax
  
    self.mesh = refine(mesh, cell_markers)
    self.U    = U

  def extrude(self, n_layers, workspace_path="", n_processors = 1):
    """
    This function writes the refinements to the mesh to msh files and
    extrudes the mesh
    
    :param n_layers:  Number of layers in the mesh
    :param workspace_path: Path to the location where the refined meshes
       will be written
    :param int n_processors: Number of processers utilized in the extrusion
    """
    import os
    layers = str(n_layers + 1)

    write_gmsh(self.mesh, workspace_path + "/2dmesh.msh")

    output = open(workspace_path + "/2dmesh.geo", 'w')

    output.write("Merge \"2dmesh.msh\";\n")
    output.write("Extrude {0,0,1} {Surface{0}; Layers{" + layers + "};}")
    output.close()
    print "Extruding mesh (This could take a while)."

    string = "mpirun -np {0:d} gmsh " + workspace_path + "/2dmesh.geo -3 -o " \
             + workspace_path + "/3dmesh.msh -format msh -v 0"
    os.system(string.format(n_processors))

    string = "dolfin-convert " + workspace_path + "/3dmesh.msh " \
             + workspace_path + "/3dmesh.xml"
    os.system(string)


class AnisotropicMeshRefiner(object):
  """
  This class performs anistropic mesh refinements on the mesh in order to 
  account for the directional nature of the velocity field.  In order to 
  accomplish this, Gauss-Steidl iterations are used to approximately solve
  the elasticity problem with computed edge errors as 'spring constants'.
  
  :param m_0  : Initial unrefined mesh
  :param U_ex : Representation of the ice sheet as a Dolfin expression
  """
  
  def __init__(self, m_0, U_ex):
    self.mesh = m_0
    self.U_ex = U_ex

  def weighted_smoothing(self, edge_errors, omega=0.1):
    """
    Smooths the points contained within the mesh
    
    :param edge_errors : Dolfin edge function containing the calculated
                         edge errors of the mesh
    :param omega       : Weighting factor used to refine the mesh
    """
    mesh  = self.mesh
    coord = mesh.coordinates()

    adjacent_points = {}
    mesh.init(1,2)
    
    #Create copies of the x coordinates
    new_x          = pl.copy(coord[:,0])
    new_y          = pl.copy(coord[:,1])
    exterior_point = {}

    for v in vertices(mesh):
      adjacent_points[v.index()] = set()

    for e in facets(mesh):
      vert = e.entities(0)
      ext  = e.exterior()
      for ii in (0,1):
        adjacent_points[vert[ii]].add((vert[ii-1], edge_errors[e]))
        adjacent_points[vert[ii-1]].add((vert[ii], edge_errors[e]))
      exterior_point[vert[0]] = ext
      exterior_point[vert[1]] = ext

    for item in adjacent_points.iteritems():
      index, data = item
      x       = coord[index,0]
      y       = coord[index,1]
      x_sum   = 0.0
      y_sum   = 0.0
      wgt_sum = 0.0
      kbar    = 0.0

      for entry in list(data):
        x_p   = coord[entry[0],0]
        y_p   = coord[entry[0],1]
        error = entry[1]
        kbar += 1./len(list(data)) * error/pl.sqrt( (x-x_p)**2 + (y-y_p)**2 ) 
      kbar = 0.0 

      for entry in list(data):
        x_p      = coord[entry[0],0]
        y_p      = coord[entry[0],1]
        error    = entry[1]
        k_ij     = error/pl.sqrt( (x-x_p)**2 + (y-y_p)**2 )
        x_sum   += (k_ij-kbar) * (x_p-x)
        y_sum   += (k_ij-kbar) * (y_p-y)
        wgt_sum += k_ij
        
      if not exterior_point[index]:
        new_x[index] = x + omega * x_sum / wgt_sum
        new_y[index] = y + omega * y_sum / wgt_sum

    return new_x, new_y 

  def get_edge_errors(self):
    """
    Calculates the error estimates of the expression projected into the mesh.
    
    :rtype: Dolfin edge function containing the edge errors of the mesh
    """
    mesh  = self.mesh
    coord = mesh.coordinates()
    
    V     = FunctionSpace(mesh, "CG", 1)
    Hxx   = TrialFunction(V)
    Hxy   = TrialFunction(V)
    Hyy   = TrialFunction(V)
    phi   = TestFunction(V)
    
    edge_errors = EdgeFunction('double', mesh)

    U    = project(self.U_ex, V)
    a_xx = Hxx * phi * dx
    L_xx = - U.dx(0) * phi.dx(0) * dx

    a_xy = Hxy * phi * dx
    L_xy = - U.dx(0) * phi.dx(1) * dx

    a_yy = Hyy * phi * dx
    L_yy = - U.dx(1) * phi.dx(1) * dx       

    Hxx  = Function(V)
    Hxy  = Function(V)
    Hyy  = Function(V)
         
    Mxx  = Function(V)
    Mxy  = Function(V)
    Myy  = Function(V)
  
    solve(a_xx == L_xx, Hxx)
    solve(a_xy == L_xy, Hxy)
    solve(a_yy == L_yy, Hyy)
    e_list = []

    for v in vertices(mesh):
      idx = v.index()
      pt  = v.point()
      x   = pt.x()
      y   = pt.y()
          
      a   = Hxx(x, y)
      b   = Hxy(x, y)
      d   = Hyy(x, y)

      H_local = ([[a,b], [b,d]])

      l, ve   = pl.eig(H_local)
      M       = pl.dot(pl.dot(ve, abs(pl.diag(l))), ve.T)       

      Mxx.vector()[idx] = M[0,0]
      Mxy.vector()[idx] = M[1,0]
      Myy.vector()[idx] = M[1,1]

    e_list = []
    for e in edges(mesh):
      I, J  = e.entities(0)
      x_I   = coord[I,:]
      x_J   = coord[J,:]
      M_I   = pl.array([[Mxx.vector()[I], Mxy.vector()[I]],
                       [Mxy.vector()[I], Myy.vector()[I]]]) 
      M_J   = pl.array([[Mxx.vector()[J], Mxy.vector()[J]],
                       [Mxy.vector()[J], Myy.vector()[J]]])
      M     = (M_I + M_J)/2.
      dX    = x_I - x_J
      error = pl.dot(pl.dot(dX, M), dX.T)
      
      e_list.append(error)
      edge_errors[e] = error
    
    return edge_errors

  def refine(self, edge_errors, gamma=1.4):
    """
    This function iterates through the cells in the mesh, then refines
    the mesh based on the relative error and the cell's location in the
    mesh.
    
    :param edge_errors : Dolfin edge function containing edge errors of 
                         of the current mesh.
    :param gamma       : Scaling factor for determining which edges need be 
                         refined.  This is determined by the average error 
                         of the edge_errors variable
    """
    mesh = self.mesh
    
    mesh.init(1,2)
    mesh.init(0,2)
    mesh.init(0,1)
    
    avg_error                 = edge_errors.array().mean()
    error_sorted_edge_indices = pl.argsort(edge_errors.array())[::-1]
    refine_edge               = FacetFunction('bool', mesh)
    for e in edges(mesh):
      refine_edge[e] = edge_errors[e] > gamma*avg_error

    coordinates = pl.copy(self.mesh.coordinates())      
    current_new_vertex = len(coordinates)
    cells_to_delete = []
    new_cells = []

    for iteration in range(refine_edge.array().sum()):
      for e in facets(self.mesh):
        if refine_edge[e] and (e.index()==error_sorted_edge_indices[0]):
          adjacent_cells = e.entities(2)
          adjacent_vertices = e.entities(0)
          if not any([c in cells_to_delete for c in adjacent_cells]):
            new_x,new_y = e.midpoint().x(),e.midpoint().y()
            coordinates = pl.vstack((coordinates,[new_x,new_y]))
            for c in adjacent_cells:
              off_facet_vertex = list(self.mesh.cells()[c])
              [off_facet_vertex.remove(ii) for ii in adjacent_vertices]
              for on_facet_vertex in adjacent_vertices:
                new_cell = pl.sort([current_new_vertex,off_facet_vertex[0],on_facet_vertex])
                new_cells.append(new_cell)
              cells_to_delete.append(c)
            current_new_vertex+=1
      error_sorted_edge_indices = error_sorted_edge_indices[1:]

    old_cells = self.mesh.cells()
    keep_cell = pl.ones(len(old_cells))
    keep_cell[cells_to_delete] = 0
    old_cells_parsed = old_cells[keep_cell.astype('bool')]
    all_cells = pl.vstack((old_cells_parsed,new_cells))
    n_cells = len(all_cells)

    e = MeshEditor()
    refined_mesh = Mesh()
    e.open(refined_mesh,self.mesh.geometry().dim(),self.mesh.topology().dim())
    e.init_vertices(current_new_vertex)
    for index,x in enumerate(coordinates):
      e.add_vertex(index,x[0],x[1])
  
    e.init_cells(n_cells)
    for index,c in enumerate(all_cells):
      e.add_cell(index,c.astype('uintc'))

    e.close()
    refined_mesh.order()
    self.mesh = refined_mesh 
 
  def extrude(self, n_layers, workspace_path="", n_processors = 1):
    """
    This function writes the refinements to the mesh to msh files and
    extrudes the mesh
    
    :param n_layers:  Number of layers in the mesh
    :param workspace_path: Path to the location where the refined meshes
       will be written
    :param int n_processors: Number of processers utilized in the extrusion
    """
    import os
    layers = str(n_layers + 1)

    write_gmsh(self.mesh, workspace_path + "/2dmesh.msh")

    output = open(workspace_path + "/2dmesh.geo", 'w')

    output.write("Merge \"2dmesh.msh\";\n")
    output.write("Extrude {0,0,1} {Surface{0}; Layers{" + layers + "};}")
    output.close()
    print "Extruding mesh (This could take a while)."

    string = "mpirun -np {0:d} gmsh " + workspace_path + "/2dmesh.geo -3 -o " \
             + workspace_path + "/3dmesh.msh -format msh -v 0"
    os.system(string.format(n_processors))

    string = "dolfin-convert " + workspace_path + "/3dmesh.msh " \
             + workspace_path + "/3dmesh.xml"
    os.system(string)


def write_gmsh(mesh,path):
  """
  This function iterates through the mesh and writes a file to the specified
  path
  
  :param mesh: Mesh that is to be written to a file
  :param path: Path to write the mesh file to
  """
  output = open(path,'w')
  
  cell_type = mesh.type().cell_type()

  nodes = mesh.coordinates()
  n_nodes = mesh.num_vertices()

  nodes = pl.hstack((nodes,pl.zeros((n_nodes,3 - pl.shape(mesh.coordinates())[1]))))

  cells = mesh.cells()
  n_cells = mesh.num_cells()

  output.write("$MeshFormat\n" + 
        "2.2 0 8\n" +
         "$EndMeshFormat\n" +
         "$Nodes \n" +
         "{0:d}\n".format(n_nodes))

  for ii,node in enumerate(nodes):
    output.write("{0:d} {1:g} {2:g} {3:g}\n".format(ii+1,node[0],node[1],node[2]))

  output.write("$EndNodes\n")

  output.write("$Elements\n" + 
          "{0:d}\n".format(n_cells))

  for ii,cell in enumerate(cells):
    #if cell_type == 1:
    #  output.write("{0:d} 1 0 {1:d} {2:d}\n".format(ii+1,int(cell[0]+1),int(cell[1]+1)))
    if cell_type == 2:
      output.write("{0:d} 2 0 {1:d} {2:d} {3:d}\n".format(ii+1,int(cell[0]+1),int(cell[1]+1),int(cell[2]+1)))
    #elif cell_type == 3:
    #  output.write("{0:d} 4 0 {1:d} {2:d} {3:d} {4:d}\n".format(ii+1,int(cell[0]+1),int(cell[1]+1),int(cell[2]+1),int(cell[3]+1)))
    else:
      print "Unknown cell type"
  
  output.write("$EndElements\n")
  output.close()


def default_nonlin_solver_params():
  """ 
  Returns a set of default solver parameters that yield good performance
  """
   
  stokes_params = NonlinearVariationalSolver.default_parameters()
  stokes_params['newton_solver']['maximum_iterations']      = 25
  stokes_params['newton_solver']['error_on_nonconvergence'] = True
  stokes_params['newton_solver']['relaxation_parameter']    = 0.7
  stokes_params['newton_solver']['method']                  = 'lu'
  stokes_params['newton_solver']['relative_tolerance']      = 1e-3
  stokes_params['newton_solver']['report']                  = True
  return stokes_params


def default_ffc_options():
  """ 
  Returns a set of default ffc options that yield good performance
  """
  ffc_options = {"optimize"               : True,
                 "eliminate_zeros"        : True,
                 "precompute_basis_const" : True,
                 "precompute_ip_const"    : True}
   
  return ffc_options


def default_config():
  """
  Returns a set of default configuration parameters to help users get started.
  """
  config = { 'mode'                         : 'steady',
             'model_order'                  : 'BP',
             'use_dukowicz'                 : True,
             't_start'                      : None,
             't_end'                        : None,
             'time_step'                    : 1.0,
             'output_path'                  : None,
             'wall_markers'                 : [],
             'periodic_boundary_conditions' : False,
             'use_pressure_boundary'        : True,
             'log'                          : True,
             'log_history'                  : False,
             'coupled' : 
             { 
               'on'                  : False,
               'inner_tol'           : 1e-10,
               'max_iter'            : 0,
             },
             'velocity' : 
             { 
               'on'                  : True,
               'log'                 : True,
               'full_BP'             : False,
               'poly_degree'         : 2,
               'newton_params'       : default_nonlin_solver_params(),
               'ffc_options'         : default_ffc_options(),
               'vert_solve_method'   : 'mumps',
               'solve_vert_velocity' : True,
               'eta_gnd'             : None,
               'eta_shf'             : None,
               'use_U0'              : False,
               'A'                   : 1e-16,
               'r'                   : 0.0,
               'E'                   : 1.0,
               'use_lat_bcs'         : False,
               'calc_pressure'       : False,
               'use_stat_beta'       : False,
             },
             'enthalpy' : 
             { 
               'on'                  : False,
               'log'                 : True,
               'N_T'                 : 8,
               'solve_method'        : 'mumps',
               'use_surface_climate' : False,
               'lateral_boundaries'  : None,
               'ffc_options'         : default_ffc_options(),
             },
             'free_surface' :
             { 
               'on'                  : False,
               'lump_mass_matrix'    : False,
               'use_pdd'             : False,
               'observed_smb'        : None,
               'use_shock_capturing' : False,
               'thklim'              : 10.0,
               'static_boundary_conditions' : False,
               'ffc_options'         : default_ffc_options(),
             },  
             'age' : 
             { 
               'on'                  : False,
               'log'                 : True,
               'use_smb_for_ela'     : True,
               'ela'                 : None,
             },
             'surface_climate' : 
             { 
               'on'                  : False,
               'T_ma'                : None,
               'T_ju'                : None,
               'beta_w'              : None,
               'sigma'               : None,
               'precip'              : None,
             },
             'adjoint' :
             { 
               'alpha'               : 0.0,
               'gamma1'              : 1.0,
               'gamma2'              : 100.0,
               'max_fun'             : 20,
               'objective_function'  : 'log',
               'surface_integral'    : 'grounded',
               'bounds'              : None,
               'control_variable'    : None,
               'regularization_type' : 'Tikhonov',
             },
             'balance_velocity' :
             {
               'on'                  : False,
               'log'                 : True,
               'kappa'               : 10.0,
             },
             'stokes_balance' :
             {
               'on'                  : False,
               'log'                 : True,
               'vert_integrate'      : False,
               'viscosity_mode'      : 'isothermal',
               'A'                   : 1e-16,
               'eta'                 : None,
               'T'                   : None,
               'W'                   : None,
               'E'                   : None,
             }}
  return config


def extract_boundary_mesh(mesh,surface_facet,marker,variable_list = []):
  """
  This function iterates through the cells and vertces of the mesh in order
  to find the boundaries

  :param mesh: The dolfin mesh for which to find the boundaries
  :param int marker: Cell marker to determine the surface facets
  :param variable_list: A list of variables corrisponding to the mesh
  :rtype: Dolfin boundary mesh containing information on the surface of the
     mesh and a list of surface variables derived from the variable_list 
     parameter
  """
  from dolfin import vertices

  D = mesh.topology().dim()
  surface_mesh = BoundaryMesh(mesh,'exterior')
  surface_mesh.clear()
 
  editor = MeshEditor()
  editor.open(surface_mesh,mesh.type().type2string(mesh.type().facet_type()), D-1,D-1)

  mesh.init(D-1,D)

  #exterior = mesh.parallel_data().exterior_facet()

  num_vertices = mesh.num_vertices()

  boundary_vertices = (pl.ones(num_vertices)*num_vertices).astype(int)
  num_boundary_vertices = 0
  num_boundary_cells = 0

  #surface_facet = mesh.domains().facet_domains(mesh)
  boundary_facet = MeshFunctionBool(mesh,D-1,False)
  for f in facets(mesh):
    if surface_facet[f] == marker and f.exterior():
      boundary_facet[f] = True
#    if boundary_facet[f]:
      for v in vertices(f):
        v_index = v.index()
        if boundary_vertices[v_index] == num_vertices:
          boundary_vertices[v_index] = num_boundary_vertices
          num_boundary_vertices += 1
      num_boundary_cells += 1 

  editor.init_vertices(num_boundary_vertices)
  editor.init_cells(num_boundary_cells)

  vertex_map = surface_mesh.entity_map(0)
  if num_boundary_vertices > 0:
    vertex_map.init(surface_mesh, 0, num_boundary_vertices)

  cell_map = surface_mesh.entity_map(D-1)
  if num_boundary_cells > 0:
    cell_map.init(surface_mesh, D-1, num_boundary_cells)

  for v in vertices(mesh):
    vertex_index = boundary_vertices[v.index()]
    if vertex_index != mesh.num_vertices():
      if vertex_map.size() > 0:
        vertex_map[vertex_index] = v.index()
      editor.add_vertex(vertex_index,v.point())

  cell = pl.zeros(surface_mesh.type().num_vertices(surface_mesh.topology().dim()),dtype = pl.uintp)
  current_cell = 0
  for f in facets(mesh):
    if boundary_facet[f]:
      vertices = f.entities(0)
      for ii in range(cell.size):
        cell[ii] = boundary_vertices[vertices[ii]]
      if cell_map.size()>0:
        cell_map[current_cell] = f.index()
      editor.add_cell(current_cell,cell)
      current_cell += 1

  surface_mesh.order()
  Q = FunctionSpace(surface_mesh,"CG",1)
  surface_variable_list = []

  for ii in range(len(variable_list)):
    surface_variable_list.append(Function(Q))

  v2d_surf = vertex_to_dof_map(Q)
  print vertex_map.array()
  v2d_3d = vertex_to_dof_map(variable_list[0].function_space())

  for ii,index in enumerate(vertex_map.array()):
    for jj,variable in enumerate(variable_list):
      surface_variable_list[jj].vector()[v2d_surf[ii]] = variable_list[jj].vector()[v2d_3d[index]]

  return surface_mesh,surface_variable_list

    
def generate_expression_from_gridded_data(x,y,var,kx=1,ky=1):
  """
  This function creates a dolfin 2D expression from data input
  
  :param x: List of x coordinates
  :param y: List of y coordinates
  :param var: List of values associated with each x and y coordinate pair
  :rtype: A dolfin Expression object representing the data
  """
  from scipy.interpolate import RectBivariateSpline
  interpolant = RectBivariateSpline(x,y,var.T,kx=kx,ky=ky)
  class DolfinExpression(Expression):
    def eval(self,values,x):
      values[0] = interpolant(x[0],x[1])

  return DolfinExpression


def extrude(f, b, d, ff, Q):
  r"""
  This extrudes a function <f> defined along a boundary <b> out onto
  the domain in the direction <d>.  It does this by formulating a 
  variational problem:

  :Conditions: 
  .. math::
  \frac{\partial u}{\partial d} = 0
  
  u|_b = f

  and solving.  
  
  :param f  : Dolfin function defined along a boundary
  :param b  : Boundary condition
  :param d  : Subdomain over which to perform differentiation
  :param ff : Subdomain FacetFunction 
  :param Q  : FunctionSpace of domain
  """
  # define test and trial based on function space :
  phi = TestFunction(Q)
  v   = TrialFunction(Q)

  # linear PDE :
  a  = v.dx(d) * phi * dx
  L  = DOLFIN_EPS * phi * dx  # really close to zero to fool FFC
  bc = DirichletBC(Q, f, ff, b)

  # solve and return new Function
  v  = Function(Q)
  solve(a == L, v, bc)
  return v


def get_bed_mesh(mesh):
  """
  Returns the bed of <mesh>.
  """
  bmesh   = BoundaryMesh(mesh, 'exterior')
  cellmap = bmesh.entity_map(2)
  pb      = CellFunction("size_t", bmesh, 0)
  for c in cells(bmesh):
    if Facet(mesh, cellmap[c.index()]).normal().z() < 0:
      pb[c] = 1
  submesh = SubMesh(bmesh, pb, 1)           # subset of surface mesh
  return submesh


def plotIce(di, u, directory, cmap='gist_yarg',  scale='lin', name='', 
            numLvls=12, tp=False, tpAlpha=0.5):
  """
  INPUTS :

    di :
      DataInput object with desired projection
    u :
      solution to plot; can be either a function on a 2D mesh, or a string 
      key to matrix variable in <di>.data.
    directory :
      directory string location to save image.
    cmap :
      colormap to use - see images directory for sample and name
    scale :
      scale to plot, either 'log' or 'lin'
    name :
      title of the plot, latex accepted
    numLvls :
      number of levels for field values
    tp :
      boolean determins plotting of triangle overlay
    tpAlpha :
      alpha level of triangles 0.0 (transparent) - 1.0 (opaque)
  
  OUTPUT :
 
    A sigle 250 dpi .png in the source directory.
  
  """
  #=============================================================================
  # data gathering :
  
  # get the original projection coordinates and data :
  if isinstance(u, str):
    vx,vy  = np.meshgrid(di.x, di.y)
    v      = di.data[u]

  elif isinstance(u, Function):
    mesh  = u.function_space().mesh()
    coord = mesh.coordinates()
    fi    = mesh.cells()
    v     = u.compute_vertex_values(mesh)
    vx    = coord[:,0]
    vy    = coord[:,1]

  # get lon,lat from coordinates :
  lon,lat = di.proj(vx, vy, inverse=True)
  
  # the width/height numbers were calculated from vertices from a mesh :
  #w = 1.05 * (vx.max() - vx.min())
  #h = 1.05 * (vy.max() - vy.min())

  # Antarctica :
  if di.cont == 'antarctica':
    w   = 5513335.22665
    h   = 4602848.6605
    fig = plt.figure(figsize=(14,10))
    ax  = fig.add_axes()
    
    # new projection :
    m = Basemap(ax=ax, width=w, height=h, resolution='h', 
                projection='stere', lat_ts=-71, 
                lon_0=0, lat_0=-90)
   
    # draw lat/lon grid lines every 5 degrees.
    # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0, 360, 20.0),
                    color = 'black',
                    labels = [True, False, True, True])
    m.drawparallels(np.arange(-90, 90, 5.0), 
                    color = 'black', 
                    labels = [True, False, True, True])
    m.drawmapscale(-130, -68, 0, -90, 400, 
                   yoffset  = 0.01 * (m.ymax - m.ymin), 
                   barstyle = 'fancy')
 
  # Greenland : 
  elif di.cont == 'greenland':
    w   = 1532453.49654
    h   = 2644074.78236
    fig = plt.figure(figsize=(8,10))
    ax  = fig.add_axes()
    
    # new projection :
    m = Basemap(ax=ax, width=w, height=h, resolution='h', 
                projection='stere', lat_ts=71, 
                lon_0=-41.5, lat_0=71)
    
    # draw lat/lon grid lines every 5 degrees.
    # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0, 360, 5.0),
                    color = 'black',
                    labels = [False, False, False, True])
    m.drawparallels(np.arange(-90, 90, 5.0), 
                    color = 'black', 
                    labels = [True, False, True, False])
    m.drawmapscale(-34, 60.5, -41.5, 71, 400, 
                   yoffset  = 0.01 * (m.ymax - m.ymin), 
                   barstyle = 'fancy')

  # convert to new projection coordinates from lon,lat :
  x, y  = m(lon, lat)
 
  m.drawcoastlines(linewidth=0.25, color = 'black')
  #m.shadedrelief()
  #m.bluemarble()
  #m.etopo()
  

  #=============================================================================
  # plotting :
  
  # countour levels :
  if scale == 'log':
    from matplotlib.ticker import LogFormatter
    vmax                = np.ceil(v.max())
    vmin                = np.floor(v.min())
    v[np.where(v<=1.0)] = 1.0
    levels              = np.logspace(0.0, np.log10(vmax+1), numLvls)
    formatter           = LogFormatter(10, labelOnlyBase=False)
    norm                = colors.LogNorm()
  
  elif scale == 'lin':
    from matplotlib.ticker import ScalarFormatter
    vmin      = np.floor(v.min())
    vmax      = np.ceil(v.max())
    levels    = np.linspace(vmin, vmax+1, numLvls)
    formatter = ScalarFormatter()
    norm      = None
  
  elif scale == 'bool':
    from matplotlib.ticker import ScalarFormatter
    levels    = [0, 1, 2]
    formatter = ScalarFormatter()
    norm      = None
  

  if isinstance(u, str):
    #cs = plt.pcolor(x, y, v, cmap=get_cmap(cmap), norm=norm)
    cs = plt.contourf(x, y, v, levels=levels, 
                      cmap=pl.get_cmap(cmap), norm=norm)
  
  elif isinstance(u, Function):
    #cs = plt.tripcolor(x, y, fi, v, shading='gouraud', 
    #                   cmap=get_cmap(cmap), norm=norm)
    cs = plt.tricontourf(x, y, fi, v, levels=levels, 
                         cmap=pl.get_cmap(cmap), norm=norm)
  
  # plot triangles :
  if tp == True:
    tp = plt.triplot(x, y, fi, '-', lw=0.2, alpha=tpAlpha)

  # include colorbar :
  cbar = m.colorbar(cs, format=formatter, 
                    ticks=np.around(levels,decimals=1), 
                    location='right', pad="5%")
  
  # title :
  #tit = plt.title(name)
  #tit.set_fontsize(40)
  
  plt.savefig(directory + '/' + name + '.png', dpi=250)
  plt.show()


# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
  def __init__(self, u, coef, dcoef):
    self.u     = u
    self.coef  = coef
    self.dcoef = dcoef

  def __call__(self,s):
    return sum([u*c(s) for u,c in zip(self.u, self.coef)])

  def ds(self,s):
    return sum([u*c(s) for u,c in zip(self.u, self.dcoef)])

  def dx(self,s,x):
    return sum([u.dx(x)*c(s) for u,c in zip(self.u, self.coef)])


# SIMILAR TO ABOVE, BUT FOR CALCULATION OF FINITE DIFFERENCE QUANTITIES.
class VerticalFDBasis(object):
  def __init__(self,u, deltax, coef, sigmas):
    self.u      = u 
    self.deltax = deltax
    self.coef   = coef
    self.sigmas = sigmas

  def __call__(self,i):
    return self.u[i]

  def eval(self,s):
    fl   = max(sum(s > self.sigmas)-1,0)
    dist = s - self.sigmas[fl]
    return self.u[fl]*(1 - dist/self.deltax) + self.u[fl+1]*dist/self.deltax

  def ds(self,i):
    return (self.u[i+1] - self.u[i-1])/(2*self.deltax)

  def d2s(self,i):
    return (self.u[i+1] - 2*self.u[i] + self.u[i-1])/(self.deltax**2)

  def dx(self,i,x):
    return self.u[i].dx(x)        


# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
  def __init__(self,points,weights):
    self.points  = points
    self.weights = weights
  def integral_term(self,f,s,w):
    return w*f(s)
  def intz(self,f):
    return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])





