from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import str
from builtins import chr
from builtins import zip
from builtins import range
from builtins import object
from dolfin                  import *
from dolfin_adjoint          import *
from pyproj                  import *
from ufl                     import indexed
from termcolor               import colored, cprint
from mpl_toolkits.basemap    import Basemap
from matplotlib              import colors, ticker
from matplotlib.ticker       import LogFormatter, ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from cslvr.inputoutput       import print_text, DataInput
import matplotlib.pyplot     as plt
import pylab                 as pl
import numpy                 as np
import inspect, re




def raiseNotDefined():
	fileName = inspect.stack()[1][1]
	line     = inspect.stack()[1][2]
	method   = inspect.stack()[1][3]

	text = "*** Method not implemented: %s at line %s of %s"
	print(text % (method, line, fileName))
	sys.exit(1)




class Boundary(object):
	def __init__(self, measure, markers, description):
		self.measure     = measure
		self.description = description
		self.markers     = markers
	def __call__(self):
		for i,m in enumerate(self.markers):
			if i == 0:
				meas  = self.measure(m)
			else:
				meas += self.measure(m)
		return meas




def download_file(url, direc, folder, extract=False):
	"""
	Download a file with url string ``url`` into directory ``direc/folder``.
	If ``extract == True``, extract the .zip or tar.gz file into the directory
	and delete the .zip or tar.gz file.

	:param url:     the URL of the file you want to download
	:param direc:   the directory relative to the script where you want to save
	:param folder:  a folder in directory ``direc`` you want to save into
	:param extract: if the file is compressed as zip or tar.gz, extract it
	:type url:      string
	:type direc:    string
	:type folder:   string
	:type extract:  bool
	"""
	import urllib.request, urllib.error, urllib.parse
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
	fn   = fn.split('?')[0]
	u    = urllib.request.urlopen(url)
	f    = open(direc + fn, 'wb')
	meta = u.info()
	fs   = int(meta.get_all("Content-Length")[0])

	s    = "Downloading: %s Bytes: %s" % (fn, fs)
	print(s)

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

	Args:

	  :m_0:  Initial unrefined mesh
	  :U_ex: Representation of the ice sheet as a dolfin expression

	Original author: Doug Brinkerhoff: https://dbrinkerhoff.org/
	"""
	def __init__(self, m_0, U_ex):
		self.mesh = m_0
		self.U_ex = U_ex

	def refine(self, REFINE_RATIO, hmin, hmax):
		"""
		Determines which cells that are above a certain error threshold based
		on the diameter of the cells

		Args:

		  :REFINE_RATIO: Long value between 0 and 1, inclusive, to select
		                 an error value in a reverse sorted list.
		                 (i.e. .5 selects the midpoint)
		  :hmin:         Minimum diameter of the cells
		  :hmax:         Maximum diameter of the cells

		"""
		print("refining %.2f%% of max error" % (REFINE_RATIO * 100))
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
		print("error midpoint :", e_mid)

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

		Args:

		 :n_layers:       Number of layers in the mesh
		 :workspace_path: Path to the location where the refined meshes
		                  will be written
		 :n_processors:   Number of processers utilized in the extrusion

		"""
		import os
		layers = str(n_layers + 1)

		write_gmsh(self.mesh, workspace_path + "/2dmesh.msh")

		output = open(workspace_path + "/2dmesh.geo", 'w')

		output.write("Merge \"2dmesh.msh\";\n")
		output.write("Extrude {0,0,1} {Surface{0}; Layers{" + layers + "};}")
		output.close()
		print("Extruding mesh (This could take a while).")

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

	Args:

	  :m_0:  Initial unrefined mesh
	  :U_ex: Representation of the ice sheet as a Dolfin expression

	Original author: Doug Brinkerhoff: https://dbrinkerhoff.org/
	"""

	def __init__(self, m_0, U_ex):
		self.mesh = m_0
		self.U_ex = U_ex

	def weighted_smoothing(self, edge_errors, omega=0.1):
		"""
		Smooths the points contained within the mesh

		Args:

		  :edge_errors:  Dolfin edge function containing the calculated
		                 edge errors of the mesh
		  :omega:        Weighting factor used to refine the mesh
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

		for item in list(adjacent_points.items()):
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

		Args:

		  :n_layers:       Number of layers in the mesh
		  :workspace_path: Path to the location where the refined meshes
		                   will be written
		  :n_processors:   Number of processers utilized in the extrusion

		"""
		import os
		layers = str(n_layers + 1)

		write_gmsh(self.mesh, workspace_path + "/2dmesh.msh")

		output = open(workspace_path + "/2dmesh.geo", 'w')

		output.write("Merge \"2dmesh.msh\";\n")
		output.write("Extrude {0,0,1} {Surface{0}; Layers{" + layers + "};}")
		output.close()
		print("Extruding mesh (This could take a while).")

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

	Args:

	  :mesh: Mesh that is to be written to a file
	  :path: Path to write the mesh file to

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
			print("Unknown cell type")

	output.write("$EndElements\n")
	output.close()




def generate_expression_from_gridded_data(x,y,var,kx=1,ky=1):
	"""
	This function creates a dolfin 2D expression from data input

	Args:

	  :x:   List of x coordinates
	  :y:   List of y coordinates
	  :var: List of values associated with each x and y coordinate pair

	Returns:

	  :rtype: A dolfin Expression object representing the data

	"""
	from scipy.interpolate import RectBivariateSpline
	interpolant = RectBivariateSpline(x,y,var.T,kx=kx,ky=ky)
	class DolfinExpression(Expression):
		def eval(self,values,x):
			values[0] = interpolant(x[0],x[1])

	return DolfinExpression




def plot_variable(u, name, direc,
                  figsize             = (8,7),
                  cmap                = 'gist_yarg',
                  scale               = 'lin',
                  numLvls             = 12,
                  levels              = None,
                  levels_2            = None,
                  umin                = None,
                  umax                = None,
                  vec_scale           = None,
                  tp                  = False,
                  tpAlpha             = 0.5,
                  show                = True,
                  hide_ax_tick_labels = False,
                  xlabel              = r'$x$',
                  ylabel              = r'$y$',
                  equal_axes          = True,
                  title               = '',
                  hide_axis           = False,
                  colorbar_loc        = 'right',
                  contour_type        = 'filled',
                  extend              = 'neither',
                  ext                 = '.pdf',
                  res                 = 150,
                  cb                  = True,
                  cb_format           = '%.1e'):
	"""
	"""
	vec  = False
	if len(u.ufl_shape) == 1:
		if type(u[0]) == indexed.Indexed:
			out    = u.split(True)
		else:
			out    = u
		vec      = True
		v        = 0
		mesh     = out[0].function_space().mesh()
		for k in out:
			kv  = k.compute_vertex_values(mesh)
			v  += kv**2
		v = np.sqrt(v + 1e-16)
		v0       = out[0].compute_vertex_values(mesh)
		v1       = out[1].compute_vertex_values(mesh)
		v2_mag   = np.sqrt(v0**2 + v1**2 + 1e-16)
		v0       = v0 / v2_mag
		v1       = v1 / v2_mag
	elif len(u.ufl_shape) == 0:
		mesh     = u.function_space().mesh()
		v        = u.compute_vertex_values(mesh)
	x    = mesh.coordinates()[:,0]
	y    = mesh.coordinates()[:,1]
	t    = mesh.cells()

	d    = os.path.dirname(direc)
	if not os.path.exists(d):
		os.makedirs(d)

	#=============================================================================
	# plotting :
	if umin != None and levels is None:
		vmin = umin
	elif levels is not None:
		vmin = levels.min()
	else:
		vmin = v.min()

	if umax != None and levels is None:
		vmax = umax
	elif levels is not None:
		vmax = levels.max()
	else:
		vmax = v.max()

	# set the extended colormap :
	cmap = pl.get_cmap(cmap)
	cmap.set_bad(color='white', alpha=1.0)

	# countour levels :
	if scale == 'log':
		if levels is None:
			levels    = np.logspace(np.log10(vmin), np.log10(vmax), numLvls)
		#v[v < vmin] = vmin + 2e-16
		#v[v > vmax] = vmax - 2e-16
		formatter   = LogFormatter(10, labelOnlyBase=False)
		norm        = colors.LogNorm()

	# countour levels :
	elif scale == 'sym_log':
		if levels is None:
			levels  = np.linspace(vmin, vmax, numLvls)
		#v[v < vmin] = vmin + 2e-16
		#v[v > vmax] = vmax - 2e-16
		formatter   = LogFormatter(e, labelOnlyBase=False)
		norm        = colors.SymLogNorm(vmin=vmin, vmax=vmax,
		                                linscale=0.001, linthresh=0.001)

	elif scale == 'lin':
		if levels is None:
			levels  = np.linspace(vmin, vmax, numLvls)
		norm = colors.BoundaryNorm(levels, cmap.N)

	elif scale == 'bool':
		v[v < 0.0] = 0.0
		levels  = [0, 1, 2]
		norm    = colors.BoundaryNorm(levels, cmap.N)

	fig = plt.figure(figsize=figsize)
	ax  = fig.add_subplot(111)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if hide_ax_tick_labels:
		ax.set_xticklabels([])
		ax.set_yticklabels([])
	if hide_axis:
		ax.axis('off')
	if equal_axes:
		ax.axis('equal')

	if contour_type == 'filled':
		if scale != 'log':
			cs = ax.tricontourf(x, y, t, v, levels=levels, vmin=vmin, vmax=vmax,
			                   cmap=cmap, norm=norm, extend=extend)
		else:
			cs = ax.tricontourf(x, y, t, v, levels=levels, vmin=vmin, vmax=vmax,
			                   cmap=cmap, norm=norm)
	elif contour_type == 'lines':
		cs = ax.tricontour(x, y, t, v, linewidths=2.0,
		                   levels=levels, colors='k')
		for line in cs.collections:
			if line.get_linestyle() != [(None, None)]:
				line.set_linestyle([(None, None)])
				line.set_color('red')
				line.set_linewidth(1.5)
		if levels_2 is not None:
			cs2 = ax.tricontour(x, y, t, v, levels=levels_2, vmin=vmin, vmax=vmax,
			                    colors='0.30')
			for line in cs2.collections:
				if line.get_linestyle() != [(None, None)]:
					line.set_linestyle([(None, None)])
					line.set_color('#c1000e')
					line.set_linewidth(0.5)
		ax.clabel(cs, inline=1, colors='k', fmt='%i')


	if vec:
		q  = ax.quiver(x, y, v0, v1, pivot='middle',
		                             color='k',
		                             scale=vec_scale,
		                             width=0.004,
		                             headwidth=4.0,
		                             headlength=4.0,
		                             headaxislength=4.0)

	# plot triangles :
	if tp == True:
		tp = ax.triplot(x, y, t, 'k-', lw=0.2, alpha=tpAlpha)

	# this enforces equal axes no matter what (yeah, a hack) :
	divider = make_axes_locatable(ax)

	# include colorbar :
	if cb and scale != 'bool' and contour_type != 'lines':
		cax  = divider.append_axes("right", "5%", pad="3%")
		cbar = fig.colorbar(cs, cax=cax,
		                    ticks=levels, format=cb_format)

	ax.set_xlim([x.min(), x.max()])
	ax.set_ylim([y.min(), y.max()])
	plt.tight_layout(rect=[0,0,1,0.95])

	#pl.mpl.rcParams['axes.titlesize'] = 'small'
	#tit = plt.title(title)

	# title :
	tit = plt.title(title)
	#tit.set_fontsize(40)

	d     = os.path.dirname(direc)
	if not os.path.exists(d):
		os.makedirs(d)
	plt.savefig(direc + name + ext, res=res)
	if show:
		plt.show()
	plt.close(fig)




def plotIce(di, u, name, direc,
            coords           = None,
            cells            = None,
            u2               = None,
            u2_levels        = None,
            u2_color         = 'k',
            u2_linewidth     = 1.0,
            title            = '',
            cmap             = 'gist_yarg',
            scale            = 'lin',
            umin             = None,
            umax             = None,
            numLvls          = 12,
            drawGridLabels   = True,
            levels           = None,
            levels_2         = None,
            tp               = False,
            tpAlpha          = 0.5,
            contour_type     = 'filled',
            params           = None,
            extend           = 'neither',
            show             = True,
            ext              = '.pdf',
            res              = 150,
            cb               = True,
            cb_format        = '%.1e',
            zoom_box         = False,
            zoom_box_kwargs  = None,
            plot_pts         = None,
            plot_texts       = None,
            plot_continent   = False,
            cont_plot_params = None,
            drawcoastlines   = True,
            box_params       = None):
	"""
	Args:

	  :di:      DataInput object with desired projection
	  :u:       solution to plot; can be either a function on a 2D mesh, or a
	            string key to matrix variable in ``di``.data.
	  :name:    title of the plot, latex accepted
	  :direc:   directory string location to save image.
	  :cmap:    colormap to use - see images directory for sample and name
	  :scale:   scale to plot, either 'log', 'lin', or 'bool'
	  :numLvls: number of levels for field values
	  :levels:  manual levels, if desired.
	  :tp:      boolean determins plotting of triangle overlay
	  :tpAlpha: alpha level of triangles 0.0 (transparent) - 1.0 (opaque)
	  :extends: for the colorbar, extend upper range and may be ["neither",
	            "both", "min", "max"]. default is "neither".

	  for plotting the zoom-box, make ``zoom_box`` = True and supply dict
	  *zoom_box_kwargs* with parameters

	  :zoom:             ammount to zoom
	  :loc:              location of box
	  :loc1:             loc of first line
	  :loc2:             loc of second line
	  :x1:               first x-coord
	  :y1:               first y-coord
	  :x2:               second x-coord
	  :y2:               second y-coord
	  :scale_font_color: scale font color
	  :scale_length:     scale length in km
	  :scale_loc:        1=top, 2=bottom
	  :plot_grid:        plot the triangles
	  :axes_color:       color of axes
	  :plot_points:      dict of points to plot

	Returns:

	  A sigle *direc/name.ext* in the source directory.

	"""
	#FIXME: re-factor this method to take generic kwargs for each method called,
	#       and generally improve readability.
	# get the original projection coordinates and data :
	if isinstance(u, str):
		s = "::: plotting %s's \"%s\" field data directly :::" % (di.name, u)
		print_text(s, '242')
		x, y    = np.meshgrid(di.x, di.y)
		v       = di.data[u]

	# if 'u' is a NumPy array and the cell arrays and coordinates are supplied :
	elif (type(u) == np.ndarray or type(u) == list or type(u) == tuple) \
		and len(coords) == 2:
		x = coords[0]
		y = coords[1]
		v = u

		if type(cells) == np.ndarray:
			fi = cells
		else:
			fi = []

		# if there are multiple components to 'u', it is a vector :
		if len(np.shape(u)) > len(np.shape(x)):
			vec = True  # used for plotting below
			vx  = u[0]
			vy  = u[1]
			# compute norm :
			v     = 0
			for k in u:
				v  += k**2
			v     = np.sqrt(v + 1e-16)

	# if 'u' is a NumPy array and cell/coordinate arrays are not supplied :
	elif (type(u) == np.ndarray or type(u) == list or type(u) == tuple) \
		and coords is None:
		print_text(">>> numpy arrays require `coords' <<<", 'red', 1)
		sys.exit(1)

	elif isinstance(u, Function) \
		or isinstance(u, dolfin.functions.function.Function):
		s = "::: plotting FEniCS Function \"%s\" :::" % name
		print_text(s, '242')
		mesh  = u.function_space().mesh()
		coord = mesh.coordinates()
		fi    = mesh.cells()
		v     = u.compute_vertex_values(mesh)
		x     = coord[:,0]
		y     = coord[:,1]

	# get the projection info :
	if isinstance(di, dict) and 'pyproj_Proj' in list(di.keys()) \
		 and 'continent' in list(di.keys()):
		lon,lat = di['pyproj_Proj'](x, y, inverse=True)
		cont    = di['continent']
	elif isinstance(di, DataInput):
		lon,lat = di.proj(x, y, inverse=True)
		cont    = di.cont
	else:
		s = ">>> plotIce REQUIRES A 'DataFactory' DICTIONARY FOR " + \
		    "PROJECTION STORED AS KEY 'pyproj_Proj' AND THE CONTINENT TYPE " + \
		    "STORED AS KEY 'continent' <<<"
		print_text(s, 'red', 1)
		sys.exit(1)

	# Antarctica :
	if params is None:
		if cont is 'antarctica':
			w   = 5513335.22665
			h   = 4602848.6605
			fig = plt.figure(figsize=(7,5))
			ax  = fig.add_subplot(111)

			lon_0 = 0
			lat_0 = -90

			# new projection :
			m = Basemap(ax=ax, width=w, height=h, resolution='h',
			            projection='stere', lat_ts=-71,
			            lon_0=lon_0, lat_0=lat_0)

			offset = 0.015 * (m.ymax - m.ymin)

			# draw lat/lon grid lines every 5 degrees.
			# labels = [left,right,top,bottom]
			if drawGridLabels:
				mer_lab = [True, False, True, True]
				par_lab = [True, False, True, True]
			else:
				mer_lab = [False, False, False, False]
				par_lab = [False, False, False, False]

			m.drawmeridians(np.arange(0, 360, 20.0),
			                color = 'black',
			                linewidth = 0.5,
			                labels = mer_lab,
			                fontsize = 8)
			m.drawparallels(np.arange(-90, 90, 5.0),
			                color = 'black',
			                linewidth = 0.5,
			                labels = par_lab,
			                fontsize = 8)
			m.drawmapscale(-130, -68, 0, -90, 1000,
			               yoffset  = offset,
			               barstyle = 'fancy')

		# Greenland :
		elif cont is 'greenland':
			w   = 1532453.49654
			h   = 2644074.78236
			fig = plt.figure(figsize=(7,10))
			ax  = fig.add_subplot(111)

			lon_0 = -41.5
			lat_0 = 71

			# new projection :
			m = Basemap(ax=ax, width=w, height=h, resolution='h',
			            projection='stere', lat_ts=71,
			            lon_0=lon_0, lat_0=lat_0)

			offset = 0.010 * (m.ymax - m.ymin)

			if drawGridLabels:
				mer_lab = [False, False, False, True]
				par_lab = [True, False, True, False]
			else:
				mer_lab = [False, False, False, False]
				par_lab = [False, False, False, False]

			# draw lat/lon grid lines every 5 degrees.
			# labels = [left,right,top,bottom]
			m.drawmeridians(np.arange(0, 360, 5.0),
			                color = 'black',
			                labels = mer_lab,
			                fontsize = 8)
			m.drawparallels(np.arange(-90, 90, 5.0),
			                color = 'black',
			                labels = par_lab,
			                fontsize = 8)
			m.drawmapscale(-34, 60.5, -41.5, 71, 400,
			               yoffset  = offset,
			               barstyle = 'fancy')

	elif type(params) is dict:
		llcrnrlat      = params['llcrnrlat']
		urcrnrlat      = params['urcrnrlat']
		llcrnrlon      = params['llcrnrlon']
		urcrnrlon      = params['urcrnrlon']
		scale_color    = params['scale_color']
		scale_length   = params['scale_length']
		scale_loc      = params['scale_loc']
		figsize        = params['figsize']
		lat_interval   = params['lat_interval']
		lon_interval   = params['lon_interval']
		plot_grid      = params['plot_grid']
		plot_scale     = params['plot_scale']
		axes_color     = params['axes_color']

		if cont is 'greenland':
			lon_0 = -41.5
			lat_0 =  71
		elif cont is 'antarctica':
			lon_0 =  0
			lat_0 = -90

		fig   = plt.figure(figsize=figsize)
		ax    = fig.add_subplot(111)

		# new projection :
		m = Basemap(ax=ax, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
		            llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='h',
		            projection='stere', lon_0=lon_0, lat_0=lat_0)

		offset = 0.015 * (m.ymax - m.ymin)

		# draw lat/lon grid lines every degree.
		# labels = [left,right,top,bottom]
		if plot_grid:
			m.drawmeridians(np.arange(0, 360, lon_interval),
			                color = 'black',
			                labels = [True, False, True, True],
			                linewidth = 0.5,
			                fontsize = 8)
			m.drawparallels(np.arange(-90, 90, lat_interval),
			                color = 'black',
			                labels = [True, False, True, True],
			                linewidth = 0.5,
			                fontsize = 8)
		else:
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.xaxis.set_ticks_position('none')
			ax.yaxis.set_ticks_position('none')

		if scale_loc == 1:
			y_fact = 1.8
			x_fact = 1.0
		elif scale_loc == 2:
			y_fact = 0.2
			x_fact = 1.0
		elif scale_loc == 3:
			y_fact = 1.8
			x_fact = 0.25

		if plot_scale :
			dx         = (m.xmax - m.xmin)/2.0
			dy         = (m.ymax - m.ymin)/2.0
			xmid       = m.xmin + x_fact*dx
			ymid       = m.ymin + y_fact*dy
			slon, slat = m(xmid, ymid, inverse=True)
			m.drawmapscale(slon, slat, slon, slat, scale_length,
			               barstyle = 'simple', linewidth=1.5, fontcolor=scale_color)

		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_color(axes_color)
			ax.spines[axis].set_linewidth(1.5)

	else:
		s = ">>> plotIce REQUIRES A 'dict' OF SPECIFIC PARAMETERS FOR 'custom' <<<"
		print_text(s, 'red', 1)
		sys.exit(1)


	# convert to new projection coordinates from lon,lat :
	x, y  = m(lon, lat)

	if drawcoastlines:
		m.drawcoastlines(linewidth=0.5, color = 'black')
	#m.shadedrelief()
	#m.bluemarble()
	#m.etopo()

	if plot_continent:
		if cont is 'greenland':
			llcrnrlat  = 57
			urcrnrlat  = 80.1
			llcrnrlon  = -57
			urcrnrlon  = 15

			axcont = inset_locator.inset_axes(ax, **cont_plot_params)
			axcont.xaxis.set_ticks_position('none')
			axcont.yaxis.set_ticks_position('none')

			# continent projection :
			mc = Basemap(ax=axcont, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
			             llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='c',
			             projection='stere', lon_0=lon_0, lat_0=lat_0)

			mc.drawcoastlines(linewidth=0.5, color='black')

			x_c, y_c  = mc(lon, lat)

			v_cont = v.copy()
			v_cont[:] = 1.0

			axcont.tricontourf(x_c, y_c, fi, v_cont, cmap=pl.get_cmap('Reds'))


	#=============================================================================
	# plotting :
	if umin != None and levels is None:
		vmin = umin
	elif levels is not None:
		vmin = levels.min()
	else:
		vmin = np.nanmin(v)

	if umax != None and levels is None:
		vmax = umax
	elif levels is not None:
		vmax = levels.max()
	else:
		vmax = np.nanmax(v)

	# set the extended colormap :
	cmap = pl.get_cmap(cmap)
	#cmap.set_under(under)
	#cmap.set_over(over)

	# countour levels :
	if scale == 'log':
		if levels is None:
			levels    = np.logspace(np.log10(vmin), np.log10(vmax), numLvls)
		v[v < vmin] = vmin + 2e-16
		v[v > vmax] = vmax - 2e-16
		formatter   = LogFormatter(10, labelOnlyBase=False)
		norm        = colors.LogNorm()

	# countour levels :
	elif scale == 'sym_log':
		if levels is None:
			levels  = np.linspace(vmin, vmax, numLvls)
		v[v < vmin] = vmin + 2e-16
		v[v > vmax] = vmax - 2e-16
		formatter   = LogFormatter(e, labelOnlyBase=False)
		norm        = colors.SymLogNorm(vmin=vmin, vmax=vmax,
		                                linscale=0.001, linthresh=0.001)

	elif scale == 'lin':
		if levels is None:
			levels  = np.linspace(vmin, vmax, numLvls)
		norm = colors.BoundaryNorm(levels, cmap.N)

	elif scale == 'bool':
		v[v < 0.0] = 0.0
		levels  = [0, 1, 2]
		norm    = colors.BoundaryNorm(levels, cmap.N)

	# please do zoom in!
	if zoom_box:
		zoom              = zoom_box_kwargs['zoom']
		loc               = zoom_box_kwargs['loc']
		loc1              = zoom_box_kwargs['loc1']
		loc2              = zoom_box_kwargs['loc2']
		llcrnrlat         = zoom_box_kwargs['llcrnrlat']
		urcrnrlat         = zoom_box_kwargs['urcrnrlat']
		llcrnrlon         = zoom_box_kwargs['llcrnrlon']
		urcrnrlon         = zoom_box_kwargs['urcrnrlon']
		plot_zoom_scale   = zoom_box_kwargs['plot_zoom_scale']
		scale_font_color  = zoom_box_kwargs['scale_font_color']
		scale_length      = zoom_box_kwargs['scale_length']
		scale_loc         = zoom_box_kwargs['scale_loc']
		plot_grid         = zoom_box_kwargs['plot_grid']
		axes_color        = zoom_box_kwargs['axes_color']
		zb_plot_pts       = zoom_box_kwargs['plot_points']

		x1, y1 = m(llcrnrlon, llcrnrlat)
		x2, y2 = m(urcrnrlon, urcrnrlat)

		axins = inset_locator.zoomed_inset_axes(ax, zoom, loc=loc)
		inset_locator.mark_inset(ax, axins, loc1=loc1, loc2=loc2,
		                         fc="none", ec=axes_color)
		for axis in ['top','bottom','left','right']:
			axins.spines[axis].set_color(axes_color)
			#axins.spines[axis].set_linewidth(2)

		if scale_loc == 1:
			fact = 1.8
		elif scale_loc == 2:
			fact = 0.2

		dx         = (x2 - x1)/2.0
		dy         = (y2 - y1)/2.0
		xmid       = x1 + dx
		ymid       = y1 + fact*dy
		slon, slat = m(xmid, ymid, inverse=True)

		# new projection :
		mn = Basemap(ax=axins, width=w, height=h, resolution='h',
		             projection='stere', lat_ts=lat_0,
		             lon_0=lon_0, lat_0=lat_0)

		if plot_zoom_scale:
			mn.drawmapscale(slon, slat, slon, slat, scale_length,
			                yoffset  = 0.025 * 2.0 * dy,
			                barstyle = 'fancy', fontcolor=scale_font_color)

		if drawcoastlines:
			mn.drawcoastlines(linewidth=0.5, color = 'black')

		axins.set_xlim(x1, x2)
		axins.set_ylim(y1, y2)

		axins.xaxis.set_ticks_position('none')
		axins.yaxis.set_ticks_position('none')

	if isinstance(u, str):
		#cs = ax.pcolor(x, y, v, cmap=get_cmap(cmap), norm=norm)
		if contour_type == 'filled':
			if scale != 'log':
				cs = ax.contourf(x, y, v, levels=levels, vmin=vmin, vmax=vmax,
				                 cmap=cmap, norm=norm, extend=extend)
			else:
				cs = ax.contourf(x, y, v, levels=levels, vmin=vmin, vmax=vmax,
				                 cmap=cmap, norm=norm)
			if zoom_box:
				if scale != 'log':
					axins.contourf(x, y, v, levels=levels, vmin=vmin, vmax=vmax,
					               cmap=cmap, norm=norm, extend=extend)
				else:
					axins.contourf(x, y, v, levels=levels, vmin=vmin, vmax=vmax,
					               cmap=cmap, norm=norm)
		if contour_type == 'lines':
			cs = ax.contour(x, y, v, levels=levels, colors='k',
			                vmin=vmin, vmax=vmax,)
			for line in cs.collections:
				if line.get_linestyle() != [(None, None)]:
					line.set_linestyle([(None, None)])
					line.set_color('red')
					line.set_linewidth(1.5)
			if levels_2 is not None:
				cs2 = ax.contour(x, y, v, levels=levels_2, colors='k')
				for line in cs2.collections:
					if line.get_linestyle() != [(None, None)]:
						line.set_linestyle([(None, None)])
						line.set_color('#c1000e')
						line.set_linewidth(0.5)
			ax.clabel(cs, inline=1, colors='k', fmt='%i')
			if zoom_box:
				axins.contour(x, y, v, levels=levels, colors='k', vmin=vmin, vmax=vmax)

	elif isinstance(u, Function) \
		or isinstance(u, dolfin.functions.function.Function) \
		or (type(u) == np.ndarray or type(u) == list or type(u) == tuple):
		#cs = ax.tripcolor(x, y, fi, v, shading='gouraud',
		#                  cmap=get_cmap(cmap), norm=norm)
		if contour_type == 'filled':
			if scale != 'log':
				cs = ax.tricontourf(x, y, fi, v, levels=levels, vmin=vmin, vmax=vmax,
				                   cmap=cmap, norm=norm, extend=extend)
			else:
				cs = ax.tricontourf(x, y, fi, v, levels=levels, vmin=vmin, vmax=vmax,
				                   cmap=cmap, norm=norm)
			if zoom_box:
				if scale != 'log':
					axins.tricontourf(x, y, fi, v, levels=levels, vmin=vmin, vmax=vmax,
					                  cmap=cmap, norm=norm, extend=extend)
				else:
					axins.tricontourf(x, y, fi, v, levels=levels, vmin=vmin, vmax=vmax,
					                  cmap=cmap, norm=norm)
		elif contour_type == 'lines':
			cs = ax.tricontour(x, y, fi, v, linewidths=2.0, vmin=vmin, vmax=vmax,
			                   levels=levels, colors='k')
			for line in cs.collections:
				if line.get_linestyle() != [(None, None)]:
					line.set_linestyle([(None, None)])
					line.set_color('red')
					line.set_linewidth(1.5)
			if levels_2 is not None:
				cs2 = ax.tricontour(x, y, fi, v, levels=levels_2, vmin=vmin, vmax=vmax,
				                    colors='0.30')
				for line in cs2.collections:
					if line.get_linestyle() != [(None, None)]:
						line.set_linestyle([(None, None)])
						line.set_color('#c1000e')
						line.set_linewidth(0.5)
			ax.clabel(cs, inline=1, colors='k', fmt='%i')
			if zoom_box:
				axins.tricontour(x, y, fi, v, levels=levels, vmin=vmin, vmax=vmax,
				                 colors='k')
				axins.clabel(cs, inline=1, colors='k', fmt='%1.2f')

	if u2 is not None:
		if isinstance(u2, str):
			v2   = di.data[u2]
			csu2 = ax.contour(x, y, v2, levels=u2_levels,
			                  linewidths=u2_linewidth, colors=u2_color)
		elif isinstance(u2, Function) \
			or isinstance(u2, dolfin.functions.function.Function):
			v2 = u2.compute_vertex_values(mesh)
			csu2 = ax.tricontour(x, y, fi, v2, linewidths=u2_linewidth,
			                     levels=u2_levels, colors=u2_color)
		#for line in csu2.collections:
		#  if line.get_linestyle() != [(None, None)]:
		#    line.set_linestyle([(None, None)])

	if plot_pts is not None:
		lat_a = plot_pts['lat']
		lon_a = plot_pts['lon']
		sty_a = plot_pts['style']
		clr_a = plot_pts['color']
		for lat_i, lon_i, sty_i, clr_i in zip(lat_a, lon_a, sty_a, clr_a):
			x_i, y_i = m(lon_i, lat_i)
			ax.plot(x_i, y_i, color=clr_i, marker=sty_i)

	if plot_texts is not None:
		lat_a = plot_texts['lat']
		lon_a = plot_texts['lon']
		txt_a = plot_texts['text']
		clr_a = plot_texts['color']
		for lat_i, lon_i, txt_i, clr_i in zip(lat_a, lon_a, txt_a, clr_a):
			x_i, y_i = m(lon_i, lat_i)
			ax.text(x_i, y_i, txt_i, color=clr_i)

	if box_params is not None:
		x1,y1   = m(box_params['llcrnrlon'], box_params['llcrnrlat'])
		x2,y2   = m(box_params['urcrnrlon'], box_params['urcrnrlat'])
		box_x_s = [x1,x2,x2,x1,x1]
		box_y_s = [y1,y1,y2,y2,y1]
		ax.plot(box_x_s, box_y_s, '-', lw=1.0, color=box_params['color'])

	if zoom_box:
		if zb_plot_pts is not None:
			lat_a = zb_plot_pts['lat']
			lon_a = zb_plot_pts['lon']
			sty_a = zb_plot_pts['style']
			clr_a = zb_plot_pts['color']
			for lat_i, lon_i, sty_i, clr_i in zip(lat_a, lon_a, sty_a, clr_a):
				x_i, y_i = m(lon_i, lat_i)
				axins.plot(x_i, y_i, color=clr_i, marker=sty_i)

	# plot triangles :
	if tp == True:
		tp = ax.triplot(x, y, fi, 'k-', lw=0.2, alpha=tpAlpha)
	if zoom_box and plot_grid:
		tpaxins = axins.triplot(x, y, fi, 'k-', lw=0.2, alpha=tpAlpha)

	# include colorbar :
	if cb and scale != 'bool':
		divider = make_axes_locatable(ax)#plt.gca())
		cax  = divider.append_axes("right", "5%", pad="3%")
		cbar = fig.colorbar(cs, cax=cax, #format=formatter,
		                    ticks=levels, format=cb_format)
		#cbar = plt.colorbar(cs, cax=cax, format=formatter,
		#                    ticks=np.around(levels,decimals=1))

	# title :
	tit = plt.title(title)
	#tit.set_fontsize(40)

	plt.tight_layout(rect=[.03,.03,0.97,0.97])
	d     = os.path.dirname(direc)
	if not os.path.exists(d):
		os.makedirs(d)
	plt.savefig(direc + name + ext, res=res)
	if show:
		plt.show()
	plt.close(fig)

	return m



def plot_l_curve(out_dir, control):
	r"""
	This function will generate a plot of the data collected by executing the
	:func:`~model.Model.L_curve` function.

	:param out_dir: Directory of data; must be the same as that used to initialize
	                the :class:`~model.Model` instance.
	:param control: The name of the control variable.
	:type out_dir:  string
	:type control:  string
	"""
	# save the resulting functional values and alphas to CSF :
	if MPI.rank(mpi_comm_world())==0:

		# iterate through the directiories we just created and grab the data :
		alphas = []
		J_logs = []
		J_l2s  = []
		J_rats = []
		J_abss = []
		R_tvs  = []
		R_tiks = []
		R_sqs  = []
		R_abss = []
		ns     = []
		for d in next(os.walk(out_dir))[1]:
			m = re.search('(alpha_)(\d\W\dE\W\d+)', d)
			if m is not None:
				do = out_dir + d + '/objective_ftnls_history/'
				alphas.append(float(m.group(2)))
				J_logs.append( np.loadtxt(do  + 'J_log.txt'))
				J_l2s.append(  np.loadtxt(do  + 'J_l2.txt'))
				J_rats.append( np.loadtxt(do  + 'J_rat.txt'))
				J_abss.append( np.loadtxt(do  + 'J_abs.txt'))
				R_tvs.append(  np.loadtxt(do  + 'R_tv_%s.txt'  % control))
				R_tiks.append( np.loadtxt(do  + 'R_tik_%s.txt' % control))
				R_sqs.append(  np.loadtxt(do  + 'R_sq_%s.txt'  % control))
				R_abss.append( np.loadtxt(do  + 'R_abs_%s.txt' % control))
				ns.append(len(J_logs[-1]))
		alphas = np.array(alphas)
		J_logs = np.array(J_logs)
		J_l2s  = np.array(J_l2s)
		J_rats = np.array(J_rats)
		J_abss = np.array(J_abss)
		R_tvs  = np.array(R_tvs)
		R_tiks = np.array(R_tiks)
		R_sqs  = np.array(R_sqs)
		R_abss = np.array(R_abss)
		ns     = np.array(ns)

		# sort everything :
		idx    = np.argsort(alphas)
		alphas = alphas[idx]
		J_logs = J_logs[idx]
		J_l2s  = J_l2s[idx]
		J_rats = J_rats[idx]
		J_abss = J_abss[idx]
		R_tvs  = R_tvs[idx]
		R_tiks = R_tiks[idx]
		R_sqs  = R_sqs[idx]
		R_abss = R_abss[idx]
		ns     = ns[idx]

		# plot the functionals :
		#=========================================================================
		fig = plt.figure(figsize=(6,2.5))
		ax  = fig.add_subplot(111)

		# we want to plot the different alpha values a different shade :
		cmap = plt.get_cmap('viridis')
		colors = [ cmap(x) for x in np.linspace(0, 1, 8) ]

		# the subscripts of file names :
		subs = ['log', 'l2',  'rat', 'abs', 'tv',  'tik', 'sq',  'abs']

		# the functional type
		ftnl_t = ['J', 'J', 'J', 'J', 'R', 'R', 'R', 'R']

		# collect the functionals :
		ftnl_a = [J_logs, J_l2s,  J_rats, J_abss, R_tvs,  R_tiks, R_sqs,  R_abss]

		k    = 0    # counter so we can plot side-by-side
		ints = [0]  # to modify the x-axis labels
		for i in range(len(alphas)):

			# create the x-interval to plot :
			xi = np.arange(k, k + ns[i])
			ints.append(xi.max())

			# if this is the first iteration, we put a legend on it :
			if i == 0:
				for ft, nm, sub, c in zip(ftnl_a, ftnl_t, subs, colors):
					ax.plot(xi, ft[i],  '-',  c=c,  lw=1.5, alpha=0.8,
					        label = r'$\mathscr{%s}_{\mathrm{%s}}$' % (nm, sub))

			# otherwise, we don't need cluttered legends :
			else:
				for ft, c in zip(ftnl_a, colors):
					ax.plot(xi, ft[i],  '-',  c=c,  lw=1.5, alpha=0.8,)

			k += ns[i] - 1

		ints = np.array(ints)

		label = []
		for i in alphas:
			label.append(r'$\gamma = %g$' % i)

		# reset the x-label to be meaningfull :
		ax.set_xticks(ints)
		ax.set_xticklabels(label, size='small', ha='left', rotation=-30)
		ax.set_xlabel(r'relative iteration')
		ax.set_xlim([0, ints[-1]])

		ax.xaxis.grid()
		ax.set_yscale('log')

		# plot the functional legend across the top in a row :
		#leg = ax.legend(loc='upper center', ncol=4)
		#leg = ax.legend(loc='center right')
		leg = ax.legend(bbox_to_anchor=(1.2,0.5), loc='center right', ncol=1)
		leg.get_frame().set_alpha(0.0)
		leg.get_frame().set_color('w')

		plt.tight_layout(rect=[0, 0, 1, 1])
		plt.savefig(out_dir + 'convergence.pdf')
		plt.close(fig)

		# plot L-curve :
		#=========================================================================

		colors    = [cmap(x) for x in np.linspace(0,1,len(alphas))]
		fig, ax_a = plt.subplots(4,4, figsize=(10,10))
		J_lbl     = r'$\mathscr{J}_{\mathrm{%s}}$'
		R_lbl     = r'$\mathscr{R}_{\mathrm{%s}}$'

		# get the last element of each array (needed, as it might be jagged) :
		J = []
		for f in ftnl_a:
			J_i = []
			for j in f:
				J_i.append(j[-1])
			J.append(J_i)
		J = np.array(J)

		# for each cost functional i :
		for i in range(4):
			J_i = J[i]
			# and for each regularization functional j :
			for j in range(4):
				R_j = J[4+j]
				ax_a[j,i].plot(J_i, R_j, '-', c='k', lw=1.5)
				# and for each regularization parameter k :
				for k in range(len(alphas)):
					ax_a[j,i].plot(J_i[k], R_j[k], 'o', c=colors[k], lw=2.0,
					               label = r'$\gamma = %g$' % alphas[k])

				#ax_a[i,j].grid()
				ax_a[j,i].set_yscale('log')
				ax_a[j,i].set_xscale('log')

				# format the labels for less clutter :
				if j == 3:
					ax_a[j,i].set_xlabel(J_lbl % subs[i])
				else:
					ax_a[j,i].xaxis.set_major_formatter(NullFormatter())
					ax_a[j,i].xaxis.set_minor_formatter(NullFormatter())
				if i == 0:
					ax_a[j,i].set_ylabel(R_lbl % subs[4+j])
				else:
					ax_a[j,i].yaxis.set_major_formatter(NullFormatter())
					ax_a[j,i].yaxis.set_minor_formatter(NullFormatter())

				# set the legend for only one plot :
				if i == 0 and j == 0:
					leg = ax_a[j,i].legend(loc='upper right', ncol=1, fontsize=5)
					leg.get_frame().set_alpha(0.0)

		plt.tight_layout()
		plt.savefig(out_dir + 'l_curve.pdf')
		plt.close(fig)

		return ftnl_a



# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
	"""
	Original author: Doug Brinkerhoff: https://dbrinkerhoff.org/
	"""
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
	"""
	Original author: Doug Brinkerhoff: https://dbrinkerhoff.org/
	"""
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




# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA,
# QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
	def __init__(self, order=4):
		if order == 4:
			points  = np.array([0.0,       0.4688, 0.8302, 1.0   ])
			weights = np.array([0.4876/2., 0.4317, 0.2768, 0.0476])
		if order == 6:
			points  = np.array([1.0,     0.89976,   0.677186, 0.36312,   0.0        ])
			weights = np.array([0.02778, 0.1654595, 0.274539, 0.3464285, 0.371519/2.])
		if order == 8:
			points  = np.array([1,         0.934001, 0.784483,
			                    0.565235,  0.295758, 0          ])
			weights = np.array([0.0181818, 0.10961,  0.18717,
			                    0.248048,  0.28688,  0.300218/2.])
		self.points  = points
		self.weights = weights
	def integral_term(self,f,s,w):
		return w*f(s)
	def intz(self,f):
		return sum([self.integral_term(f,s,w)
		            for s,w in zip(self.points,self.weights)])





