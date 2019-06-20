from __future__        import division
from dolfin            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import print_text, get_text, print_min_max
from cslvr.model       import Model
from cslvr.helper      import Boundary
from scipy.spatial     import cKDTree
from copy              import copy
import numpy               as np
import matplotlib.pyplot   as plt
import sys, os

class D3Model(Model):
	"""
	"""
	OMEGA_GND   = 0   # internal cells over bedrock
	OMEGA_FLT   = 1   # internal cells over water
	GAMMA_S_GND = 2   # grounded upper surface
	GAMMA_B_GND = 3   # grounded lower surface (bedrock)
	GAMMA_S_FLT = 6   # shelf upper surface
	GAMMA_B_FLT = 5   # shelf lower surface
	GAMMA_L_DVD = 7   # basin divides
	GAMMA_L_OVR = 4   # terminus over water
	GAMMA_L_UDR = 10  # terminus under water
	GAMMA_U_GND = 8   # grounded surface with U observations
	GAMMA_U_FLT = 9   # shelf surface with U observations
	GAMMA_ACC   = 1   # areas with positive surface accumulation

	# external boundaries :
	ext_boundaries = {GAMMA_S_GND : 'grounded upper surface',
	                  GAMMA_B_GND : 'grounded lower surface (bedrock)',
	                  GAMMA_S_FLT : 'shelf upper surface',
	                  GAMMA_B_FLT : 'shelf lower surface',
	                  GAMMA_L_DVD : 'basin divides',
	                  GAMMA_L_OVR : 'terminus over water',
	                  GAMMA_L_UDR : 'terminus under water',
	                  GAMMA_U_GND : 'grounded upper surface with U observations',
	                  GAMMA_U_FLT : 'shelf upper surface with U observations',
	                  GAMMA_ACC   : 'upper surface with accumulation'}

	# internal boundaries :
	int_boundaries = {OMEGA_GND   : 'internal cells located over bedrock',
	                  OMEGA_FLT   : 'internal cells located over water'}

	# union :
	boundaries = {'OMEGA' : int_boundaries,
	              'GAMMA' : ext_boundaries}

	def __init__(self, mesh, out_dir='./results/', order=1,
		           use_periodic=False):
		"""
		A three-dimensional model.
		"""
		s = "::: INITIALIZING 3D MODEL :::"
		print_text(s, cls=self)

		Model.__init__(self, mesh, out_dir, order, use_periodic)

	def color(self):
		return '130'

	def generate_pbc(self):
		"""
		return a SubDomain of periodic lateral boundaries.
		"""
		s = "    - using 3D periodic boundaries -"
		print_text(s, cls=self)

		xmin = MPI.min(MPI.comm_world, self.mesh.coordinates()[:,0].min())
		xmax = MPI.max(MPI.comm_world, self.mesh.coordinates()[:,0].max())
		ymin = MPI.min(MPI.comm_world, self.mesh.coordinates()[:,1].min())
		ymax = MPI.max(MPI.comm_world, self.mesh.coordinates()[:,1].max())

		class PeriodicBoundary(SubDomain):

			def inside(self, x, on_boundary):
				"""
				Return True if on left or bottom boundary AND NOT on one
				of the two corners (0, 1) and (1, 0).
				"""
				return bool((near(x[0], xmin) or near(x[1], ymin)) and \
				            (not ((near(x[0], xmin) and near(x[1], ymax)) \
				             or (near(x[0], xmax) and near(x[1], ymin)))) \
				             and on_boundary)

			def map(self, x, y):
				"""
				Remap the values on the top and right sides to the bottom and left
				sides.
				"""
				if near(x[0], xmax) and near(x[1], ymax):
					y[0] = x[0] - xmax
					y[1] = x[1] - ymax
					y[2] = x[2]
				elif near(x[0], xmax):
					y[0] = x[0] - xmax
					y[1] = x[1]
					y[2] = x[2]
				elif near(x[1], ymax):
					y[0] = x[0]
					y[1] = x[1] - ymax
					y[2] = x[2]
				else:
					y[0] = x[0]
					y[1] = x[1]
					y[2] = x[2]

		self.pBC = PeriodicBoundary()

	def set_mesh(self, mesh):
		"""
		Sets the mesh.

		:param mesh : Dolfin mesh to be written
		"""
		super(D3Model, self).set_mesh(mesh)

		s = "::: setting 3D mesh :::"
		print_text(s, cls=self)

		self.mesh.init(1,2)
		if self.dim != 3:
			s = ">>> 3D MODEL REQUIRES A 3D MESH, EXITING <<<"
			print_text(s, 'red', 1)
			sys.exit(1)
		else:
			self.num_facets   = self.mesh.num_facets()
			self.num_cells    = self.mesh.num_cells()
			self.num_vertices = self.mesh.num_vertices()
		s = "    - %iD mesh set, %i cells, %i facets, %i vertices - " \
		    % (self.dim, self.num_cells, self.num_facets, self.num_vertices)
		print_text(s, cls=self)

		# create a copy of the non-deformed mesh :
		self.flat_mesh = Mesh(self.mesh)

	def set_srf_mesh(self, srfmesh):
		"""
		Set the surface boundary mesh.
		"""
		s = "::: setting surface boundary mesh :::"
		print_text(s, cls=self)

		if isinstance(srfmesh, dolfin.cpp.io.HDF5File):
			self.srfmesh = Mesh()
			srfmesh.read(self.srfmesh, 'srfmesh', False)

		elif isinstance(srfmesh, dolfin.cpp.mesh.Mesh):
			self.srfmesh = srfmesh

	def set_bed_mesh(self, bedmesh):
		"""
		Set the basal boundary mesh.
		"""
		s = "::: setting basal boundary mesh :::"
		print_text(s, cls=self)

		if isinstance(bedmesh, dolfin.cpp.io.HDF5File):
			self.bedmesh = Mesh()
			bedmesh.read(self.bedmesh, 'bedmesh', False)

		elif isinstance(bedmesh, dolfin.cpp.mesh.Mesh):
			self.bedmesh = bedmesh

	def set_lat_mesh(self, latmesh):
		"""
		Set the lateral boundary mesh.
		"""
		s = "::: setting lateral boundary mesh :::"
		print_text(s, cls=self)

		if isinstance(latmesh, dolfin.cpp.io.HDF5File):
			self.latmesh = Mesh()
			latmesh.read(self.latmesh, 'latmesh', False)

		elif isinstance(latmesh, dolfin.cpp.mesh.Mesh):
			self.latmesh = latmesh

		self.Q_lat = FunctionSpace(self.latmesh, 'CG', 1)

	def set_dvd_mesh(self, dvdmesh):
		"""
		Set the lateral divide boundary mesh.
		"""
		s = "::: setting lateral divide boundary mesh :::"
		print_text(s, cls=self)

		if isinstance(dvdmesh, dolfin.cpp.io.HDF5File):
			self.dvdmesh = Mesh()
			dvdmesh.read(self.dvdmesh, 'dvdmesh', False)

		elif isinstance(dvdmesh, dolfin.cpp.mesh.Mesh):
			self.dvdmesh = dvdmesh

		self.Q_dvd = FunctionSpace(self.dvdmesh, 'CG', 1)

	def form_srf_mesh(self):
		"""
		sets self.srfmesh, the surface boundary mesh for this model instance.
		"""
		s = "::: extracting surface mesh :::"
		print_text(s, cls=self)

		bmesh   = BoundaryMesh(self.mesh, 'exterior')
		cellmap = bmesh.entity_map(2)
		pb      = MeshFunction("size_t", bmesh, 2, 0)
		for c in cells(bmesh):
			if Facet(self.mesh, cellmap[c.index()]).normal().z() > 1e-3:
				pb[c] = 1
		submesh = SubMesh(bmesh, pb, 1)
		self.srfmesh = submesh

	def form_bed_mesh(self):
		"""
		sets self.bedmesh, the basal boundary mesh for this model instance.
		"""
		s = "::: extracting bed mesh :::"
		print_text(s, cls=self)

		bmesh   = BoundaryMesh(self.mesh, 'exterior')
		cellmap = bmesh.entity_map(2)
		pb      = MeshFunction("size_t", bmesh, 2, 0)
		for c in cells(bmesh):
			if Facet(self.mesh, cellmap[c.index()]).normal().z() < -1e-3:
				pb[c] = 1
		submesh = SubMesh(bmesh, pb, 1)
		self.bedmesh = submesh

	def form_lat_mesh(self):
		"""
		sets self.latmesh, the lateral boundary mesh for this model instance.
		"""
		s = "::: extracting lateral mesh :::"
		print_text(s, cls=self)

		bmesh   = BoundaryMesh(self.mesh, 'exterior')
		cellmap = bmesh.entity_map(2)
		pb      = MeshFunction("size_t", bmesh, 2, 0)
		for c in cells(bmesh):
			if abs(Facet(self.mesh, cellmap[c.index()]).normal().z()) < 1e-3:
				pb[c] = 1
		submesh = SubMesh(bmesh, pb, 1)
		self.latmesh = submesh

	def form_dvd_mesh(self, contour):
		"""
		sets self.dvdmesh, the lateral divide boundary mesh for this model instance.

		:param contour: NumPy array of exterior points to exclude.
		:type contour:  :class:`numpy.ndarray`
		"""
		s = "::: extracting lateral divide mesh :::"
		print_text(s, cls=self)

		tree = cKDTree(contour)

		bmesh   = BoundaryMesh(self.mesh, 'exterior')
		cellmap = bmesh.entity_map(2)
		pb      = MeshFunction("size_t", bmesh, 2, 0)
		for c in cells(bmesh):
			f       = Facet(self.mesh, cellmap[c.index()])
			n       = f.normal()
			x_m     = f.midpoint().x()
			y_m     = f.midpoint().y()
			z_m     = f.midpoint().z()
			if abs(n.z()) < 1e-3 and tree.query((x_m, y_m))[0] > 1000:
				pb[c] = 1
		submesh = SubMesh(bmesh, pb, 1)
		self.dvdmesh = submesh

	def generate_function_spaces(self):
		"""
		Generates the appropriate finite-element function spaces from parameters
		specified in the config file for the model.
		"""
		super(D3Model, self).generate_function_spaces()

		s = "::: generating 3D function spaces :::"
		print_text(s, cls=self)

		self.Q_flat  = FunctionSpace(self.flat_mesh, self.Q1e,
		                             constrained_domain=self.pBC)
		self.Q2_flat = FunctionSpace(self.flat_mesh, self.QM2e,
		                             constrained_domain=self.pBC)
		self.V_flat  = FunctionSpace(self.flat_mesh, self.Ve,
		                             constrained_domain=self.pBC)

		s = "    - 3D function spaces created - "
		print_text(s, cls=self)

	def assign_from_submesh_variable(self, u, u_sub, surface="upper"):
		r"""
		Assign the values from the variable ``u_sub`` defined on a submesh of
		this :class:`~model.Model`'s mesh to the variable ``u``.
		"""
		n = len(u.function_space().split())

		# TODO: make this work for arbitrary function spaces
		# pick the right function :
		if   n == 0:    u_flat = self.u_flat
		elif n == 2:    u_flat = self.u2_flat
		elif n == 3:    u_flat = self.u3_flat

		# first, Lagrange interpolate the submesh data onto the flat mesh :
		self.Lg.interpolate(u_flat, u_sub)

		# then update the 3D variable :
		u.vector().set_local(u_flat.vector().get_local())
		u.vector().apply('insert')

		# finally, extrude the function throughout the domain :
		if   surface == 'upper':
			self.assign_variable(u, self.vert_extrude(u, d='down'))
		elif surface == 'lower':
			self.assign_variable(u, self.vert_extrude(u, d='up'))

	def assign_to_submesh_variable(self, u, u_sub):
		r"""
		Assign the values from the variable ``u`` defined on a 3D mesh used
		with a :class:`~model.D3Model` to the submesh variable ``u_sub``.
		"""
		n = len(u.function_space().split())

		# TODO: make this work for arbitrary function spaces
		# pick the right function :
		if   n == 0:    u_flat = self.u_flat
		elif n == 2:    u_flat = self.u2_flat
		elif n == 3:    u_flat = self.u3_flat

		# first, update the flat-mesh variable :
		u_flat.vector().set_local(u.vector().get_local())
		u_flat.vector().apply('insert')

		# then, simply interpolate it onto the submesh variable :
		self.Lg.interpolate(u_sub, u_flat)

	def calculate_boundaries(self,
	                         mask        = None,
	                         S_ring      = None,
	                         U_mask      = None,
	                         mark_divide = False,
	                         contour     = None):
		"""
		Determines the boundaries of the current model mesh
		"""
		s = "::: calculating boundaries :::"
		print_text(s, cls=self)

		# save this for physics (momentum classes) :
		self.mark_divide = mark_divide

		if    (contour is     None and     mark_divide) \
		   or (contour is not None and not mark_divide):
			s = ">>> IF PARAMETER <mark_divide> OF calculate_boundaries() IS " + \
			    "TRUE, PARAMETER <contour> MUST BE A NUMPY ARRAY OF COORDINATES " + \
			    "OF THE ICE-SHEET EXTERIOR BOUNDARY <<<"
			print_text(s, 'red', 1)
			sys.exit(1)

		elif contour is not None and mark_divide:
			s = "    - marking the interior facets for incomplete meshes -"
			print_text(s, cls=self)
			tree = cKDTree(contour)
			self.contour_tree = tree # save this for ``self.form_dvd_mesh()``

		# this function contains markers which may be applied to facets of the mesh
		self.ff      = MeshFunction('size_t', self.mesh, 2, 0)
		self.ff_acc  = MeshFunction('size_t', self.mesh, 2, 0)
		self.cf      = MeshFunction('size_t', self.mesh, 3, 0)
		dofmap       = self.Q.dofmap()

		# default to all grounded ice :
		if mask == None:
			mask = Expression('1.0', element=self.Q.ufl_element())

		# default to zero accumulation :
		if S_ring == None:
			S_ring = Expression('0.0', element=self.Q.ufl_element())

		# convert constants to expressions that can be evaluated at (x,y,z) :
		elif    type(S_ring) == Constant \
		     or type(S_ring) == float \
		     or type(S_ring) == int:
			S_ring = Expression('S_ring', S_ring=S_ring, degree=0)

		# default to U observations everywhere :
		if U_mask == None:
			U_mask = Expression('1.0', element=self.Q.ufl_element())

		self.init_S_ring(S_ring)
		self.init_mask(mask)
		self.init_U_mask(U_mask)

		tol = 1e-6

		# iterate through the facets and mark each if on a boundary :
		#
		#   2 = high slope, upward facing ................ grounded surface
		#   3 = grounded high slope, downward facing ..... grounded base
		#   4 = low slope, upward or downward facing ..... sides
		#   5 = floating ................................. floating base
		#   6 = floating ................................. floating surface
		#   7 = floating sides
		#
		# facet for accumulation :
		#
		#   1 = high slope, upward facing ................ positive S_ring
		s = "    - iterating through %i facets - " % self.num_facets
		print_text(s, cls=self)
		for f in facets(self.mesh):
			n        = f.normal()
			x_m      = f.midpoint().x()
			y_m      = f.midpoint().y()
			z_m      = f.midpoint().z()
			mask_xy  = mask(x_m, y_m, z_m)

			if   n.z() >=  tol and f.exterior():
				S_ring_xy   = S_ring(x_m, y_m, z_m)
				U_mask_xy = U_mask(x_m, y_m, z_m)
				if S_ring_xy > 0:
					self.ff_acc[f] = self.GAMMA_ACC
				if mask_xy > 1:
					if U_mask_xy > 0:
						self.ff[f] = self.GAMMA_U_FLT
					else:
						self.ff[f] = self.GAMMA_S_FLT
				else:
					if U_mask_xy > 0:
						self.ff[f] = self.GAMMA_U_GND
					else:
						self.ff[f] = self.GAMMA_S_GND

			elif n.z() <= -tol and f.exterior():
				if mask_xy > 1:
					self.ff[f] = self.GAMMA_B_FLT
				else:
					self.ff[f] = self.GAMMA_B_GND

			elif n.z() >  -tol and n.z() < tol and f.exterior():
				# if we want to use a basin, we need to mark the interior facets :
				if mark_divide:
					if tree.query((x_m, y_m))[0] < 1000:
						if z_m > 0:
							self.ff[f] = self.GAMMA_L_OVR
						else:
							self.ff[f] = self.GAMMA_L_UDR
					else:
						self.ff[f] = self.GAMMA_L_DVD
				# otherwise just mark for over (4) and under (10) water :
				else:
					if z_m > 0:
						self.ff[f] = self.GAMMA_L_OVR
					else:
						self.ff[f] = self.GAMMA_L_UDR

		s = "    - done - "
		print_text(s, cls=self)

		s = "    - iterating through %i cells - " % self.num_cells
		print_text(s, cls=self)
		for c in cells(self.mesh):
			x_m     = c.midpoint().x()
			y_m     = c.midpoint().y()
			z_m     = c.midpoint().z()
			mask_xy = mask(x_m, y_m, z_m)

			if mask_xy > 1:
				self.cf[c] = self.OMEGA_FLT
			else:
				self.cf[c] = self.OMEGA_GND

		s = "    - done - "
		print_text(s, cls=self)

		self.set_measures()

	def set_measures(self):
		"""
		set the new measure space for facets ``self.ds`` and cells ``self.dx`` for
		the boundaries marked by FacetFunction ``self.ff`` and CellFunction
		``self.cf``, respectively.

		Also, the number of facets marked by
		:func:`calculate_boundaries` :

		* ``self.N_OMEGA_GND``   -- number of cells marked ``self.OMEGA_GND``
		* ``self.N_OMEGA_FLT``   -- number of cells marked ``self.OMEGA_FLT``
		* ``self.N_GAMMA_S_GND`` -- number of facets marked ``self.GAMMA_S_GND``
		* ``self.N_GAMMA_B_GND`` -- number of facets marked ``self.GAMMA_B_GND``
		* ``self.N_GAMMA_S_FLT`` -- number of facets marked ``self.GAMMA_S_FLT``
		* ``self.N_GAMMA_B_FLT`` -- number of facets marked ``self.GAMMA_B_FLT``
		* ``self.N_GAMMA_L_DVD`` -- number of facets marked ``self.GAMMA_L_DVD``
		* ``self.N_GAMMA_L_OVR`` -- number of facets marked ``self.GAMMA_L_OVR``
		* ``self.N_GAMMA_L_UDR`` -- number of facets marked ``self.GAMMA_L_UDR``
		* ``self.N_GAMMA_U_GND`` -- number of facets marked ``self.GAMMA_U_GND``
		* ``self.N_GAMMA_U_FLT`` -- number of facets marked ``self.GAMMA_U_FLT``

		The subdomains corresponding to FacetFunction ``self.ff`` are :

		* ``self.dOmega``     --  entire interior
		* ``self.dOmega_g``   --  internal above grounded
		* ``self.dOmega_w``   --  internal above floating

		The subdomains corresponding to CellFunction ``self.cf`` are :

		* ``self.dGamma_s``   -- entire upper surface
		* ``self.dGamma_b``   -- entire basal surface
		* ``self.dGamma_l``   -- entire exterior and interior lateral surface
		* ``self.dGamma_w``   -- exterior surface in contact with water
		* ``self.dGamma_bg``  -- grounded basal surface
		* ``self.dGamma_bw``  -- floating basal surface
		* ``self.dGamma_sgu`` -- upper surface with U observations above grounded ice
		* ``self.dGamma_swu`` -- upper surface with U observations above floating ice
		* ``self.dGamma_su``  -- entire upper surface with U observations
		* ``self.dGamma_sg``  -- upper surface above grounded ice
		* ``self.dGamma_sw``  -- upper surface above floating ice
		* ``self.dGamma_ld``  -- lateral interior surface
		* ``self.dGamma_lto`` -- exterior lateral surface above water
		* ``self.dGamma_ltu`` -- exterior lateral surface below water
		* ``self.dGamma_lt``  -- entire exterior lateral surface

		This method will create a :py:class:`dict` called ``self.measures``
		containing all of the :class:`ufl.measure.Measure` instances for
		this :class:`~model.Model`.
		"""
		# calculate the number of cells and facets that are of a certain type
		# for determining Dirichlet boundaries :
		local_N_OMEGA_GND   = sum(self.cf.array()     == self.OMEGA_GND)
		local_N_OMEGA_FLT   = sum(self.cf.array()     == self.OMEGA_FLT)
		local_N_GAMMA_S_GND = sum(self.ff.array()     == self.GAMMA_S_GND)
		local_N_GAMMA_B_GND = sum(self.ff.array()     == self.GAMMA_B_GND)
		local_N_GAMMA_S_FLT = sum(self.ff.array()     == self.GAMMA_S_FLT)
		local_N_GAMMA_B_FLT = sum(self.ff.array()     == self.GAMMA_B_FLT)
		local_N_GAMMA_L_DVD = sum(self.ff.array()     == self.GAMMA_L_DVD)
		local_N_GAMMA_L_OVR = sum(self.ff.array()     == self.GAMMA_L_OVR)
		local_N_GAMMA_L_UDR = sum(self.ff.array()     == self.GAMMA_L_UDR)
		local_N_GAMMA_U_GND = sum(self.ff.array()     == self.GAMMA_U_GND)
		local_N_GAMMA_U_FLT = sum(self.ff.array()     == self.GAMMA_U_FLT)
		local_N_GAMMA_ACC   = sum(self.ff_acc.array() == self.GAMMA_ACC)

		# find out if any are marked over all processes :
		self.N_OMEGA_GND   = MPI.sum(MPI.comm_world, local_N_OMEGA_GND)
		self.N_OMEGA_FLT   = MPI.sum(MPI.comm_world, local_N_OMEGA_FLT)
		self.N_GAMMA_S_GND = MPI.sum(MPI.comm_world, local_N_GAMMA_S_GND)
		self.N_GAMMA_B_GND = MPI.sum(MPI.comm_world, local_N_GAMMA_B_GND)
		self.N_GAMMA_S_FLT = MPI.sum(MPI.comm_world, local_N_GAMMA_S_FLT)
		self.N_GAMMA_B_FLT = MPI.sum(MPI.comm_world, local_N_GAMMA_B_FLT)
		self.N_GAMMA_L_DVD = MPI.sum(MPI.comm_world, local_N_GAMMA_L_DVD)
		self.N_GAMMA_L_OVR = MPI.sum(MPI.comm_world, local_N_GAMMA_L_OVR)
		self.N_GAMMA_L_UDR = MPI.sum(MPI.comm_world, local_N_GAMMA_L_UDR)
		self.N_GAMMA_U_GND = MPI.sum(MPI.comm_world, local_N_GAMMA_U_GND)
		self.N_GAMMA_U_FLT = MPI.sum(MPI.comm_world, local_N_GAMMA_U_FLT)
		self.N_GAMMA_ACC   = MPI.sum(MPI.comm_world, local_N_GAMMA_ACC)

		# create new measures of integration :
		self.ds      = Measure('ds', subdomain_data=self.ff)
		self.dx      = Measure('dx', subdomain_data=self.cf)

		# create subdomain boundaries :
		self.dOmega     = Boundary(self.dx, [self.OMEGA_GND, self.OMEGA_FLT],
		                  'entire interior')
		self.dOmega_g   = Boundary(self.dx, [self.OMEGA_GND],
		                  'interior above grounded ice')
		self.dOmega_w   = Boundary(self.dx, [self.OMEGA_FLT],
		                  'interior above floating ice')
		self.dGamma     = Boundary(self.ds, [self.GAMMA_S_GND, self.GAMMA_S_FLT,
		                                     self.GAMMA_B_GND, self.GAMMA_B_FLT,
		                                     self.GAMMA_L_OVR, self.GAMMA_L_UDR,
		                                     self.GAMMA_U_GND, self.GAMMA_U_FLT,
		                                     self.GAMMA_L_DVD],
		                  'entire exterior')
		self.dGamma_bg  = Boundary(self.ds, [self.GAMMA_B_GND],
		                  'grounded basal surface')
		self.dGamma_bw  = Boundary(self.ds, [self.GAMMA_B_FLT],
		                  'floating basal surface')
		self.dGamma_b   = Boundary(self.ds, [self.GAMMA_B_GND, self.GAMMA_B_FLT],
		                  'entire basal surface')
		self.dGamma_sgu = Boundary(self.ds, [self.GAMMA_U_GND],
		                  'upper surface with U observations above grounded ice')
		self.dGamma_swu = Boundary(self.ds, [self.GAMMA_U_FLT],
		                  'upper surface with U observations above floating ice')
		self.dGamma_su  = Boundary(self.ds, [self.GAMMA_U_GND, self.GAMMA_U_FLT],
		                  'entire upper surface with U observations')
		self.dGamma_sg  = Boundary(self.ds, [self.GAMMA_S_GND, self.GAMMA_U_GND],
		                  'upper surface above grounded ice')
		self.dGamma_sw  = Boundary(self.ds, [self.GAMMA_S_FLT, self.GAMMA_U_FLT],
		                  'upper surface above floating ice')
		self.dGamma_s   = Boundary(self.ds, [self.GAMMA_S_GND, self.GAMMA_S_FLT,
		                                     self.GAMMA_U_GND, self.GAMMA_U_FLT],
		                  'entire upper surface')
		self.dGamma_ld  = Boundary(self.ds, [self.GAMMA_L_DVD],
		                  'lateral interior surface')
		self.dGamma_lto = Boundary(self.ds, [self.GAMMA_L_OVR],
		                  'exterior lateral surface above water')
		self.dGamma_ltu = Boundary(self.ds, [self.GAMMA_L_UDR],
		                  'exterior lateral surface below= water')
		self.dGamma_lt  = Boundary(self.ds, [self.GAMMA_L_OVR, self.GAMMA_L_UDR],
		                  'entire exterior lateral surface')
		self.dGamma_l   = Boundary(self.ds, [self.GAMMA_L_OVR, self.GAMMA_L_UDR,
		                                     self.GAMMA_L_DVD],
		                  'entire exterior and interior lateral surface')
		self.dGamma_w   = Boundary(self.ds, [self.GAMMA_L_UDR, self.GAMMA_B_FLT],
		                  'exterior surface in contact with water')

		# save a dictionary of boundaries for later access by user :
		measures = [self.dOmega,
		            self.dOmega_g,
		            self.dOmega_w,
		            self.dGamma,
		            self.dGamma_bg,
		            self.dGamma_bw,
		            self.dGamma_b,
		            self.dGamma_sgu,
		            self.dGamma_swu,
		            self.dGamma_su,
		            self.dGamma_sg,
		            self.dGamma_sw,
		            self.dGamma_s,
		            self.dGamma_ld,
		            self.dGamma_lto,
		            self.dGamma_ltu,
		            self.dGamma_lt,
		            self.dGamma_l]
		self.measures = {}
		for m in measures:
			self.measures[m.description] = m


	def deform_mesh_to_geometry(self, S, B):
		"""
		Deforms the 3D mesh to the geometry from FEniCS Expressions for the
		surface <S> and bed <B>.
		"""
		s = "::: deforming mesh to geometry :::"
		print_text(s, cls=self)

		# initialize the topography functions :
		self.init_S(S)
		self.init_B(B)

		# convert constants to expressions that can be evaluated at (x,y,z) :
		if type(S) == Constant or type(S) == float or type(S) == int:
			S = Expression('S', S=S, degree=0)
		if type(B) == Constant or type(B) == float or type(B) == int:
			B = Expression('B', B=B, degree=0)

		# transform z :
		# thickness = surface - base, z = thickness + base
		# Get the height of the mesh, assumes that the base is at z=0
		max_height  = self.mesh.coordinates()[:,2].max()
		min_height  = self.mesh.coordinates()[:,2].min()
		mesh_height = max_height - min_height

		# sigma coordinate :
		self.init_sigma( project((self.x[2] - min_height) / mesh_height, self.Q1, \
		                         annotate=False) )

		s = "    - iterating through %i vertices - " % self.num_vertices
		print_text(s, cls=self)

		for x in self.mesh.coordinates():
			x[2] = (x[2] / mesh_height) * (S(x[0], x[1], x[2]) - B(x[0], x[1], x[2]))
			x[2] = x[2] + B(x[0], x[1], x[2])
		s = "    - done - "
		print_text(s, cls=self)

		# mark the exterior facets and interior cells appropriately :
		self.calculate_boundaries()

	def calc_thickness(self):
		"""
		Calculate the continuous thickness field which increases from 0 at the
		surface to the actual thickness at the bed.
		"""
		s = "::: calculating z-varying thickness :::"
		print_text(s, cls=self)
		#H = project(self.S - self.x[2], self.Q, annotate=False)
		H          = self.vert_integrate(Constant(1.0), d='down')
		Hv         = H.vector()
		Hv[Hv < 0] = 0.0
		print_min_max(H, 'H', cls=self)
		return H

	def solve_hydrostatic_pressure(self, annotate=False, cls=None):
		"""
		Solve for the hydrostatic pressure 'p'.
		"""
		if cls is None:
			cls = self
		# solve for vertical velocity :
		s  = "::: solving hydrostatic pressure :::"
		print_text(s, cls=cls)
		rho_i   = self.rho_i
		g      = self.g
		p      = self.vert_integrate(rho_i*g, d='down', annotate=annotate)
		pv     = p.vector()
		pv[pv < 0] = 0.0
		self.assign_variable(self.p, p, annotate=annotate)

	def vert_extrude(self, u, d='up', Q='self', annotate=False):
		r"""
		This extrudes a function *u* vertically in the direction *d* = 'up' or
		'down'.  It does this by solving a variational problem:

		.. math::

		   \frac{\partial v}{\partial z} = 0 \hspace{10mm}
		   v|_b = u

		"""
		s = "::: extruding function %swards :::" % d
		print_text(s, cls=self)
		if type(Q) != FunctionSpace:
			Q  = self.Q_non_periodic
			#Q = u.function_space()
		ff   = self.ff
		phi  = TestFunction(Q)
		v    = TrialFunction(Q)
		a    = v.dx(2) * phi * dx
		L    = DOLFIN_EPS * phi * dx
		bcs  = []
		# extrude bed (ff = 3,5)
		if d == 'up':
			if self.N_GAMMA_B_GND != 0:
				bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_GND))  # grounded
			if self.N_GAMMA_B_FLT != 0:
				bcs.append(DirichletBC(Q, u, ff, self.GAMMA_B_FLT))  # shelves
		# extrude surface (ff = 2,6)
		elif d == 'down':
			if self.N_GAMMA_S_GND != 0:
				bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_GND))  # grounded
			if self.N_GAMMA_S_FLT != 0:
				bcs.append(DirichletBC(Q, u, ff, self.GAMMA_S_FLT))  # shelves
			if self.N_GAMMA_U_GND != 0:
				bcs.append(DirichletBC(Q, u, ff, self.GAMMA_U_GND))  # grounded
			if self.N_GAMMA_U_FLT != 0:
				bcs.append(DirichletBC(Q, u, ff, self.GAMMA_U_FLT))  # shelves
		try:
			name = '%s extruded %s' % (u.name(), d)
		except AttributeError:
			name = 'extruded'
		v    = Function(Q, name=name)
		solve(a == L, v, bcs, annotate=annotate,
		      solver_parameters=self.linear_solve_params())
		print_min_max(v, 'extruded function')
		return v

	def vert_integrate(self, u, d='up', Q='self', annotate=False):
		"""
		Integrate <u> from the bed to the surface.
		"""
		s = "::: vertically integrating function :::"
		print_text(s, cls=self)

		if type(Q) != FunctionSpace:
			Q  = self.Q_non_periodic
			#Q = u.function_space()
		ff  = self.ff
		phi = TestFunction(Q)
		v   = TrialFunction(Q)
		bcs = []
		# integral is zero on bed (ff = 3,5)
		if d == 'up':
			if self.N_GAMMA_B_GND != 0:
				bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_GND))  # grounded
			if self.N_GAMMA_B_FLT != 0:
				bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_B_FLT))  # shelves
			a      = v.dx(2) * phi * dx
		# integral is zero on surface (ff = 2,6)
		elif d == 'down':
			if self.N_GAMMA_S_GND != 0:
				bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_GND))  # grounded
			if self.N_GAMMA_S_FLT != 0:
				bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_S_FLT))  # shelves
			if self.N_GAMMA_U_GND != 0:
				bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_GND))  # grounded
			if self.N_GAMMA_U_FLT != 0:
				bcs.append(DirichletBC(Q, 0.0, ff, self.GAMMA_U_FLT))  # shelves
			a      = -v.dx(2) * phi * dx
		L      = u * phi * dx
		name   = 'value integrated %s' % d
		v      = Function(Q, name=name)
		solve(a == L, v, bcs, annotate=annotate,
		      solver_parameters=self.linear_solve_params())
		print_min_max(v, 'vertically integrated function')
		return v

	def calc_vert_average(self, u, annotate=False):
		"""
		Calculates the vertical average of a given function *u*.

		:param u: Function to avergage vertically
		:rtype:   the vertical average of *u*
		"""
		s = "::: calculating vertical average :::"
		print_text(s, cls=self)

		# vertically integrate the function up and then extrude that down :
		ubar = self.vert_integrate(u,  d='up',   annotate=annotate)
		ubar = self.vert_extrude(ubar, d='down', annotate=annotate)

		try:
			name = 'vertical average of %s' % u.name()
		except AttributeError:
			name = 'vertical average'
		ubar.rename(name, '')

		# divide by the thickness for vertical average :
		ubar_v = ubar.vector().get_local()
		S_v    = self.S.vector().get_local()
		B_v    = self.B.vector().get_local()
		H_v    = S_v - B_v + DOLFIN_EPS
		self.assign_variable(ubar, ubar_v / H_v)
		return ubar

	def save_bed_mesh(self, h5File):
		"""
		save the basal boundary mesh to hdf5 file <h5File>.
		"""
		s = "::: writing 'bedmesh' to supplied hdf5 file :::"
		print_text(s, cls=self.this)
		h5File.write(self.bedmesh, 'bedmesh')

	def save_srf_mesh(self, h5File):
		"""
		save the surface boundary mesh to hdf5 file <h5File>.
		"""
		s = "::: writing 'srfmesh' to supplied hdf5 file :::"
		print_text(s, cls=self.this)
		h5File.write(self.srfmesh, 'srfmesh')

	def save_lat_mesh(self, h5File):
		"""
		save the lateral boundary mesh to hdf5 file <h5File>.
		"""
		s = "::: writing 'latmesh' to supplied hdf5 file :::"
		print_text(s, cls=self.this)
		h5File.write(self.latmesh, 'latmesh')

	def save_dvd_mesh(self, h5File):
		"""
		save the divide boundary mesh to hdf5 file <h5File>.
		"""
		s = "::: writing 'dvdmesh' to supplied hdf5 file :::"
		print_text(s, cls=self.this)
		h5File.write(self.dvdmesh, 'dvdmesh')

	def initialize_variables(self):
		"""
		Initializes the class's variables to default values that are then set
		by the individually created model.
		"""
		super(D3Model, self).initialize_variables()

		s = "::: initializing 3D variables :::"
		print_text(s, cls=self)

		# Depth below sea level :
		class Depth(UserExpression):
			def eval(self, values, x):
				values[0] = -min(0, x[2])
		self.D = Depth(element=self.Q.ufl_element())

		# only need one flat-mesh variable in order to transfer data between the
		# 3D mesh and 2D mesh :
		self.u_flat        = Function(self.Q_flat,  name='u_flat')
		self.u2_flat       = Function(self.Q2_flat, name='u2_flat')
		self.u3_flat       = Function(self.V_flat,  name='u3_flat')

		# age  :
		self.age           = Function(self.Q, name='age')
		self.a0            = Function(self.Q, name='a0')

		# mass :
		self.mhat          = Function(self.Q, name='mhat')  # mesh velocity

		# specical dof mapping for periodic spaces :
		if self.use_periodic:
			self.mhat_non = Function(self.Q_non_periodic)
			self.assmhat  = FunctionAssigner(self.mhat.function_space(),
			                                 self.Q_non_periodic)

		# surface climate :
		# TODO: re-implent this (it was taken out somewhere)
		self.precip        = Function(self.Q, name='precip')

	def update_mesh(self):
		"""
		This method will update the surface height ``self.S`` and vertices
		of deformed mesh ``self.mesh``.
		"""
		print_text("    - updating mesh -", cls=self)

		# update the mesh :
		sigma                        = self.sigma.compute_vertex_values()
		S                            = self.S.compute_vertex_values()
		B                            = self.B.compute_vertex_values()
		self.mesh.coordinates()[:,2] = sigma*(S - B) + B  # only the z coordinate

	def transient_iteration(self, momentum, mass, time_step, adaptive, annotate):
		r"""
		This function defines one interation of the transient solution, and is
		called by the function :func:`~model.transient_solve`.

		Currently, the dolfin-adjoint ``annotate`` is not supported.

		"""
		dt        = time_step
		mesh      = self.mesh
		dSdt      = self.dSdt

		def f(S):
			mass.model.assign_variable(mass.model.S, S)
			mass.solve_dSdt()
			return mass.model.dSdt.vector().get_local()

		# calculate velocity :
		momentum.solve()

		# update the mass model's velocity field :
		self.assign_to_submesh_variable(u = self.u,   u_sub = mass.model.u)

		# save the initial surface :
		S_0     = self.S.vector().get_local()

		# calculate new surface :
		#mass.solve()

		# impose the thickness limit and update the model's surface :
		S_a         = self.RK4(f, mass.model.S.vector().get_local(), dt)
		B_a         = mass.model.B.vector().get_local()
		thin        = (S_a - B_a) < mass.thklim
		S_a[thin]   = B_a[thin] + mass.thklim
		mass.model.assign_variable(mass.model.S, S_a)

		# update the 3D model's surface :
		self.assign_from_submesh_variable(u = self.S,  u_sub = mass.model.S)

		# save the new surface :
		S_1     = self.S.vector().get_local()

		# deform the mesh :
		self.update_mesh()

		# calculate mesh velocity :
		sigma_a   = self.sigma.vector().get_local()
		mhat_a    = (S_1 - S_0) / dt * sigma_a

		# set mesh velocity depending on periodicity due to the fact that
		# the topography S and B are always not periodic :
		if self.use_periodic:
			self.assign_variable(self.mhat_non, mhat_a)
			self.assmhat.assign(self.mhat, self.mhat_non, annotate=annotate)
		else:
			self.assign_variable(self.mhat, mhat_a)



