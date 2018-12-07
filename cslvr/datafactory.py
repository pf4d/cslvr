from tifffile           import TiffFile
from numpy              import array, sqrt, shape, arange, meshgrid, loadtxt
from scipy.io           import loadmat, netcdf_file
from netCDF4            import Dataset
from scipy.interpolate  import griddata
from pyproj             import Proj, transform
from cslvr.inputoutput  import print_text
import inspect
import os
import sys

class DataFactory(object):
	"""
	This class contains several static methods that fetch and return raw data
	downloaded prevously by :func:`~helper.download_file`.  These data are
	located in the ``data/continent`` subdirectory of the CSLVR root directory,
	where ``continent`` is currently either ``antarctica`` or ``greenland``.

	Each one of the static methods defined here return a :py:class:`~dict` of
	parameters needed by the :class:`~inputoutput.DataInput` class for conversion
	to the FEniCS code used by CSLVR.

	* ``pyproj_Proj``        -- the geographical projection :class:`~pyproj.Proj` instance associated with this data
	* ``map_western_edge``   -- the Western-most edge of the data in projection coordinates
	* ``map_eastern_edge``   -- the Eastern-most edge of the data in projection coordinates
	* ``map_southern_edge``  -- the Southern-most edge of the data in projection coordinates
	* ``map_northern_edge``  -- the Northern-most edge of the data in projection coordinates
	* ``nx``                 -- the number of x divisions of the data
	* ``ny``                 -- the number of y divisions of the data
	* ``dataset``            -- the name of the dataset
	* ``continent``          -- the continent of the datset
	"""

	color = '229'
	global home
	filename = inspect.getframeinfo(inspect.currentframe()).filename
	home     = os.path.dirname(os.path.abspath(filename)) + '/../data'

	@staticmethod
	def print_dim(rg):

		for i in rg.variables.keys():

			dim = " ".join(rg.variables[i].dimensions)
			print i + "\n dimensons: " + dim

			if dim != "":
				print " Length of time: %d \n" % (len(rg.variables[i]), )
			else:
				print "\n"


	@staticmethod
	def get_ant_measures(res = 450):
		"""
		`Antarctica Measures <https://nsidc.org/data/docs/measures/nsidc0484_rignot/>`_ surface velocity data.  This function creates a new data field with
		key ``mask`` that is 1 where velocity measurements are present and
		0 where they are not.

		You will have to download the data and manually place it in the
		``cslvr_root_dir/data/antarctica`` directory.

		The keys of the dictionary returned by this function are :

		* ``vx``  -- :math:`x`-component of velocity
		* ``vy``  -- :math:`y`-component of velocity
		* ``v_err``  -- velocity error
		* ``mask`` -- observation mask

		:param res: resolution of the data, may be either 450 or 900
		:type res: int
		:rtype: dict
		"""

		s    = "::: getting Antarctica 'Measures' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		if res == 900:
			direc = home + '/antarctica/antarctica_ice_velocity_900m_v2.nc'
		elif res == 450:
			direc = home + '/antarctica/antarctica_ice_velocity_450m_v2.nc'
		else:
			print "get_ant_measures() 'res' arg must be either 900 or 450"
			exit(0)

		data     = Dataset(direc, mode = 'r')
		vara     = dict()

		# retrieve data :
		vx   = array(data.variables['VX'][:])
		vy   = array(data.variables['VY'][:])
		mask = (vx != 0.0).astype('i')

		names = ['vx', 'vy', 'mask']
		ftns  = [ vx,   vy,   mask ]

		for n in names:
			print_text('      Measures : %-*s key : "%s" '%(30,n,n), '230')

		# extents of domain :
		nx,ny =  shape(vx)
		dx    =  res
		west  = -2800000.0
		east  =  west + nx*dx
		north =  2800000.0
		south =  north - ny*dx

		#projection info :
		proj   = 'stere'
		lat_0  = '-90'
		lat_ts = '-71'
		lon_0  = '0'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		# save the data in matlab format :
		vara['dataset']   = 'measures'
		vara['continent'] = 'antarctica'
		for n, f in zip(names, ftns):
			vara[n] = f[::-1, :]
		return vara


	@staticmethod
	def get_bedmachine(thklim = 0.0):
		"""
		Greenland `Bedmachine <http://onlinelibrary.wiley.com/doi/10.1002/2017GL074954/full>`_ geometry.

		You will have to download the data and manually
		(from `here <https://nsidc.org/data/idbmg4>`_)
		and place it in the
		``cslvr_root_dir/data/greenland`` directory.

		The keys of the dictionary returned by this function are :

		* ``B``  -- basal topography height
		* ``S``  -- surface topography height
		* ``H``  -- ice thickness
		* ``Bo`` -- basal topography height before imposing ``thklim``
		* ``mask`` -- ice shelf mask (1 where shelves, 0 where grounded)
		* ``mask_orig`` -- original ice mask from the data

		:param thklim: minimum-allowed ice thickness
		:type thklim: float
		:rtype: dict
		"""
		s    = "::: getting Greenland 'Bedmachine v3' data from DataFactory :::"
		print_text(s, DataFactory.color)
		s    = "    - imposing thickness limit ``thklim = %g`` m to data -"
		print_text(s % thklim, DataFactory.color)

		global home

		direc = home + '/greenland/'
		filen = 'BedMachineGreenland-2017-09-20.nc'

		data  = Dataset(direc + filen, mode = 'r')
		vara  = dict()

		needed_vars = {'surface'   : 'S',
		               'bed'       : 'B',
		               'thickness' : 'H',
		               'mask'      : 'mask_orig'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Bedmachine : %-*s key : %s '%(30,v, txt), '230')

		# retrieve data :
		S          = array(data.variables['surface'][:])
		B          = array(data.variables['bed'][:])
		H          = array(data.variables['thickness'][:])
		mask_orig  = array(data.variables['mask'][:])

		try:
			data_new = Dataset(direc + 'bedmachine_cslvr.nc', 'r')
		except IOError:
			s    = "::: cslvr bedmachine data not present, calculating :::"
			print_text(s, 'red', 1)

			# create mask for use by cslvr :
			mask = mask_orig.copy(True)
			mask[mask_orig == 1] = 0
			mask[mask_orig == 2] = 1  # grounded ice
			mask[mask_orig == 3] = 2  # floating ice
			mask[mask_orig == 4] = 0  # non-Greenland land

			# create a new netcdf4 file :
			data_new  = Dataset(direc + 'bedmachine_cslvr.nc', mode = 'w',
			                    format='NETCDF4')

			data_new.description = "CSLVR data generated for use with bedmachine."

			# copy the dimensions :
			for name, dimension in data.dimensions.iteritems():
				data_new.createDimension(name,
				                    len(dimension) if not dimension.isunlimited()
				                                     else None)
			# create the variables :
			data_new.createVariable('x',        data.variables['x'].dtype,
			                                    data.variables['x'].dimensions)
			data_new.createVariable('y',        data.variables['y'].dtype,
			                                    data.variables['y'].dimensions)
			data_new.createVariable('mask',     data.variables['mask'].dtype,
			                                    data.variables['mask'].dimensions)

			# copy the attributes of the dimensions :
			for ncattr in data.variables['x'].ncattrs():
				data_new.setncattr(ncattr, data.variables['x'].getncattr(ncattr))
			for ncattr in data.variables['y'].ncattrs():
				data_new.setncattr(ncattr, data.variables['y'].getncattr(ncattr))
			for ncattr in data.variables['mask'].ncattrs():
				data_new.setncattr(ncattr, data.variables['mask'].getncattr(ncattr))

			# write the variables to the new netcdf :
			data_new.variables['x'][:]        =  data.variables['x'][:]
			data_new.variables['y'][:]        =  data.variables['y'][:]
			data_new.variables['mask'][:]     =  mask

		mask = array(data_new.variables['mask'][:])

		# remove the junk data and impose thickness limit :
		B   = B.copy(True)
		H[H == data.no_data] = 0.0
		S[H < thklim] = B[H < thklim] + thklim
		H[H < thklim] = thklim
		B             = S - H

		# extents of domain :
		nx    = int(data.ny)
		ny    = int(data.nx)
		dx    = data.spacing
		west  = data.xmin
		east  = west + nx*dx
		north = data.ymax
		south = north - ny*dx

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '70'
		lon_0  = '-45'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# close the datasets, we are done with them :
		data.close()
		data_new.close()

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		names = ['S', 'B', 'H', 'mask_orig', 'mask']
		ftns  = [ S,   B,   H,   mask_orig,   mask ]

		s = '      DataInput  : %-*s key : "%s"'
		print_text(s % (30,names[-1],names[-1]), '230')

		# save the data in matlab format :
		vara['dataset']   = 'Bedmachine v3'
		vara['continent'] = 'greenland'
		for n, f in zip(names, ftns):
			vara[n] = f[::-1, :]
		return vara


	@staticmethod
	def get_mouginot():
		"""
		`Greenland <http://www.mdpi.com/2072-4292/9/4/364>`_ velocity mosiac.
		This function creates a new data field with
		key ``mask`` that is 1 where velocity measurements are present and
		0 where they are not.

		You will have to contact the authors of that paper to get the data.
		When you download it from them, you can just put the .nc file named
		``Greenland_ice_speed_v2017.nc`` in the ``root_dir/data/greenland``
		directory.

		The keys of the dictionary returned by this function are :

		* ``vx``  -- :math:`x`-component of velocity
		* ``vy``  -- :math:`y`-component of velocity
		* ``mask`` -- observation mask

		:rtype: dict
		"""
		s    = "::: getting Greenland 'Mouginot' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		direc = home + '/greenland/'
		filen = 'Greenland_ice_speed_v2017.nc'

		data  = Dataset(direc + filen, mode = 'r')
		vara  = dict()

		needed_vars = {'VX'   : 'vx',
		               'VY'   : 'vx'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Mouginot   : %-*s key : %s '%(30,v, txt), '230')

		# retrieve data :
		vx         = array(data.variables['VY'][:])
		vy         = array(data.variables['VX'][:])
		mask       = (vx != 0.0).astype('i')

		# extents of domain :
		nx    = len(data.variables['x'][:])
		ny    = len(data.variables['y'][:])
		dx    = 150.0
		west  = data.variables['x'][:].min()
		east  = data.variables['x'][:].max()
		north = data.variables['y'][:].max()
		south = data.variables['y'][:].min()

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '70'
		lon_0  = '-45'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# close the datasets, we are done with them :
		data.close()

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		names = ['vx', 'vy', 'mask']
		ftns  = [ vx,   vy,   mask ]

		s = '      DataInput  : %-*s key : "%s"'
		print_text(s % (30,names[-1],names[-1]), '230')

		# save the data in matlab format :
		vara['dataset']   = 'Mouginot'
		vara['continent'] = 'greenland'
		for n, f in zip(names, ftns):
			vara[n] = f[::-1, :]
		return vara


	@staticmethod
	def get_gimp():
		"""
		`Greenland Ice Mapping Project <http://nsidc.org/data/NSIDC-0645/versions/1>`_
		surface topography data.

		You will have to download the data and manually place it in the
		``cslvr_root_dir/data/greenland`` directory.

		The keys of the dictionary returned by this function are :

		* ``S``  -- surface topography height

		:rtype: dict
		"""

		s    = "::: getting 'GIMP' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home
		direc    = home + '/greenland'

		S        = TiffFile(direc + 'gimpdem_90m_v1.1.tif').asarray()
		mask     = TiffFile(direc + 'GimpIceMask_90m_v1.1.tif').asarray()

		# remove ice-free areas :
		S[mask == 0] = 0

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '70'
		lon_0  = '-45'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		east,south = p(-55.7983, 59.1996)

		# extents of domain :
		ny,nx =  S.shape
		dx    =  90.0
		west  = east - nx*dx
		north = south + ny*dx

		# save the data in matlab format :
		vara                      = dict()
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		print_text('      GIMP : %-*s key : "%s"' % (30,'S',   'S'),    '230')
		#print_text('      GIMP : %-*s key : "%s"' % (30,'mask','mask'), '230')

		# retrieve data :
		vara['dataset']   = 'GIMP'
		vara['continent'] = 'greenland'
		vara['S'] = S[::-1, :]
		return vara


	@staticmethod
	def get_noel():
		"""
		Greenland `surface-mass balance <https://www.the-cryosphere.net/10/2361/2016/>`_ at 1km resolution averaged over the years (1958 -- 2015).

		You will have to download the data from the authors and manually place
		it in the ``cslvr_root_dir/data/greenland`` directory.

		The keys of the dictionary returned by this function are :

		* ``S``  -- surface topography height
		* ``mask`` -- ice shelf mask (1 where ice, 0 where no ice)
		* ``smb`` -- surface-mass balance in m.i.e. a^{-1}.

		:param thklim: minimum-allowed ice thickness
		:type thklim: float
		:rtype: dict
		"""
		s    = "::: getting Greenland 'Noel' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		direc = home + '/greenland/'
		filen = 'SMB_rec.1960-1989.BN_RACMO2.3p2_1km.YYmean.nc'

		data  = Dataset(direc + filen, mode = 'r')
		vara  = dict()

		needed_vars = {'SMB_rec' : 'smb'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Noeel : %-*s key : %s '%(30,v, txt), '230')

		filen = 'Icemask_Topo_Iceclasses_lon_lat_average_1km.nc'
		data2 = Dataset(direc + filen, mode = 'r')

		needed_vars = {'GrIS'       : 'mask',
		               'Topography' : 'S'}

		for v in data2.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Noeel : %-*s key : %s '%(30,v, txt), '230')

		# retrieve data :
		# convert from [mm w.e.] to [m i.e.] :
		smb  = array(data.variables['SMB_rec'][:][0]) / 1000.0 * 910.0/1000.0
		mask = array(data2.variables['GrIS'][:])
		S    = array(data2.variables['Topography'][:])
		lat  = array(data2.variables['LAT'][:])
		lon  = array(data2.variables['LON'][:])

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '70'
		lon_0  = '-45'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# NOTE: used this to calculate the values of 'west' and 'north' below :
		#x,y = p(lon, lat)

		# extents of domain :
		ny,nx = shape(S)
		dx    = 1000.0
		west  = -639456.29834742961
		east  = west + nx*dx
		north = -656095.58634326793
		south = north - ny*dx

		# close the datasets, we are done with them :
		data.close()
		data2.close()

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		names = ['S', 'mask', 'smb']
		ftns  = [ S,   mask,   smb ]

		# save the data in matlab format :
		vara['dataset']   = 'Noel'
		vara['continent'] = 'greenland'
		for n, f in zip(names, ftns):
			vara[n] = f#[::-1, :]
		return vara


	@staticmethod
	def get_gre_measures():
		"""
		`Greenland Measures <https://nsidc.org/data/NSIDC-0478/versions/2#>`_
		surface velocity data.  This function creates a new data field with
		key ``mask`` that is 1 where velocity measurements are present
		and 0 where they are not.

		You will have to download the data from the authors and manually place
		it in the ``cslvr_root_dir/data/greenland`` directory.

		The keys of the dictionary returned by this function are :

		* ``vx``  -- :math:`x`-component of velocity
		* ``vy``  -- :math:`y`-component of velocity
		* ``ex``  -- :math:`x`-component of velocity error
		* ``ey``  -- :math:`y`-component of velocity error
		* ``mask`` -- observation mask

		:rtype: dict
		"""

		s    = "::: getting Greenland 'Measures' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		#direc    = home + '/greenland/measures/greenland_vel_mosaic500_2008_2009'
		direc = home + '/greenland/greenland_vel_mosaic500_2016_2017_'
		#TODO: find a way to intelligently leave out the error if you don't want
		#      to download them:
		files    = ['mask', 'vx_v2', 'vy_v2']#, '_ex_v2', '_ey_v2']
		vara     = dict()

		d    = TiffFile(direc + 'vx_v2.tif')
		mask = (d.asarray() != -2e9).astype('i')

		ftns = [mask]
		for n in files[1:]:
			data    = TiffFile(direc + n + '.tif')
			ftns.append(data.asarray())
			print_text('      Measures : %-*s key : "%s" '%(30,n,n), '230')
		print_text('      Measures : %-*s key : "%s"'%(30,files[0],files[0]), '230')

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '70'
		lon_0  = '-45'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# extents of domain :
		ny,nx       =  shape(d.asarray())
		dx          = 500
		lon_min     = -75.0
		lon_max     = -14.0
		lat_min     =  60.0
		lat_max     =  83.0

		# old v1 values :
		# FIXME: no projection extents provided, only longitude ranges which
		#        do not mach the data.  What a fucking disappointment.
		west  = -645000.0
		east  =  west  + nx*dx
		south = -3370000.0
		north =  south + ny*dx

		# set up a dictionary for use with cslvr::DataInput class :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		# retrieve data :
		vara['dataset']   = 'measures'
		vara['continent'] = 'greenland'
		for f,n in zip(ftns, files):
			vara[n] = f[::-1, :]
		return vara


	@staticmethod
	def get_rignot():
		"""
		Greenland `Rignot <http://www.ess.uci.edu/group/erignot/data/ice-flow-greenland-international-polar-year-2008%E2%80%932009>`_ surface velocity data.
		This function creates a new data field with key ``mask`` that is 1 where
		velocity measurements are present and 0 where they are not.

		You will have to download the data from the authors and manually place
		it in the ``cslvr_root_dir/data/greenland`` directory.

		The keys of the dictionary returned by this function are :

		* ``vx``  -- :math:`x`-component of velocity
		* ``vy``  -- :math:`y`-component of velocity
		* ``v_err``  -- velocity error
		* ``mask`` -- observation mask

		:rtype: dict
		"""

		s    = "::: getting Greenland 'Rignot' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		direc = home + '/greenland/velocity_greenland_v4Aug2014.nc'
		data  = netcdf_file(direc, mode = 'r')
		vara  = dict()

		needed_vars = {'vx'  : 'vx',
		               'vy'  : 'vy',
		               'err' : 'v_err'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Rignot : %-*s key : %s '%(30,v, txt), '230')

		# retrieve data :
		vx   = array(data.variables['vx'][:])
		vy   = array(data.variables['vy'][:])
		err  = array(data.variables['err'][:])
		mask = (vx != 0.0).astype('i')

		# extents of domain :
		ny,nx =  shape(vx)
		dx    =  150
		west  = -638000.0
		east  =  west + nx*dx
		north = -657600.0
		south =  north - ny*dx

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '70'
		lon_0  = '-45'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		names = ['vx', 'vy', 'v_err', 'mask']
		ftns  = [ vx,   vy,   err,     mask ]

		print_text('      Rignot : %-*s key : "%s"'%(30,names[-1],names[-1]), '230')

		# save the data in matlab format :
		vara['dataset']   = 'Rignot'
		vara['continent'] = 'greenland'
		for n, f in zip(names, ftns):
			vara[n] = f[::-1, :]
		return vara


	@staticmethod
	def get_gre_qgeo_fox_maule():
		"""
		Greenland `Fox Maule <http://websrv.cs.umt.edu/isis/index.php/Greenland_Basal_Heat_Flux>`_ geothermal-heat-flux data.  This function converts the
		geothermal-heat flux data from J s\ :sup:`-1` m\ :sup:`-2` to that used
		by CSLVR, J a\ :sup:`-1` m\ :sup:`-2`.

		This data is downloaded by executing the script
		``cslvr_root/scripts/download_greenland_data.py``

		The keys of the dictionary returned by this function are :

		* ``q_geo`` -- geothermal-heat flux

		:rtype: dict
		"""

		global home

		direc = home + "/greenland/Greenland_heat_flux_5km.nc"
		data  = netcdf_file(direc, mode = 'r')
		vara  = dict()

		# retrieve data :
		x     = array(data.variables['x1'][:])
		y     = array(data.variables['y1'][:])
		q_geo = array(data.variables['bheatflx'][:][0]) * 60 * 60 * 24 * 365

		# extents of domain :
		east  = max(x)
		west  = min(x)
		north = max(y)
		south = min(y)

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '71'
		lon_0  = '-39'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = len(x)
		vara['ny']                = len(y)
		vara['dx']                = 5000.0

		vara['dataset']   = 'Fox Maule'
		vara['continent'] = 'greenland'
		vara['q_geo']     =  q_geo
		return vara


	@staticmethod
	def get_bedmap1(thklim = 0.0):
		"""
		Antarctica `Bedmap 1 <https://doi.pangaea.de/10.1594/PANGAEA.734145>`_
		data.  This function converts the geothermal-heat flux data from
		J s\ :sup:`-1` m\ :sup:`-2` to that used by CSLVR,
		J a\ :sup:`-1` m\ :sup:`-2`, and the surface temperature data from
		degrees Celsius to degrees Kelvin.  The parameter ``thklim`` sets the
		minimum allowed ice thickness.

		This data is downloaded by executing the script
		``cslvr_root/scripts/download_antarctica_data.py``

		The keys of the dictionary returned by this function are :

		* ``B``  -- basal topography
		* ``S``  -- surface topography
		* ``T``  -- surface temperature
		* ``acca`` -- accumulation/ablation function "a"
		* ``accr`` -- accumulation/ablation function "r"
		* ``ghffm`` -- Fox-Maule geothermal-heat flux
		* ``ghfsr`` -- Shapiro-Ritzwoller geothermal-heat flux

		:param thklim: minimum-allowed ice thickness
		:type thklim: float
		:rtype: dict
		"""

		s    = "::: getting 'Bedmap 1' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		direc = home + '/antarctica/ALBMAPv1.nc'
		data  = netcdf_file(direc, mode = 'r')
		vara  = dict()

		needed_vars = {'lsrf'       : 'B',
		               'usrf'       : 'S',
		               'temp'       : 'T',
		               'acca'       : 'acca',
		               'accr'       : 'accr',
		               'ghffm'      : 'ghffm',
		               'ghfsr'      : 'ghfsr'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Bedmap 1 : %-*s key : %s '%(30,v, txt), '230')


		# retrieve data :
		x       = array(data.variables['x1'][:])
		y       = array(data.variables['y1'][:])
		b       = array(data.variables['lsrf'][:])
		h       = array(data.variables['usrf'][:])
		S_ringa = array(data.variables['acca'][:])
		S_ringr = array(data.variables['accr'][:])
		mask    = array(data.variables['mask'][:])
		srfTemp = array(data.variables['temp'][:]) + 273.15
		q_geo_f = array(data.variables['ghffm'][:]) * 60 * 60 * 24 * 365 / 1000
		q_geo_s = array(data.variables['ghfsr'][:]) * 60 * 60 * 24 * 365 / 1000

		H             = h - b
		h[H < thklim] = b[H < thklim] + thklim
		H[H < thklim] = thklim

		S_ringa[S_ringa < -1000] = 0
		S_ringr[S_ringr < -1000] = 0

		names = ['B','S','H','acca','accr','ghffm','ghfsr','temp']
		ftns  = [b, h, H, S_ringa, S_ringr, q_geo_f, q_geo_s, srfTemp]

		# extents of domain :
		east    = max(x)
		west    = min(x)
		north   = max(y)
		south   = min(y)

		#projection info :
		proj   = 'stere'
		lat_0  = '-90'
		lat_ts = '-71'
		lon_0  = '0'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['dataset']           = 'bedmap 1'
		vara['continent']         = 'antarctica'
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = len(x)
		vara['ny']                = len(y)
		vara['dx']                = 5000.0
		for n, f in zip(names, ftns):
			vara[n] = f
		return vara


	@staticmethod
	def get_bedmap2(thklim = 0.0):
		"""
		Antarctica `Bedmap 2 <https://www.bas.ac.uk/project/bedmap-2/>`_
		topography data.

		This data is downloaded by executing the script
		``cslvr_root/scripts/download_antarctica_data.py``

		The keys of the dictionary returned by this function are :

		* ``B``  -- basal topography height
		* ``S``  -- surface topography height
		* ``H``  -- ice thickness
		* ``mask`` -- ice shelf mask
		* ``rock_mask`` -- rock outcrop mask
		* ``b_uncert`` -- basal-topography uncertainty
		* ``coverage`` -- is a binary grid showing the distribution of ice thickness data used in the grid of ice thickness
		* ``gl04c_WGS84`` -- gives the values (as floating point) used to convert from heights relative to WGS84 datum to heights relative to EIGEN-GL04C geoid (to convert back to WGS84, add this grid)

		:param thklim: minimum-allowed ice thickness
		:type thklim: float
		:rtype: dict
		"""

		s    = "::: getting 'Bedmap 2' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home
		direc    = home + '/antarctica/bedmap2/'

		B           = TiffFile(direc + 'bedmap2_bed.tif')
		S           = TiffFile(direc + 'bedmap2_surface.tif')
		H           = TiffFile(direc + 'bedmap2_thickness.tif')
		mask        = TiffFile(direc + 'bedmap2_icemask_grounded_and_shelves.tif')
		rock_mask   = TiffFile(direc + 'bedmap2_rockmask.tif')
		b_uncert    = TiffFile(direc + 'bedmap2_grounded_bed_uncertainty.tif')
		coverage    = TiffFile(direc + 'bedmap2_coverage.tif')
		gl04c_WGS84 = TiffFile(direc + 'gl04c_geiod_to_WGS84.tif')

		B           = B.asarray()
		S           = S.asarray()
		H           = H.asarray()
		mask        = mask.asarray()
		rock_mask   = rock_mask.asarray()
		b_uncert    = b_uncert.asarray()
		coverage    = coverage.asarray()
		gl04c_WGS84 = gl04c_WGS84.asarray()

		# format the mask for cslvr :
		mask[mask == 1]   = 2
		mask[mask == 0]   = 1
		mask[mask == 127] = 0

		# remove the junk data and impose thickness limit :
		B = S - H
		H[H == 32767]  = thklim
		H[H <= thklim] = thklim
		S = B + H

		vara        = dict()

		# extents of domain :
		nx    =  6667
		ny    =  6667
		dx    =  1000.0
		west  = -3333500.0
		east  =  3333500.0
		north =  3333500.0
		south = -3333500.0

		#projection info :
		proj   = 'stere'
		lat_0  = '-90'
		lat_ts = '-71'
		lon_0  = '0'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = nx
		vara['ny']                = ny
		vara['dx']                = dx

		names = ['B', 'S', 'H', 'mask', 'rock_mask', 'b_uncert',
		         'coverage', 'gl04c_WGS84']
		ftns  = [B, S, H, mask, rock_mask, b_uncert, coverage, gl04c_WGS84]

		for n in names:
			print_text('      Bedmap 2 : %-*s key : "%s" '%(30,n,n), '230')

		# retrieve data :
		vara['dataset']   = 'bedmap 2'
		vara['continent'] = 'antarctica'
		for n, f in zip(names, ftns):
			vara[n] = f[::-1, :]
		return vara


	@staticmethod
	def get_bamber(thklim = 0.0):
		"""
		Greenland `Bamber <https://nsidc.org/data/NSIDC-0092>`_ topography data.

		You will have to download the data from the authors and manually place
		it in the ``cslvr_root_dir/data/greenland`` directory.

		The keys of the dictionary returned by this function are :

		* ``B``  -- basal topography height
		* ``S``  -- surface topography height
		* ``H``  -- ice thickness
		* ``Herr``  -- ice thickness error
		* ``Bo`` -- basal topography height before imposing ``thklim``
		* ``mask`` -- ice shelf mask (1 where shelves, 0 where grounded)
		* ``mask_orig`` -- original ice mask from the data

		:param thklim: minimum-allowed ice thickness
		:type thklim: float
		:rtype: dict
		"""

		s    = "::: getting 'Bamber' topography data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		direc = home + '/greenland/Greenland_bedrock_topography_V2.nc'
		data  = netcdf_file(direc, mode = 'r')
		vara  = dict()

		needed_vars = {'BedrockElevation' : 'B',
		               'SurfaceElevation' : 'S',
		               'IceThickness'     : 'H',
		               'BedrockError'     : 'Herr',
		               'LandMask'         : 'mask_orig'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Bamber : %-*s key : %s '%(30,v, txt), '230')

		# retrieve data :
		x         = array(data.variables['projection_x_coordinate'][:])
		y         = array(data.variables['projection_y_coordinate'][:])
		Bo        = array(data.variables['BedrockElevation'][:])
		S         = array(data.variables['SurfaceElevation'][:])
		H         = array(data.variables['IceThickness'][:])
		Herr      = array(data.variables['BedrockError'][:])
		mask_orig = array(data.variables['LandMask'][:])

		# format the mask for cslvr :
		mask = mask_orig.copy(True)
		mask[mask == 1] = 0
		mask[mask == 2] = 1  # grounded ice
		mask[mask == 3] = 0
		mask[mask == 4] = 2  # ice shelves

		# remove the junk data and impose thickness limit :
		B   = Bo.copy(True)
		H[H == -9999.0] = 0.0
		S[H < thklim] = B[H < thklim] + thklim
		H[H < thklim] = thklim
		B             = S - H

		# extents of domain :
		east  = max(x)
		west  = min(x)
		north = max(y)
		south = min(y)

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '71'
		lon_0  = '-39'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = len(x)
		vara['ny']                = len(y)
		vara['dx']                = 1000.0

		names = ['B', 'Bo', 'S', 'H', 'Herr', 'mask', 'mask_orig']
		ftns  = [ B,   Bo,   S,   H,   Herr,   mask,   mask_orig]

		# save the data in matlab format :
		vara['dataset']   = 'Bamber'
		vara['continent'] = 'greenland'
		for n, f in zip(names, ftns):
			vara[n] = f
		return vara


	@staticmethod
	def get_searise(thklim = 0.0):
		"""
		Greenland `Searise <http://websrv.cs.umt.edu/isis/index.php/Present_Day_Greenland>`_ data.

		This function converts the geothermal-heat flux data from J s\ :sup:`-1` m\ :sup:`-2` to that used by CSLVR, J a\ :sup:`-1` m\ :sup:`-2`, and the surface temperature data from degrees Celsius to degrees Kelvin.

		This data is downloaded by executing the script
		``cslvr_root/scripts/download_greenland_data.py``

		The keys of the dictionary returned by this function are :

		* ``B`` -- basal surface height
		* ``S``  -- upper surface height
		* ``T`` -- surface temperature
		* ``lat`` -- grid latitude
		* ``lon`` -- grid longitude
		* ``S_ring``  -- accumulation/ablation function
		* ``q_geo`` -- geothermal-heat flux
		* ``dhdt`` -- suface height rate of change
		* ``U_sar`` -- surface velocity magnitude

		:rtype: dict
		"""

		s    = "::: getting 'Searise' data from DataFactory :::"
		print_text(s, DataFactory.color)

		global home

		direc = home + "/greenland/Greenland_5km_dev1.2.nc"
		data  = netcdf_file(direc, mode = 'r')
		vara  = dict()

		needed_vars = {'topg'       : 'B',
		               'usrf'       : 'S',
		               'surftemp'   : 'T',
		               'lat'        : 'lat',
		               'lon'        : 'lon',
		               'smb'        : 'S_ring',
		               'bheatflx'   : 'q_geo',
		               'dhdt'       : 'dhdt',
		               'surfvelmag' : 'U_sar'}

		s    = "    - data-fields collected : python dict key to access -"
		print_text(s, DataFactory.color)
		for v in data.variables:
			try:
				txt = '"' + needed_vars[v] + '"'
			except KeyError:
				txt = ''
			print_text('      Searise : %-*s key : %s '%(30,v, txt), '230')


		# retrieve data :
		x      = array(data.variables['x1'][:])
		y      = array(data.variables['y1'][:])
		S      = array(data.variables['usrf'][:][0])
		S_ring = array(data.variables['smb'][:][0])
		B      = array(data.variables['topg'][:][0])
		T      = array(data.variables['surftemp'][:][0]) + 273.15
		q_geo  = array(data.variables['bheatflx'][:][0]) * 60 * 60 * 24 * 365
		lat    = array(data.variables['lat'][:][0])
		lon    = array(data.variables['lon'][:][0])
		U_sar  = array(data.variables['surfvelmag'][:][0])
		dhdt   = array(data.variables['dhdt'][:][0])

		H             = S - B
		S[H < thklim] = B[H < thklim] + thklim

		# extents of domain :
		east  = max(x)
		west  = min(x)
		north = max(y)
		south = min(y)

		#projection info :
		proj   = 'stere'
		lat_0  = '90'
		lat_ts = '71'
		lon_0  = '-39'

		# create projection :
		txt  =   " +proj="   + proj \
		       + " +lat_0="  + lat_0 \
		       + " +lat_ts=" + lat_ts \
		       + " +lon_0="  + lon_0 \
		       + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
		       + " +towgs84=0.000,0.000,0.000 +to_meter=1"
		p    = Proj(txt)

		# save the data in matlab format :
		vara['pyproj_Proj']       = p
		vara['map_western_edge']  = west
		vara['map_eastern_edge']  = east
		vara['map_southern_edge'] = south
		vara['map_northern_edge'] = north
		vara['nx']                = len(x)
		vara['ny']                = len(y)
		vara['dx']                = 5000.0

		names = ['S', 'S_ring', 'B', 'T', 'q_geo','U_sar', \
		         'lat', 'lon', 'dhdt']
		ftns  = [S, S_ring, B, T, q_geo,U_sar, lat, lon, dhdt]

		vara['dataset']   = 'Searise'
		vara['continent'] = 'greenland'
		for n, f in zip(names, ftns):
			vara[n] = f
		return vara



