from cslvr import download_file
import os
import inspect


def convert_measures_projection(direc, var):
	"""
	convert the measures .tif files to _new.tif files with the projection we
	require using gdalwarp.
	"""
	proj    = '\"+units=m  +proj=stere +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 ' \
	          + '+x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563 ' \
	          + '+towgs84=0.000,0.000,0.000 +to_meter=1\"'
	te      = '-645000.0 -3370000.0 860000.0 -640000.0'
	infile  = direc + '/greenland_vel_mosaic500_2008_2009_' + var + '.tif'
	outfile = direc + '/greenland_vel_mosaic500_2008_2009_' + var + '_new.tif'

	cmd = 'gdalwarp ' + infile + ' ' + outfile + ' -t_srs ' + proj + ' -te ' + te

	print "\nExecuting :\n\n\t", cmd, "\n\n"
	os.system(cmd)


#===============================================================================
filename = inspect.getframeinfo(inspect.currentframe()).filename
home     = os.path.dirname(os.path.abspath(filename)) + '/../data/antarctica/'

# Fox Maule et al. (2005) basal heat flux :
q_geo = 'http://websrv.cs.umt.edu/isis/images/c/c8/Antarctica_heat_flux_5km.nc'
fldr  = ''
download_file(q_geo, home, fldr)

# bedmap 1 :
bm1  = "https://www.dropbox.com/s/tqcdbe3d2chq9py/ALBMAPv1.tar.gz?dl=1"
fldr = ''
download_file(bm1, home, fldr, extract=True)

# bedmap 2 :
bm2  = "https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip"
fldr = 'bedmap2'
download_file(bm2, home, fldr, extract=True)

# NASA basins dataset for Antarctica :
basins_shape  = 'http://icesat4.gsfc.nasa.gov/cryo_data/' + \
                'drainage_divides/Ant_Full_DrainageSystem_Polygons.txt'
basins_image  = 'http://icesat4.gsfc.nasa.gov/cryo_data/drainage_divides/' + \
                'Ant_ICESatDSMaps_Fig_1.jpg'
fldr    = ''
download_file(basins_shape, home, fldr)
download_file(basins_image, home, fldr)



