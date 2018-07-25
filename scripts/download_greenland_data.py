from cslvr import download_file
import inspect
import os

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
home     = os.path.dirname(os.path.abspath(filename)) + '/../data/greenland'

#NOTE: measures is no longer available without registration, you'll have to 
#      go to NSDIC's website and download it, then place it in the
#      relative directory '../data/greenland/measures/'
#
## measures velocity dataset :
#fldr = 'measures'
#meas = 'ftp://n5eil01u.ecs.nsidc.org/SAN/MEASURES/NSIDC-0478.001/' + \
#       '2008.12.01/greenland_vel_mosaic500_2008_2009'
#download_file(meas + '_vx.tif', home, fldr)
#download_file(meas + '_vy.tif', home, fldr)
#download_file(meas + '.ex.tif', home, fldr)
#download_file(meas + '.ey.tif', home, fldr)

## convert to searise projection via raster warp :
#convert_measures_projection(home + '/' + fldr, 'vx')
#convert_measures_projection(home + '/' + fldr, 'vy')
#convert_measures_projection(home + '/' + fldr, 'ex')
#convert_measures_projection(home + '/' + fldr, 'ey')
#convert_measures_projection(home + '/' + fldr, 'sp')

# Fox Maule et al. (2005) basal heat flux :
q_geo   = 'http://websrv.cs.umt.edu/isis/images/d/da/Greenland_heat_flux_5km.nc'
fldr    = ''
download_file(q_geo, home, fldr)

# searise dataset :
searise = 'http://websrv.cs.umt.edu/isis/images/e/e9/Greenland_5km_dev1.2.nc'
fldr    = ''
download_file(searise, home, fldr)

# NASA basins dataset for Greenland:
basins_shape  = 'http://icesat4.gsfc.nasa.gov/cryo_data/' + \
                'drainage_divides/GrnDrainageSystems_Ekholm.txt'
basins_image  = 'http://icesat4.gsfc.nasa.gov/cryo_data/' + \
                'drainage_divides/Grn_Drainage_Systems_sm.png'
fldr    = ''
download_file(basins_shape, home, fldr)
download_file(basins_image, home, fldr)



