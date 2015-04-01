import urllib2
import sys
import os
import zipfile
import tarfile
import inspect

def download_file(url, direc, folder, extract=False, name = None):
  """
  download a file with url <url> into directory <direc>/<folder>.  If <extract>
  is True, extract the .zip file into the directory and delete the .zip file.
  If name is specified, the file will be changed to the string passed as name.
  """
  # make the directory if needed :
  direc = direc + '/' + folder + '/'
  d     = os.path.dirname(direc)
  if not os.path.exists(d):
    os.makedirs(d)

  # url file info :
  fn   = url.split('/')[-1]

  # Only download files that don't exist. 
  # This will allow a restart of a terminated download
  if name:
      fn = name

  EXISTS = os.path.isfile(direc+fn)
    
  if not EXISTS:
    u    = urllib2.urlopen(url)
    f    = open(direc + fn, 'wb')
    meta = u.info()
    fs   = int(meta.getheaders("Content-Length")[0])
    
    s    = "Downloading: %s Bytes: %s" % (fn, fs)
    print s
    
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
  else:
    print "WARNING: "+fn+" already downloaded. \nDelete file if it is was" +\
          " partial download"

  # extract the zip/tar.gz file if necessary :
  if extract and not EXISTS:
    ty = fn.split('.')[-1]
    if ty == 'zip':
      cf = zipfile.ZipFile(direc + fn)
    else:
      cf = tarfile.open(direc + fn, 'r:gz')
    cf.extractall(direc)
    os.remove(direc + fn)

  if name and not EXISTS:
    os.rename(direc+fn,direc+name)
  
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
home     = os.path.dirname(os.path.abspath(filename))

# measures velocity dataset :
fldr = 'measures'
meas = 'ftp://sidads.colorado.edu/pub/DATASETS/' + \
       'nsidc0478_MEASURES_greenland_V01/2008/' + \
       'greenland_vel_mosaic500_2008_2009'
download_file(meas + '_sp.tif', home, fldr)
download_file(meas + '_vx.tif', home, fldr)
download_file(meas + '_vy.tif', home, fldr)

# get errors in .tif format
errors  = 'https://dl.dropboxusercontent.com/s/65q1wtc8mofotnz/' +\
          'greenland_measures_error.tar.gz?dl=1&token_hash=AAEp' +\
          '-BIqyJTzkCTmUg-1hAtfU0iZYDqlkww5Oo5qgc0mTQ'
download_file(errors, home, fldr, extract=True)

## convert to searise projection via raster warp :
#convert_measures_projection(home + '/' + fldr, 'vx')
#convert_measures_projection(home + '/' + fldr, 'vy')
#convert_measures_projection(home + '/' + fldr, 'ex')
#convert_measures_projection(home + '/' + fldr, 'ey')
#convert_measures_projection(home + '/' + fldr, 'sp')

# Fox Maule et al. (2005) basal heat flux :
q_geo   = 'http://websrv.cs.umt.edu/isis/images/d/da/Greenland_heat_flux_5km.nc'
fldr    = 'fox_maule'
download_file(q_geo, home, fldr)

# searise dataset :
searise = 'http://websrv.cs.umt.edu/isis/images/e/e9/Greenland_5km_dev1.2.nc'
fldr    = 'searise'
download_file(searise, home, fldr)

# NASA basins dataset for Greenland:
basins_shape  = 'https://umt.box.com/shared/static/s18gkw5k7ldozg0l9kfreh3hsxk0flpq.txt'
basins_image  = 'https://umt.box.com/shared/static/pgo62nopbwid6hkuc5suyvggjozvywqv.png'
fldr    = 'basins'
download_file(basins_shape, home, fldr ,name = 'GrnDrainageSystems_Ekholm.txt')
download_file(basins_image, home, fldr ,name = 'Grn_Drainage_Systems.png')

# smooth target matlab matrix :
smooth  = 'https://dl.dropboxusercontent.com/s/e8r0x37mil03hvu/' +\
          'smooth_target.tar.gz?dl=1&token_hash=AAGafyrXdL72vZL' +\
          'ffoBX3_kfAcEzvFhvrw8rERNx2WQShA'
fldr    = 'searise'
download_file(smooth, home, fldr, extract=True)

# Bamber 2013 bedrock topography dataset :
v2      = 'https://dl.dropboxusercontent.com/s/qd02y99d1xrkdz3/' + \
          'Greenland_bedrock_topography_V2.tar.gz?dl=1&token_ha' + \
          'sh=AAFzWa8fuvcKC2tBYqY9VzDLctRWwqX2EuN-179bJ74XEg'
fldr    = 'bamber13'
download_file(v2, home, fldr, extract=True)

# updated Rignot 2014 velocity data :
merged  = 'https://www.dropbox.com/s/ov5bl30jsojws8g/velocity_' + \
          'greenland_v4Aug2014.tar.gz?dl=1'
fldr    = 'rignot'
download_file(merged, home, fldr, extract=True)

# gimp ice-mask :
mask    = 'ftp://ftp.bpcrc.osu.edu/downloads/gdg/gimpicemask/' + \
          'GimpIceMask_90m.tif'
fldr    = 'gimp'
download_file(mask, home, fldr, extract=False)

