import urllib2
import sys
import os
import zipfile

def download_file(url, direc, folder, extract=False):
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
  
  print "Downloading: %s Bytes: %s" % (fn, fs)
  
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
  
  # extract the zip file if necessary :
  if extract:
    zf = zipfile.ZipFile(direc + fn)
    zf.extractall(direc)
    os.remove(direc + fn)


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
home = os.getcwd()

fldr = 'measures'
meas = 'ftp://sidads.colorado.edu/pub/DATASETS/' + \
       'nsidc0478_MEASURES_greenland_V01/2008/' + \
       'greenland_vel_mosaic500_2008_2009'

download_file(meas + '_sp.tif', home, fldr)
download_file(meas + '_vx.tif', home, fldr)
download_file(meas + '_vy.tif', home, fldr)

convert_measures_projection(home + '/' + fldr, 'vx')
convert_measures_projection(home + '/' + fldr, 'vy')
convert_measures_projection(home + '/' + fldr, 'sp')

searise = 'http://websrv.cs.umt.edu/isis/images/e/e9/Greenland_5km_dev1.2.nc'
fldr    = 'searise'

download_file(searise, home, fldr)




