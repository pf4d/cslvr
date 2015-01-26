import urllib2
import sys
import os
import zipfile
import tarfile
import inspect

def download_file(url, direc, folder, extract=False):
  """
  download a file with url <url> into directory <direc>/<folder>.  If <extract>
  is True, extract the .zip file into the directory and delete the .zip file.
  """
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
  
  # extract the zip/tar.gz file if necessary :
  if extract:
    ty = fn.split('.')[-1]
    if ty == 'zip':
      cf = zipfile.ZipFile(direc + fn)
    else:
      cf = tarfile.open(direc + fn, 'r:gz')
    cf.extractall(direc)
    os.remove(direc + fn)

#===============================================================================
filename = inspect.getframeinfo(inspect.currentframe()).filename
home     = os.path.dirname(os.path.abspath(filename))

# Fox Maule et al. (2005) basal heat flux :
q_geo = 'http://websrv.cs.umt.edu/isis/images/c/c8/Antarctica_heat_flux_5km.nc'
fldr  = 'fox_maule'
download_file(q_geo, home, fldr)

# bedmap 1 :
bm1  = "https://www.dropbox.com/s/tqcdbe3d2chq9py/ALBMAPv1.tar.gz?dl=1"
fldr = "bedmap1"
download_file(bm1, home, fldr, extract=True)

# bedmap 2 :
bm2  = "https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip"
fldr = "bedmap2"
download_file(bm2, home, fldr, extract=True)

## measures velocity :
#mea  = 'ftp://sidads.colorado.edu/pub/DATASETS/' + \
#       'nsidc0484_MEASURES_antarc_vel_V01/450m/antarctica_ice_velocity_450m.nc'
#fldr = "measures"
#download_file(mea, home, fldr)

mea  = 'ftp://sidads.colorado.edu/pub/DATASETS/' + \
       'nsidc0484_MEASURES_antarc_vel_V01/900m/antarctica_ice_velocity.nc'
fldr = "measures"
download_file(mea, home, fldr)


