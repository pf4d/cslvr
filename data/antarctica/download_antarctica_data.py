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


#===============================================================================
home = os.getcwd()

bm1  = "http://www.pangaea.de/Publications/LeBrocq_et_al_2010/ALBMAPv1.nc.zip"
fldr = "bedmap1"
download_file(bm1, home, fldr, extract=True)

bm2  = "https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip"
fldr = "bedmap2"
download_file(bm2, home, fldr, extract=True)

mea  = "ftp://sidads.colorado.edu/pub/DATASETS/" \
       + "nsidc0484_MEASURES_antarc_vel_V01/Antarctica_ice_velocity.nc"
fldr = "measures"
download_file(mea, home, fldr)



