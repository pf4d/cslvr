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

# NASA basins dataset for Antarctica :
basins_shape  = 'http://icesat4.gsfc.nasa.gov/cryo_data/' + \
                'drainage_divides/Ant_Full_DrainageSystem_Polygons.txt'
basins_image  = 'http://icesat4.gsfc.nasa.gov/cryo_data/drainage_divides/' + \
                'Ant_ICESatDSMaps_Fig_1.jpg'
fldr    = 'basins'
download_file(basins_shape, home, fldr, 
              name='Ant_Grounded_DrainageSystem_Polygons.txt')
download_file(basins_image, home, fldr, 
              name='Ant_ICESatDSMaps_Fig_1_sm.png')



