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
  fn   = fn.split('?')[0]
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


filename = inspect.getframeinfo(inspect.currentframe()).filename
home     = os.path.dirname(os.path.abspath(filename)) + '/mesh'

# coarse greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/evtzuxhupnwlght/greenland' + \
       '_coarse_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# medium greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/zckpt9uqn0qqyl1/greenland' + \
       '_medium_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# detailed greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/hz2ih2qpfpm5htb/greenland' + \
       '_detailed_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 3D 5H greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/yiof92z34ccjbbe/greenland' + \
       '_3D_5H_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 2D 1H greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/2wkfzkha7jtjyxq/' + \
       'greenland_2D_1H_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 2D 5H greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/pyclc3p1caxx85n/greenland_' + \
       '2D_5H_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 2D 5H sealevel greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/plkc5gxk61e06ay/' + \
       'greenland_2D_5H_sealevel.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# coarse, no-shelf antarctica mesh :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/n99zjixyzo2c2i8/antarctica.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 50H, shelf, antarctica mesh :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/yjkpimym57d6inl/antarctica' + \
       '_3D_50H_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 100H, shelf, antarctica mesh :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/20dovwv32v7clde/antarctica' + \
       '_3D_100H_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# gradS, shelf, antarctica mesh crude :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/jk0nctbxo9gafqb/antarctica' + \
       '_3D_gradS_mesh_crude.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# gradS, shelf, antarctica mesh detailed :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/h36739k2dd4ht5d/antarctica' + \
       '_3D_gradS_mesh_detailed.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# 10km, antarctica mesh :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/g46veuwbmwb354t/antarctica' + \
       '_3D_10k.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# medium 2D antarctica mesh :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/bvdbi6c1eiobr9h' + \
       '/antarctica_2D_medium_mesh.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# Ronne shelf, antarctica mesh 10H :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/2xwauazntbob22v/antarctica' + \
       '_ronne_shelf.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# Ronne shelf, antarctica mesh 50H :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/xxn2cdc3y4cwihv/antarctica' + \
       '_ronne_shelf_crude.tar.gz?dl=1'
download_file(url, home, fldr, extract=True)

# circle mesh :
fldr = 'test'
url  = 'https://www.dropbox.com/s/cj2cwkz22tzqtqh/circle_mesh.xml.gz?dl=1'
download_file(url, home, fldr, extract=False)




