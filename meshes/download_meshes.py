import sys
import os
src_directory = '../'
sys.path.append(src_directory)

from src.helper import download_file


home = os.getcwd()


fldr  = 'test'
url   = 'https://dl.dropboxusercontent.com/u/55658/test.tar.gz'

download_file(url, home, fldr, extract=True)

#fldr = 'greenland'
#url  = 'ftp://.../greenland_detailed_mesh.tar.gz'
#
#download_file(url, home, fldr)
#
#
#fldr = 'greenland'
#url  = 'ftp://.../greenland_coarse_mesh.tar.gz'
#
#download_file(url, home, fldr)
#
#
#fldr = 'antarctica'
#url  = 'ftp://.../antarctica_mesh.tar.gz'
#
#download_file(url, home, fldr)
#
#fldr = 'test'
#url  = 'ftp://.../circle_mesh.tar.gz'
#
#download_file(url, home, fldr)




