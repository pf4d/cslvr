import sys
import os
src_directory = '../'
sys.path.append(src_directory)

from src.helper import download_file


home = os.getcwd()


# coarse greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/evtzuxhupnwlght/greenland_coarse_mesh.tar.gz'
download_file(url, home, fldr, extract=True)

# medium greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/zckpt9uqn0qqyl1/greenland_medium_mesh.tar.gz'
download_file(url, home, fldr, extract=True)

# detailed greenland mesh :
fldr = 'greenland'
url  = 'https://www.dropbox.com/s/hz2ih2qpfpm5htb/' + \
       'greenland_detailed_mesh.tar.gz'
download_file(url, home, fldr, extract=True)

# antarctica mesh :
fldr = 'antarctica'
url  = 'https://www.dropbox.com/s/n99zjixyzo2c2i8/antarctica.tar.gz'
download_file(url, home, fldr, extract=True)

# circle mesh :
fldr = 'test'
url  = 'https://www.dropbox.com/s/hk63j3l9fhx6zty/circle.tar.gz'
download_file(url, home, fldr, extract=True)




