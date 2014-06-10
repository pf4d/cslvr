import sys
import os
src_directory = '../'
sys.path.append(src_directory)

from src.helper import download_file


home = os.getcwd()


# coarse greenland mesh :
fldr = 'greenland'
url  = 'https://dl.dropboxusercontent.com/s/evtzuxhupnwlght/' + \
       'greenland_coarse_mesh.tar.gz?dl=1&token_hash=AAEFc43' + \
       '8YYCRbQAiEN1eToNlSB4kIUak4jL6sRncZMnNrg'
download_file(url, home, fldr, extract=True)

# medium greenland mesh :
fldr = 'greenland'
url  = 'https://dl.dropboxusercontent.com/s/zckpt9uqn0qqyl1/' + \
       'greenland_medium_mesh.tar.gz?dl=1&token_hash=AAHvYKB' + \
       'bNjM-U07GaqP3vJTN_H45Nd1eGJxucmhTEuRrDg'
download_file(url, home, fldr, extract=True)

# detailed greenland mesh :
fldr = 'greenland'
url  = 'https://dl.dropboxusercontent.com/s/hz2ih2qpfpm5htb/' + \
       'greenland_detailed_mesh.tar.gz?dl=1&token_hash=AAGtX' + \
       'NVrgZiNsiJ_ixg5wxXKPscwH4xFzDQP387tbGQc9w&expiry=1400783674'
download_file(url, home, fldr, extract=True)

# 1 ice-thickness 2D greenland mesh :
fldr = 'greenland'
url  = 'https://dl.dropboxusercontent.com/s/2wkfzkha7jtjyxq/' + \
       'greenland_2D_1H_mesh.tar.gz?dl=1&token_hash=AAFTj-T2' + \
       '4idxuG7Jv5CYlhSVxnGNLcjZoj_IiObdkHDmPQ&expiry=1400783547'
download_file(url, home, fldr, extract=True)

# antarctica mesh :
fldr = 'antarctica'
url  = 'https://dl.dropboxusercontent.com/s/n99zjixyzo2c2i8/' + \
       'antarctica.tar.gz?dl=1&token_hash=AAGFYWbn7p4JOywM4G' + \
       'NzNbUhCPhM4oGfg0KV7HX0ACXN8w'
download_file(url, home, fldr, extract=True)

# circle mesh :
fldr = 'test'
url  = 'https://dl.dropboxusercontent.com/s/hk63j3l9fhx6zty/' + \
       'circle.tar.gz?dl=1&token_hash=AAFGUYbuRVk3A56uAxImZT' + \
       'DiADzs6Du7Xi9WiOp_yTW7Ng'
download_file(url, home, fldr, extract=True)




