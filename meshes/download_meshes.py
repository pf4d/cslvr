import sys
import os
src_directory = '../'
sys.path.append(src_directory)

from src.helper import download_file


home = os.getcwd()


# coarse greenland mesh :
fldr = 'greenland'
url  = 'http://ubuntuone.com/6ZPTI3cSfaTyVCVkRPFQTi'
download_file(url, home, fldr, extract=True)

# medium greenland mesh :
fldr = 'greenland'
url  = 'http://ubuntuone.com/3PlD2d6ApRquWmGXRBHQIJ'
download_file(url, home, fldr, extract=True)

# detailed greenland mesh :
fldr = 'greenland'
url  = 'http://ubuntuone.com/55UQDSre2O8iZhnPLxCDuh'
download_file(url, home, fldr, extract=True)

# antarctica mesh :
fldr = 'antarctica'
url  = 'http://ubuntuone.com/5X1xcYHLcChAFFCceLkBVO'
download_file(url, home, fldr, extract=True)

# circle mesh :
fldr = 'test'
url  = 'http://ubuntuone.com/08uKNvX3ap2pvzHPD4vVv6'
download_file(url, home, fldr, extract=True)




