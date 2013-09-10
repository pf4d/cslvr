import sys
import os
src_directory = '../'
sys.path.append(src_directory)

from src.helper import download_file


home = os.getcwd()


# detailed greenland mesh :
fldr = 'greenland'
url  = 'http://ubuntuone.com/36EhUdGvmSphJwmsDarn1i'
download_file(url, home, fldr, extract=True)

# medium greenland mesh :
fldr = 'greenland'
url  = 'http://ubuntuone.com/5M4H8znzPmHm9QMHF0Epa3'
download_file(url, home, fldr, extract=True)

# coarse greenland mesh :
fldr = 'greenland'
url  = 'http://ubuntuone.com/6ZPTI3cSfaTyVCVkRPFQTi'
download_file(url, home, fldr, extract=True)

# antarctica mesh :
fldr = 'antarctica'
url  = 'http://ubuntuone.com/5X1xcYHLcChAFFCceLkBVO'
download_file(url, home, fldr, extract=True)

# circle mesh :
fldr = 'test'
url  = 'http://ubuntuone.com/08uKNvX3ap2pvzHPD4vVv6'
download_file(url, home, fldr, extract=True)




