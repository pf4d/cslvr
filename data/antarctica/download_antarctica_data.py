import sys
import os
src_directory = '../../'
sys.path.append(src_directory)

from src.helper import download_file


home = os.getcwd()

# Fox Maule et al. (2005) basal heat flux :
q_geo = 'http://websrv.cs.umt.edu/isis/images/c/c8/Antarctica_heat_flux_5km.nc'
fldr  = 'fox_maule'
download_file(q_geo, home, fldr)

# bedmap 1 :
bm1  = "http://www.pangaea.de/Publications/LeBrocq_et_al_2010/ALBMAPv1.nc.zip"
fldr = "bedmap1"
download_file(bm1, home, fldr, extract=True)

# bedmap 2 :
bm2  = "https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip"
fldr = "bedmap2"
download_file(bm2, home, fldr, extract=True)

# measures velocity :
mea  = 'ftp://sidads.colorado.edu/pub/DATASETS/' + \
       'nsidc0484_MEASURES_antarc_vel_V01/450m/antarctica_ice_velocity_450m.nc'
fldr = "measures"
download_file(mea, home, fldr)



