import inspect
import os
import sys
from numpy    import *
from scipy.io import loadmat, netcdf_file
from osgeo    import gdal

class DataFactory(object):
 
  @staticmethod 
  def print_dim(rg):
  
    for i in rg.variables.keys():
    
      dim = " ".join(rg.variables[i].dimensions)
      print i + "\n dimensons: " + dim
      
      if dim != "":
        print " Length of time: %d \n" % (len(rg.variables[i]), )
      else:
        print "\n"
  
  
  @staticmethod
  def get_ant_measures():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc    = home + '/antarctica/measures/Antarctica_ice_velocity_450m.nc' 
    data     = netcdf_file(direc, mode = 'r')
    vara     = dict()
  
    # retrieve data :
    vx   = array(data.variables['vx'][:])
    vy   = array(data.variables['vy'][:])
    err  = array(data.variables['err'][:])
    vmag = sqrt(vx**2 + vy**2)
     
    # extents of domain :
    m,n   =  shape(vx)
    dx    =  450
    west  = -2800000.0
    east  =  west + n*dx
    north =  2800000.0
    south =  north - m*dx

    #projection info :
    proj   = 'stere'
    lat_0  = '-90'
    lat_ts = '-71'
    lon_0  = '0'
    
    names = ['vx', 'vy', 'v_err', 'v_mag']
    ftns  = [vx, vy, err, vmag]
    
    # save the data in matlab format :
    for n, f in zip(names, ftns):
      vara[n] = {'map_data'          : f,
                 'map_western_edge'  : west, 
                 'map_eastern_edge'  : east, 
                 'map_southern_edge' : south, 
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara
  
  
  @staticmethod
  def get_gre_measures():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    
    sys.path.append(home + '/external_import_scripts')
    from tifffile import TiffFile
    
    direc    = home + '/greenland/measures/greenland_vel_mosaic500_2008_2009_' 
    files    = ['sp', 'vx', 'vy', 'ex', 'ey']
    vara     = dict()
     
    # extents of domain :
    nx    =  3010
    ny    =  5460
    dx    =  500
    west  = -645000.0
    east  =  west  + nx*dx
    south = -3370000.0 
    north =  south + ny*dx

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '70'
    lon_0  = '-45'
    
    # retrieve data :
    for f in files:
      data    = TiffFile(direc + f + '.tif')
      vara[f] = {'map_data'          : data.asarray(),
                 'map_western_edge'  : west,
                 'map_eastern_edge'  : east,  
                 'map_southern_edge' : south,
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara
  
  
  @staticmethod
  def get_shift_gre_measures():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    
    sys.path.append(home + '/external_import_scripts')
    from tifffile import TiffFile
    
    direc    = home + '/greenland/measures/greenland_vel_mosaic500_2008_2009_' 
    files    = ['sp', 'vx', 'vy']
    vara     = dict()
     
    # extents of domain :
    nx    =  3010
    ny    =  5460
    dx    =  500
    west  = -645000.0
    east  =  west  + nx*dx
    south = -3370000.0 
    north =  south + ny*dx

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '71'
    lon_0  = '-39'

    # retrieve data :
    for f in files:
      data    = TiffFile(direc + f + '_new.tif')
      vara[f] = {'map_data'          : data.asarray(),
                 'map_western_edge'  : west,
                 'map_eastern_edge'  : east,  
                 'map_southern_edge' : south,
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara
 
  
  @staticmethod
  def get_gre_qgeo_fox_maule():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + "/greenland/fox_maule/Greenland_heat_flux_5km.nc"
    data  = netcdf_file(direc, mode = 'r')
    vara  = dict()
    
    # retrieve data :
    x     = array(data.variables['x1'][:])
    y     = array(data.variables['y1'][:])
    q_geo = array(data.variables['bheatflx'][:][0]) * 60 * 60 * 24 * 365
 
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '71'
    lon_0  = '-39'
 
    vara['bheatflx'] = {'map_data'          : q_geo,
                        'map_western_edge'  : west, 
                        'map_eastern_edge'  : east, 
                        'map_southern_edge' : south, 
                        'map_northern_edge' : north,
                        'projection'        : proj,
                        'standard lat'      : lat_0,
                        'standard lon'      : lon_0,
                        'lat true scale'    : lat_ts}
    return vara
    
  
  @staticmethod
  def get_ant_qgeo_fox_maule():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + "/antarctica/fox_maule/Antarctica_heat_flux_5km.nc"
    data  = netcdf_file(direc, mode = 'r')
    vara  = dict()
    
    # retrieve data :
    x     = array(data.variables['x1'][:])
    y     = array(data.variables['y1'][:])
    q_geo = array(data.variables['bheatflx'][:][0]) * 60 * 60 * 24 * 365
 
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)

    #projection info :
    proj   = 'stere'
    lat_0  = '-90'
    lat_ts = '71'
    lon_0  = '0'
 
    vara['bheatflx'] = {'map_data'          : q_geo,
                        'map_western_edge'  : west, 
                        'map_eastern_edge'  : east, 
                        'map_southern_edge' : south, 
                        'map_northern_edge' : north,
                        'projection'        : proj,
                        'standard lat'      : lat_0,
                        'standard lon'      : lon_0,
                        'lat true scale'    : lat_ts}
    return vara
  

  @staticmethod
  def get_lebrocq():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + '/antarctica/bedmap1/ALBMAPv1.nc'
    data  = netcdf_file(direc, mode = 'r')
    vara  = dict()
    
    # retrieve data :
    x       = array(data.variables['x1'][:])
    y       = array(data.variables['y1'][:])
    b       = array(data.variables['lsrf'][:])
    h       = array(data.variables['usrf'][:])
    adot    = array(data.variables['acca'][:])
    mask    = array(data.variables['mask'][:])
    srfTemp = array(data.variables['temp'][:])
    q_geo   = array(data.variables['ghffm'][:])
    H       = h - b
    
    # extents of domain :
    east    = max(x)
    west    = min(x)
    north   = max(y)
    south   = min(y)

    #projection info :
    proj   = 'stere'
    lat_0  = '-90'
    lat_ts = '-71'
    lon_0  = '0'
    
    names = ['b', 'h', 'H', 'adot', 'q_geo', 'srfTemp']
    ftns  = [b, h, H, adot, q_geo, srfTemp]
    
    # save the data in matlab format :
    for n, f in zip(names, ftns):
      vara[n] = {'map_data'          : f,
                 'map_western_edge'  : west, 
                 'map_eastern_edge'  : east, 
                 'map_southern_edge' : south, 
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara 
  
  
  @staticmethod
  def get_bamber(thklim = 10.0):
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
   
    direc = home + '/greenland/bamber13/Greenland_bedrock_topography_V2.nc' 
    data  = netcdf_file(direc, mode = 'r')
    vara  = dict()
    
    # retrieve data :
    x    = array(data.variables['projection_x_coordinate'][:])
    y    = array(data.variables['projection_y_coordinate'][:])
    b    = array(data.variables['BedrockElevation'][:])
    h    = array(data.variables['SurfaceElevation'][:])
    H    = array(data.variables['IceThickness'][:])
    mask = array(data.variables['IceShelfSourceMask'][:])
    
    H_n               = h - b
    h_n               = h.copy()
    h_n[H_n < thklim] = b[H_n < thklim] + thklim

    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '71'
    lon_0  = '-39'
     
    names = ['b', 'h', 'h_n', 'H', 'H_n', 'mask']
    ftns  = [ b,   h,   h_n,   H,   H_n,   mask]
    
    # save the data in matlab format :
    for n, f in zip(names, ftns):
      vara[n] = {'map_data'          : f,
                 'map_western_edge'  : west, 
                 'map_eastern_edge'  : east, 
                 'map_southern_edge' : south, 
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara 
  
  
  @staticmethod
  def get_searise(thklim = 10.0):
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + "/greenland/searise/Greenland_5km_dev1.2.nc"
    data  = netcdf_file(direc, mode = 'r')
    vara  = dict()
    
    # retrieve data :
    x     = array(data.variables['x1'][:])
    y     = array(data.variables['y1'][:])
    h     = array(data.variables['usrf'][:][0])
    adot  = array(data.variables['smb'][:][0])
    b     = array(data.variables['topg'][:][0])
    T     = array(data.variables['surftemp'][:][0]) + 273.15
    q_geo = array(data.variables['bheatflx'][:][0]) * 60 * 60 * 24 * 365
    lat   = array(data.variables['lat'][:][0])
    lon   = array(data.variables['lon'][:][0])
    U_sar = array(data.variables['surfvelmag'][:][0])
 
    direc = home + "/greenland/searise/smooth_target.mat" 
    U_ob  = loadmat(direc)['st']
    
    H             = h - b
    h[H < thklim] = b[H < thklim] + thklim

    Tn            = 41.83 - 6.309e-3*h - 0.7189*lat - 0.0672*lon + 273
    
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '71'
    lon_0  = '-39'
 
    names = ['H', 'h', 'adot', 'b', 'T', 'q_geo','U_sar', \
             'U_ob', 'lat', 'lon', 'Tn']
    ftns  = [H, h, adot, b, T, q_geo,U_sar, U_ob, lat, lon, Tn]

    for n, f in zip(names, ftns):
      vara[n] = {'map_data'          : f,
                 'map_western_edge'  : west, 
                 'map_eastern_edge'  : east, 
                 'map_southern_edge' : south, 
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara
 
  
  @staticmethod
  def get_bedmap2():

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    
    direc    = home + '/antarctica/bedmap2/bedmap2_tiff/' 
    files    = ['bedmap2_bed',
                'bedmap2_surface', 
                'bedmap2_thickness',
                'bedmap2_icemask_grounded_and_shelves', 
                'bedmap2_rockmask', 
                #'bedmap2_lakemask_vostok', 
                'bedmap2_grounded_bed_uncertainty', 
                #'bedmap2_thickness_uncertainty_5km', 
                'bedmap2_coverage',
                'gl04c_geiod_to_WGS84']
    vara     = dict()
     
    # extents of domain :
    dx    =  1000
    west  = -3333500.0
    east  =  3333500.0
    north =  3333500.0
    south = -3333500.0

    #projection info :
    proj   = 'stere'
    lat_0  = '-90'
    lat_ts = '-71'
    lon_0  = '0'
    
    names = ['b', 'h', 'H', 'mask', 'rock_mask', 'b_uncert', 
             'coverage', 'gl04c_to_WGS84']
   

    sys.path.append(home + '/external_import_scripts')
    from tifffile import TiffFile
    # retrieve data :
    for n, f in zip(names, files):
      data    = TiffFile(direc + f + '.tif')
      vara[n] = {'map_data'          : data.asarray(),
                 'map_western_edge'  : west,
                 'map_eastern_edge'  : east,  
                 'map_southern_edge' : south,
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara 



