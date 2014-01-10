import inspect
import os
import sys
from numpy             import *
from scipy.io          import loadmat, netcdf_file
from scipy.interpolate import griddata
from osgeo             import gdal
from pyproj            import Proj, transform

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
  def get_ant_measures(res = 900):
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    if res == 900:
      direc    = home + '/antarctica/measures/antarctica_ice_velocity.nc' 
    else:
      direc    = home + '/antarctica/measures/antarctica_ice_velocity_450m.nc' 

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
      vara[n] = {'map_data'          : f[::-1, :],
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
      vara[f] = {'map_data'          : data.asarray()[::-1, :],
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
      vara[f] = {'map_data'          : data.asarray()[::-1, :],
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
 
    vara['q_geo'] = {'map_data'          : q_geo,
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
  def get_gre_qgeo_secret():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + "/greenland/secret_qgeo/ghf_10x10_2000_JVJ.dat"
    data  = loadtxt(direc)
    vara  = dict()
    
    # retrieve data :
    lon   = data[:,2]
    lat   = data[:,3]
    q_geo = data[:,4] * 60 * 60 * 24 * 365

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '71'
    lon_0  = '-39'
    proj_s =   " +proj="   + proj \
             + " +lat_0="  + lat_0 \
             + " +lat_ts=" + lat_ts \
             + " +lon_0="  + lon_0 \
             + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
             + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p      = Proj(proj_s)
    x, y   = p(lon, lat)
    
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)
    xs    = arange(west,  east,  10000)
    ys    = arange(south, north, 10000)
    X, Y  = meshgrid(xs, ys)
    q_geo = griddata((x, y), q_geo, (X, Y), fill_value=0.0)
    
    vara['q_geo'] = {'map_data'          : q_geo,
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
 
    vara['q_geo'] = {'map_data'          : q_geo,
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
  def get_gre_qgeo_secret():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + "/greenland/secret_qgeo/ghf_10x10_2000_JVJ.dat"
    data  = loadtxt(direc)
    vara  = dict()
    
    # retrieve data :
    lon   = data[:,2]
    lat   = data[:,3]
    q_geo = data[:,4] * 60 * 60 * 24 * 365 / 1000

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '71'
    lon_0  = '-39'
    proj_s =   " +proj="   + proj \
             + " +lat_0="  + lat_0 \
             + " +lat_ts=" + lat_ts \
             + " +lon_0="  + lon_0 \
             + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
             + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p      = Proj(proj_s)
    x, y   = p(lon, lat)
    
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)
    xs    = arange(west,  east,  10000)
    ys    = arange(south, north, 10000)
    X, Y  = meshgrid(xs, ys)
    q_geo = griddata((x, y), q_geo, (X, Y), fill_value=0.0)
    
    vara['q_geo'] = {'map_data'          : q_geo,
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
 
    vara['q_geo'] = {'map_data'          : q_geo,
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
  def get_bedmap1(thklim = 0.0):
    
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
    srfTemp = array(data.variables['temp'][:]) + 273.15
    q_geo   = array(data.variables['ghffm'][:]) * 60 * 60 * 24 * 365
   
    H             = h - b
    h[H < thklim] = b[H < thklim] + thklim
    H[H < thklim] = thklim
    
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
    ftns  = [ b,   h,   H,   adot,   q_geo,   srfTemp]
    
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
  def get_bedmap2(thklim = 0.0):

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    direc    = home + '/antarctica/bedmap2/bedmap2_tiff/' 

    sys.path.append(home + '/external_import_scripts')
    from tifffile import TiffFile
   
    b           = TiffFile(direc + 'bedmap2_bed.tif')
    h           = TiffFile(direc + 'bedmap2_surface.tif') 
    H           = TiffFile(direc + 'bedmap2_thickness.tif')
    mask        = TiffFile(direc + 'bedmap2_icemask_grounded_and_shelves.tif') 
    rock_mask   = TiffFile(direc + 'bedmap2_rockmask.tif') 
    b_uncert    = TiffFile(direc + 'bedmap2_grounded_bed_uncertainty.tif') 
    coverage    = TiffFile(direc + 'bedmap2_coverage.tif')
    gl04c_WGS84 = TiffFile(direc + 'gl04c_geiod_to_WGS84.tif')
   
    b           = b.asarray()
    h           = h.asarray()
    H           = H.asarray()
    mask        = mask.asarray()
    rock_mask   = rock_mask.asarray()
    b_uncert    = b_uncert.asarray()
    coverage    = coverage.asarray() 
    gl04c_WGS84 = gl04c_WGS84.asarray()
    
    h[H < thklim] = b[H < thklim] + thklim
    H[H < thklim] = thklim

    vara        = dict()
     
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
             'coverage', 'gl04c_WGS84']
    ftns  = [b, h, H, mask, rock_mask, b_uncert, coverage, gl04c_WGS84]
   
    # retrieve data :
    for n, f in zip(names, ftns):
      vara[n] = {'map_data'          : f[::-1, :],
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
  def get_bamber(thklim = 0.0):
    
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
    Herr = array(data.variables['BedrockError'][:])
    mask = array(data.variables['IceShelfSourceMask'][:])
    
    h[H < thklim] = b[H < thklim] + thklim
    H[H < thklim] = thklim

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
     
    names = ['b', 'h', 'H', 'Herr', 'H_n', 'mask']
    ftns  = [ b,   h,   H,   Herr,   H_n,   mask]
    
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
  def get_searise(thklim = 0.0):
    
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
    dhdt  = array(data.variables['dhdt'][:][0])
 
    direc = home + "/greenland/searise/smooth_target.mat" 
    U_ob  = loadmat(direc)['st']
    
    H             = h - b
    h[H < thklim] = b[H < thklim] + thklim
    H[H < thklim] = thklim

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
             'U_ob', 'lat', 'lon', 'Tn','dhdt']
    ftns  = [H, h, adot, b, T, q_geo,U_sar, U_ob, lat, lon, Tn, dhdt]

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
 


