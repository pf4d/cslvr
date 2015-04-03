import inspect
import os
import sys
from numpy             import array, sqrt, shape, arange, meshgrid, loadtxt
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
    elif res == 450:
      direc    = home + '/antarctica/measures/antarctica_ice_velocity_450m.nc' 
    else:
      print "get_ant_measures() 'res' arg must be either 900 or 450"
      exit(0)

    data     = netcdf_file(direc, mode = 'r')
    vara     = dict()
  
    # retrieve data :
    vx   = array(data.variables['vx'][:])
    vy   = array(data.variables['vy'][:])
    err  = array(data.variables['err'][:])
    vmag = sqrt(vx**2 + vy**2)
    
    names = ['vx', 'vy', 'v_err', 'U_ob']
    ftns  = [vx, vy, err, vmag]
     
    # extents of domain :
    nx,ny =  shape(vx)
    dx    =  res
    west  = -2800000.0
    east  =  west + nx*dx
    north =  2800000.0
    south =  north - ny*dx

    #projection info :
    proj   = 'stere'
    lat_0  = '-90'
    lat_ts = '-71'
    lon_0  = '0'
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = nx
    vara['ny']                = ny
    
    # save the data in matlab format :
    vara['dataset']   = 'measures'
    vara['continent'] = 'antarctica'
    for n, f in zip(names, ftns):
      vara[n] = f[::-1, :]
    return vara
  
  
  @staticmethod
  def get_gre_measures():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    
    from tifffile.tifffile import TiffFile
    
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = nx
    vara['ny']                = ny
    
    # retrieve data :
    vara['dataset']   = 'measures'
    vara['continent'] = 'greenland'
    for f in files:
      data    = TiffFile(direc + f + '.tif')
      vara[f] = data.asarray()[::-1, :]
    return vara
 
  
  @staticmethod
  def get_gre_gimp():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    
    from tifffile.tifffile import TiffFile
    vara     = dict()
     
    # extents of domain :
    nx    =  16620
    ny    =  30000
    dx    =  90
    west  = -935727.9959405478
    east  =  west  + nx*dx
    south = -3698069.596373792
    north =  south + ny*dx

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '70'
    lon_0  = '-45'
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = nx
    vara['ny']                = ny
    
    # retrieve data :
    data              = TiffFile(home + '/greenland/gimp/GimpIceMask_90m.tif')
    vara['dataset']   = 'gimp'
    vara['continent'] = 'greenland'
    vara['mask']      = data.asarray()[::-1, :]
    return vara
 
  
  @staticmethod
  def get_gre_rignot():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
 
    direc = home + '/greenland/rignot/velocity_greenland_v4Aug2014.nc'
    data  = netcdf_file(direc, mode = 'r')
    vara  = dict()
    
    # retrieve data :
    vx   = array(data.variables['vx'][:])
    vy   = array(data.variables['vy'][:])
    err  = array(data.variables['err'][:])
    vmag = sqrt(vx**2 + vy**2)
     
    # extents of domain :
    ny,nx =  shape(vx)
    dx    =  150
    west  = -638000.0
    east  =  west + nx*dx
    north = -657600.0
    south =  north - ny*dx

    #projection info :
    proj   = 'stere'
    lat_0  = '90'
    lat_ts = '70'
    lon_0  = '-45'
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = nx
    vara['ny']                = ny
    
    names = ['vx', 'vy', 'v_err', 'U_ob']
    ftns  = [ vx,   vy,   err,     vmag]
    
    # save the data in matlab format :
    vara['dataset']   = 'Rignot'
    vara['continent'] = 'greenland'
    for n, f in zip(names, ftns):
      vara[n] = f[::-1, :]
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
 
    vara['dataset']   = 'Fox Maule'
    vara['continent'] = 'greenland'
    vara['q_geo']     =  q_geo
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

    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    x,y  = p(lon, lat)
    
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)
    xs    = arange(west,  east,  10000)
    ys    = arange(south, north, 10000)
    X, Y  = meshgrid(xs, ys)
    q_geo = griddata((x, y), q_geo, (X, Y), fill_value=0.0)
    
    vara['dataset']           = 'secret'
    vara['continent']         = 'greenland'
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
    vara['q_geo']             = q_geo
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
 
    vara['dataset']   = 'Fox Maule'
    vara['continent'] = 'antarctica'
    vara['q_geo']     = q_geo
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    x,y  = p(lon, lat)
    
    # extents of domain :
    east  = max(x)
    west  = min(x)
    north = max(y)
    south = min(y)
    xs    = arange(west,  east,  10000)
    ys    = arange(south, north, 10000)
    X, Y  = meshgrid(xs, ys)
    q_geo = griddata((x, y), q_geo, (X, Y), fill_value=0.0)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
    
    vara['dataset']   = 'secret'
    vara['continent'] = 'greenland'
    vara['q_geo']     = q_geo
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
    adota   = array(data.variables['acca'][:])
    adotr   = array(data.variables['accr'][:])
    mask    = array(data.variables['mask'][:])
    srfTemp = array(data.variables['temp'][:]) + 273.15
    q_geo_f = array(data.variables['ghffm'][:]) * 60 * 60 * 24 * 365 / 1000
    q_geo_s = array(data.variables['ghfsr'][:]) * 60 * 60 * 24 * 365 / 1000

    H             = h - b
    h[H < thklim] = b[H < thklim] + thklim
    H[H < thklim] = thklim
    
    names = ['B','S','H','acca','accr','ghffm','ghfsr','temp']
    ftns  = [b, h, H, adota, adotr, q_geo_f, q_geo_s, srfTemp]
    
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['dataset']           = 'bedmap 1'
    vara['continent']         = 'antarctica'
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
    for n, f in zip(names, ftns):
      vara[n] = f
    return vara 
  
  
  @staticmethod
  def get_bedmap2(thklim = 0.0):

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))
    direc    = home + '/antarctica/bedmap2/bedmap2_tiff/' 

    from tifffile.tifffile import TiffFile
   
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
    
    b = h - H
    H[H == 0.0] = thklim
    h = b + H

    vara        = dict()
     
    # extents of domain :
    nx    =  6667
    ny    =  6667
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = nx
    vara['ny']                = ny
    
    names = ['B', 'S', 'H', 'mask', 'rock_mask', 'b_uncert', 
             'coverage', 'gl04c_WGS84']
    ftns  = [b, h, H, mask, rock_mask, b_uncert, coverage, gl04c_WGS84]
   
    # retrieve data :
    vara['dataset']   = 'bedmap 2'
    vara['continent'] = 'antarctica'
    for n, f in zip(names, ftns):
      vara[n] = f[::-1, :]
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
    B    = array(data.variables['BedrockElevation'][:])
    S    = array(data.variables['SurfaceElevation'][:])
    H    = array(data.variables['IceThickness'][:])
    Herr = array(data.variables['BedrockError'][:])
    mask = array(data.variables['IceShelfSourceMask'][:])

    S[H < thklim] = B[H < thklim] + thklim
    H[H < thklim] = thklim
    B             = S - H

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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
     
    names = ['B', 'S', 'H', 'Herr', 'mask']
    ftns  = [ B,   S,   H,   Herr,   mask]
    
    # save the data in matlab format :
    vara['dataset']   = 'Bamber'
    vara['continent'] = 'greenland'
    for n, f in zip(names, ftns):
      vara[n] = f
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
    
    # create projection :
    txt  =   " +proj="   + proj \
           + " +lat_0="  + lat_0 \
           + " +lat_ts=" + lat_ts \
           + " +lon_0="  + lon_0 \
           + " +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563" \
           + " +towgs84=0.000,0.000,0.000 +to_meter=1"
    p    = Proj(txt)
    
    # save the data in matlab format :
    vara['pyproj_Proj']       = p
    vara['map_western_edge']  = west 
    vara['map_eastern_edge']  = east 
    vara['map_southern_edge'] = south 
    vara['map_northern_edge'] = north
    vara['nx']                = len(x)
    vara['ny']                = len(y)
 
    names = ['H', 'S', 'adot', 'B', 'T', 'q_geo','U_sar', \
             'U_ob', 'lat', 'lon', 'Tn','dhdt']
    ftns  = [H, h, adot, b, T, q_geo,U_sar, U_ob, lat, lon, Tn, dhdt]

    vara['dataset']   = 'searise'
    vara['continent'] = 'greenland'
    for n, f in zip(names, ftns):
      vara[n] = f
    return vara
 


