from mpl_toolkits.basemap import Basemap
from pylab                import *
from matplotlib           import colors
from pyproj               import Proj

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

def plotIce(u, direc, cmap, scale='lin', name='', units='', proj_in=None, 
            numLvls=12, tp=False, tpAlpha=0.5, plot_type='tripcolor'):
  """
  INPUTS :
    u :
      solution to plot.
    direc :
      directory relative to the src directory to save image.
    cmap :
      colormap to use - see images directory for sample and name
    scale :
      scale to plot, either 'log' or 'lin'
    name :
      title of the plot, latex accepted
    units :
      units for the colorbar, latex approved
    numLvls :
      number of levels for field values
    proj_in : 
      string - if 'ant', set up for Antarctica, else Greenland.
    tp :
      boolean determins plotting of triangle overlay
    tpAlpha :
      alpha level of triangles 0.0 (transparent) - 1.0 (opaque)
    plot_type :
      type of plot : 'tripcolor', 'contourf', or 'gouraud'.
  
  OUTPUT :
    A sigle 250 dpi .png in the source directory.
  
  """
  def parse_proj4(p_in):
    p_out = {}
  
    l = ['proj','lat_0','lat_ts','lon_0']
    for item in l:
      start = p_in.find(item) + 1 + len(item)
      end   = p_in.find('+',start)
      p_out[item] = p_in[start:end-1]
    return p_out

  #=============================================================================
  # format stuff that needs to be formatted :  
  dec = 1
 
  field = '' 
  dec = 1

  # data gathering :
  mesh  = u.function_space().mesh()
  coord = mesh.coordinates()
  fi    = mesh.cells()
  v     = u.compute_vertex_values(mesh)
  vx    = coord[:,0]
  vy    = coord[:,1]
  
  vmin = min(v)
  vmax = max(v)

  #=============================================================================
  # map functions :
  fig = plt.figure(figsize=(10,8), dpi=90)
  ax  = fig.add_axes()
  
  if proj_in == 'ant':
    proj = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 ' + \
           '+y_0=0 +no_defs +a=6378137 +rf=298.257223563 ' + \
           '+towgs84=0.000,0.000,0.000 +to_meter=1'
  
  else:
    proj = '+proj=stere +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 +x_0=0 ' + \
           '+y_0=0 +no_defs +a=6378137 +rf=298.257223563 ' + \
           '+towgs84=0.000,0.000,0.000 +to_meter=1'
  
  p = Proj(proj)
  
  # Get lon,lat from mesh coordinates
  lon,lat = p(vx,vy,inverse=True)
  
  # Parse output projection for basemap to read
  #p_out = parse_proj4(proj)

  # Extents of plot
  #lat_min,lat_max = min(lat), max(lat)
  #lon_min,lon_max = min(lon), max(lon)

  #width  = 1.05 * (max(vx) - min(vx))
  #height = 1.05 * (max(vy) - min(vy))

  #lat_0_out = median(lat) - 2 # This 2 degree offset is a bad hack...
  #lon_0_out = median(lon)

  #m = Basemap(ax=ax, width=width, height=height, resolution='h', 
  #            projection=p_out['proj'], lat_ts=float(p_out['lat_ts']), 
  #            lon_0=lon_0_out, lat_0=lat_0_out)
  
  m = Basemap(ax=ax, width=2850000,height=2500000,
              resolution='h',projection='stere',
              lat_ts=0,lat_0=-90.0,lon_0=0.)

  # conversion to projection coordinates from lon, lat :
  x, y = m(lon, lat)
  
  m.drawcoastlines(linewidth=0.25, color = 'black')
  m.etopo()
  
  # The divisors in the following may have to be adjusted to get regular spacing
  #meridians = np.arange(lon_min, lon_max, int((lon_max-lon_min)/5.)).round()
  #parallels = np.arange(lat_min, lat_max, int((lat_max-lat_min)/4)).round()
  
  #m.drawmeridians(meridians, color='black', labels=[False, False, False, True])
  #m.drawparallels(parallels, color='black', labels=[True, False, False, False])
 
  ## draw lat/lon grid lines every 5 degrees.
  ## labels = [left,right,top,bottom]
  #m.drawmeridians(np.arange(0, 360, 0.25),
  #                color = 'black',
  #                labels = [False, False, False, True])
  #m.drawparallels(np.arange(-90, 90, 0.25), 
  #                color = 'black', 
  #                labels = [True, False, False, False])
  #m.drawmapscale(-34, 60.5, lon_0_out, lat_0_out, 400, 
  #               yoffset  = 0.01 * (m.ymax - m.ymin), 
  #               barstyle = 'fancy')
  
  m.drawparallels(np.arange(-90.,0.,20.))
  m.drawmeridians(np.arange(-179.89769,180.,20.), labels=[1,0,0,1])
  m.drawmapboundary(fill_color='aqua') 

  #=============================================================================
  # plotting :
  # contour levels :
  if scale == 'log':
    vmax   = 1500.0#ceil(max(v))
    if field == 'melt_rate':
      v[where(v<=0.0)] = 0.00001
      #v[where(v<=0.0)] = 0.0 + min(v[where(v>0.0)])/10
      levels    = logspace(log10(0.001), log10(vmax), numLvls)
      cb_levels = logspace(log10(0.001), log10(vmax), 20)
      #cb_levels = [0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02, 
      #          0.04, 0.08, 0.20, 0.40,0.80 ]
    elif field == 'sliding_ratio':
      v[where(v<=0.0)] = 0.001
      levels    = logspace(log10(0.1), log10(2), numLvls)
      cb_levels = logspace(log10(0.1), log10(2), 20)
    else:
      vmin = 1.0
      vmax = 1500.0
      v[where(v<=vmin)] = vmin
      v[where(v>=vmax)] = vmax
      levels    = logspace(log10(vmin), log10(vmax), numLvls)
      cb_levels = logspace(log10(vmin), log10(vmax), 20)
    from matplotlib.ticker import LogFormatter
    formatter = LogFormatter(10, labelOnlyBase=False)
    norm = colors.LogNorm()
  
  elif scale == 'lin':
    vmin      = floor(min(v))
    vmax      = ceil(max(v))
    levels    = linspace(vmin, vmax+1, numLvls)
    cb_levels = linspace(vmin, vmax+1, 20)
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    norm = None
  
  elif scale == 'bool':
    levels    = [0, 1, 2]
    cb_levels = [0, 1, 2]
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    norm = None
  

  # plots with generated triangles :
  if plot_type == 'gouraud':
    cs = tripcolor(x, y, fi, v, shading='gouraud', 
                   cmap=get_cmap(cmap), norm=norm)
 
  elif plot_type == 'tripcolor': 
    cs = tripcolor(x, y, fi, v, cmap=get_cmap(cmap), norm=norm)
  
  else:
    cs = tricontourf(x, y, fi, v, levels=levels, 
                     cmap=get_cmap(cmap), norm=norm)
  
  # plot triangles :
  if tp == True:
    tp = triplot(x, y, fi, '-', lw=0.2, alpha=tpAlpha)
  
  # include colorbar :
  if field != 'melted' :
    if field == 'beta2' or field == 'U_observed' or field == 'U_surface'\
       or field == 'U_bed' or field == 'sliding_ratio':
      levels[0] = 0.0
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=around(cb_levels,decimals=dec), 
                        location='right', pad="5%")
    
    elif field == 'T_surf' or field == 'T_basal':
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=around(cb_levels,decimals=dec), 
                        location='right', pad="5%")

    elif field == 'melt_rate':
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=cb_levels, 
                        location='right', pad="5%")
  
    elif field == 'sliding_ratio':
      levels[0] = 0.0
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=cb_levels, 
                        location='right', pad="5%")
    
    else :
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=around(cb_levels,decimals=dec), 
                        location='right', pad="5%")
  
  # colorbar label :
  txt = figtext(.94, .06, units)
  txt.set_fontsize(20.0)
  tit = title(name)
  
  savefig(direc + '/images/' + name + '.png', dpi=250)
  show()
