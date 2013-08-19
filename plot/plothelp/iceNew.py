from mpl_toolkits.basemap import Basemap
from pylab import *
from matplotlib import colors

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

def plotIce(u, cmap, scale='lin', name='', units='', 
            numLvls=12, tp=False, tpAlpha=0.5):
  """
  INPUTS :

    u :
      solution to plot.
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
    tp :
      boolean determins plotting of triangle overlay
    tpAlpha :
      alpha level of triangles 0.0 (transparent) - 1.0 (opaque)
  
  OUTPUT :
 
    A sigle 250 dpi .png in the source directory.
  
  """
  #=============================================================================
  # format stuff that needs to be formatted :  
  dec = 1
 
  field = '' 
  dec = 1

  # data gathering :
  mesh  = u.function_space().mesh()
  coord = mesh.coordinates()
  fi    = mesh.cells()
  vv    = u.compute_vertex_values(mesh)
  vx    = coord[:,0]
  vy    = coord[:,1]
  

  #=============================================================================
  # map functions :
  fig = plt.figure(figsize=(13.5,7), dpi=80)
  ax  = fig.add_axes()
  
  # map projection that the data is based on :
  m   = Basemap(projection='stere',
                lat_ts=70.0, lat_0=90.0, lon_0=-45.0, resolution='l',
                llcrnrlat=55,llcrnrlon=-55,urcrnrlat=80,urcrnrlon=0.0)
  
  # adjustment of x, y coordinates for N. polar projection :
  x = vx + m.projparams['x_0']
  y = vy + m.projparams['y_0']
  v = vv 
  
  # conversion to lon, lat coordinates :
  lon, lat = m(x, y, inverse = True)
  
  # new projection of our choosing :
  m   = Basemap(ax=ax, width=62000.0, height=26000.0,
                resolution='h',projection='stere',
                lat_ts=67.03, lon_0=-48.675, lat_0=67.03)
  
  m.etopo()

  # conversion to projection coordinates from lon, lat :
  x, y = m(lon, lat)
  
  m.drawcoastlines(linewidth=0.25, color = 'black')
 
  # draw lat/lon grid lines every 5 degrees.
  # labels = [left,right,top,bottom]
  m.drawmeridians(np.arange(0, 360, 0.25),
                  color = 'black',
                  labels = [False, False, False, True])
  m.drawparallels(np.arange(-90, 90, 0.25), 
                  color = 'black', 
                  labels = [True, False, False, False])
  m.drawmapscale(-48.25, 66.94, -48.675, 67.675, 10,
                  yoffset=(0.01*(m.ymax-m.ymin)),
                  barstyle='fancy')
  
  #=============================================================================
  # plotting :
  # countour levels :
  if scale == 'log':
    vmax   = ceil(max(v))
    if field == 'melt_rate':
      v[where(v<=0.0)] = 0.00001
      #v[where(v<=0.0)] = 0.0 + min(v[where(v>0.0)])/10
      levels = logspace(log10(0.001), log10(vmax), numLvls)
      #levels = [0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02, 
      #          0.04, 0.08, 0.20, 0.40,0.80 ]
    elif field == 'sliding_ratio':
      v[where(v<=0.0)] = 0.001
      levels = logspace(log10(0.1), log10(2), numLvls)
    else:
      vmin = floor(min(v))
      v[where(v<=0.0)] = 1.0
      levels = logspace(0.0, log10(vmax+1), numLvls)
    from matplotlib.ticker import LogFormatter
    formatter = LogFormatter(10, labelOnlyBase=False)
    norm = colors.LogNorm()
  
  elif scale == 'lin':
    vmin   = floor(min(v))
    vmax   = ceil(max(v))
    levels = linspace(vmin, vmax+1, numLvls)
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    norm = None
  
  elif scale == 'bool':
    levels = [0, 1, 2]
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    norm = None
  

  # plots with generated triangles :
  cs = tripcolor(x, y, fi, v, shading='gouraud', 
                 cmap=get_cmap(cmap), norm=norm)
  
  #cs = tricontourf(x, y, fi, v, #levels=levels, 
  #                 cmap=get_cmap(cmap), norm=norm)
  
  # plot triangles :
  if tp == True:
    tp = triplot(x, y, fi, '-', lw=0.2, alpha=tpAlpha)
  
  # include colorbar :
  if field != 'melted' :
    if field == 'beta2' or field == 'U_observed' or field == 'U_surface'\
       or field == 'U_bed' or field == 'sliding_ratio':
      levels[0] = 0.0
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=around(levels,decimals=dec), 
                        location='right', pad="5%")
    
    elif field == 'T_surf' or field == 'T_basal':
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=around(levels,decimals=dec), 
                        location='right', pad="5%")

    elif field == 'melt_rate':
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=levels, 
                        location='right', pad="5%")
  
    elif field == 'sliding_ratio':
      levels[0] = 0.0
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=levels, 
                        location='right', pad="5%")
    
    else :
      cbar = m.colorbar(cs, format=formatter, 
                        ticks=around(levels,decimals=dec), 
                        location='right', pad="5%")
  
  # colorbar label :
  txt = figtext(.95, .50, units)
  txt.set_fontsize(20.0)
  tit = title(name)
  
  savefig(name + '.png', dpi=250)
  show()
