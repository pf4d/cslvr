import numpy as np
from pylab import *
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
import pyproj
import inspect
import os

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

def plotIce(u, cmap = 'wiki', overlay = None,overlay2= None, overlay_alpha=.3, overlay_alpha2=.8,name = 'No name provided',\
        units = '', mesh_plot = False, meshAlpha = 0.5, lines=False):
    """
               started as a fairly general tool, but has been degrated into
               something specific for Isunnguata Sermia. Will require a
               significant overhaul to make it general.
    
    Written by: Evan Cummings modified by JVJ
    
    INPUTS :
    
      u :
        FeNICS solution object to plot...must be object of type Function.
      name :
        title of the plot, latex accepted
      units :
        units for the colorbar, latex approved
      mesh_plot :
        boolean determins plotting of mesh overlay
      meshAlpha :
        alpha level of mesh 0.0 (transparent) - 1.0 (opaque)
    
    OUTPUT :
    
      A sigle 250 dpi name.png in the source directory.
    
    """
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    home     = os.path.dirname(os.path.abspath(filename))

    # Mesh data needed for plotting
    mesh = u.function_space().mesh()  # Pull mesh
    coord = mesh.coordinates()        # Gather coordinates

    vx = coord[:, 0]
    vy = coord[:, 1]
    fi = mesh.cells()        # These are the sets of vertices forming a triangle
    # Values corresponding to the dataset being plotted
    v = u.compute_vertex_values(mesh) 

    # Set plot size:
    fig = plt.figure(figsize=(13.5, 7), dpi=80)
    ax = fig.add_axes()
        
    width = 236. * 250
    height = 90. * 250

    # This is the 'new' way to move x,y mesh points to lon,lat. Based on pyproj, instead of basemap.
    #It seems to work.
    p = pyproj.Proj("+proj=stere +lat_0=90 +lat_ts=71 +lon_0=-46 +k=1 +x_0=0 +y_0=0 +no_defs +a=6378137 +rf=298.257223563 +towgs84=0.000,0.000,0.000 +to_meter=1") 

    lon,lat = p(vx,vy,inverse=True)

    # Given original data and projection, place onto projection here that
    # 'looks' nice
    m = Basemap(ax=ax, width=width, height=height, resolution='h', projection='stere', lat_ts=71, lon_0=-49.67, lat_0=67.185)
    x, y = m(lon, lat) 

    # Annotations on plot
    m.drawmeridians(np.arange(-55, -40, 0.25), color='black', labels=[False, False, False, True])
    m.drawparallels(np.arange(66, 68, 0.1), color='black')

    m.drawmapscale(-49.125, 67.097, -49.65, 67.19, 10, yoffset=0.02 * (m.ymax - m.ymin), barstyle='fancy')

    m.drawmapboundary(fill_color='white')

    # This has been a major hassle to generalize, mostly due to the colormaps
    # requiring certain asymmetry about zero, and my incistence that zero be
    # sealevel
    levels = np.arange(-3,.5,.25)

    from matplotlib import colors
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()

    if cmap == 'mby':
        cdict = gmtColormap(home + '/plothelp/mby.cpt')
        color_map = colors.LinearSegmentedColormap('color_map', cdict)
    elif cmap == 'wiki':
        vmin = -550   # These are the topo extents, scaled to make 0 make sense
        vmax = 1100
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.arange(vmin, vmax, 50)
        cdict = gmtColormap(home + '/plothelp/wiki-2.0.cpt')
        color_map = colors.LinearSegmentedColormap('color_map', cdict)
    elif cmap == 'gist_rainbow':
        vmin = 12.   # These are the topo extents, scaled to make 0 make sense
        vmax = 425. 
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, 256)
        color_map = cm.gist_rainbow
        ticks =  linspace(vmin,vmax,10).round()
    else:
        cdict == gmtColormap(home + '/plothelp/ETOPO1.cpt')
        color_map = colors.LinearSegmentedColormap('color_map', cdict)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cs = tricontourf(x, y, fi, v, norm=norm, cmap=get_cmap(color_map), levels=levels)

    if mesh_plot == True:
        mesh_handle = triplot(x, y, fi, '-', lw=0.2, alpha=meshAlpha)

    if overlay:
        mesh = overlay.function_space().mesh()  # Pull mesh
        vo = overlay.compute_vertex_values(mesh) 
        vo[vo>0.]=1.
        norm = colors.Normalize(0., 1.)
        ol = tricontourf(x,y,fi,vo,levels=[.75,1.25],alpha=overlay_alpha,cmap = cm.Purples,norm=norm)

    if overlay2:
        mesh = overlay2.function_space().mesh()  # Pull mesh
        vo = overlay2.compute_vertex_values(mesh) 
        ol = tricontour(x,y,fi,vo,levels=[12.],linewidths=2.,cmap = cm.gist_gray)

    if lines:
        print "Adding the radar lines..."
        #file = open("plothelp/radar_lines_small.txt")
        file = open(home + "/plothelp/1993_2011_Radar_LL.txt")
        lon,lat=[],[]
        count = 0
        for line in file.readlines():
            sl = line.split('|')
            if count%8 == 0:
                lon.append(float(sl[0]))
                lat.append(float(sl[1]))
            count += 1
        x,y = m(lon,lat)
        m.scatter(x,y,linewidth=0.75,marker='.',s=.6,color='r')

    try:
        cbar = m.colorbar(cs,ticks=ticks, location='right', pad='5%')
    except NameError:
        cbar = m.colorbar(cs, format=formatter, location='right', pad='5%')

    txt = figtext(0.873, 0.77, units)
    txt.set_fontsize(16.0)
    tit = title(name)
    tit.set_fontsize(22.0)
    ll1 = figtext(0.075, 0.275, '67.1$^{\\circ}$N')
    ll1 = figtext(0.075, 0.527, '67.2$^{\\circ}$N')
  
    direc = 'images/' + name + '.png'
    d     = os.path.dirname(direc)
    if not os.path.exists(d):
      os.makedirs(d)
    savefig(direc, dpi=120)
    #show()


def gmtColormap(fileName, log_color = False, reverse = False):
    """
    Import a CPT colormap from GMT.
    
    Parameters
    ----------
    fileName : a cpt file.
    
    Example
    -------
    >>> cdict = gmtColormap("mycolormap.cpt")
    >>> gmt_colormap = colors.LinearSegmentedColormap("my_colormap", cdict)
    
    Notes
    -----
    This code snipplet modified after
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg09547.html
    """
    import colorsys
    try:
        f = open(fileName)
    except:
        print 'file ', fileName, 'not found'
        return

    lines = f.readlines()
    f.close()
    x = []
    r = []
    g = []
    b = []
    colorModel = 'RGB'
    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x.append(float(ls[0]))
            r.append(float(ls[1]))
            g.append(float(ls[2]))
            b.append(float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

    x.append(xtemp)
    r.append(rtemp)
    g.append(gtemp)
    b.append(btemp)
    if reverse:
        r.reverse()
        g.reverse()
        b.reverse()
    x = np.array(x, np.float32)
    r = np.array(r, np.float32)
    g = np.array(g, np.float32)
    b = np.array(b, np.float32)
    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb

    if colorModel == 'RGB':
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    if log_color:
        xNorm = np.zeros((len(x),))
        xNorm[1::] = np.logspace(-1, 0, len(x) - 1)
        xNorm[1::-2] /= 4
    else:
        xNorm = (x - x[0]) / (x[-1] - x[0])
    red = []
    blue = []
    green = []
    for i in range(len(x)):
        red.append([xNorm[i], r[i], r[i]])
        green.append([xNorm[i], g[i], g[i]])
        blue.append([xNorm[i], b[i], b[i]])

    colorDict = {'red': red,
     'green': green,
     'blue': blue}
    return colorDict
