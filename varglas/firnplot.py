from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from pylab import *

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'
    
class FixedOrderFormatter(ScalarFormatter):
  """
  Formats axis ticks using scientific notation with a constant order of 
  magnitude
  """
  def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
    self._order_of_mag = order_of_mag
    ScalarFormatter.__init__(self, useOffset=useOffset, 
                             useMathText=useMathText)
  def _set_orderOfMagnitude(self, range):
    """
    Over-riding this to avoid having orderOfMagnitude reset elsewhere
    """
    self.orderOfMagnitude = self._order_of_mag


class FirnPlot(object):
  """
  Plotting class handles all things related to plotting.
  """
  def __init__(self, model, config):
    """
    Initialize plots with model object as input.
    """   
    self.model  = model
    self.config = config
    
    # plotting extents :
    zMin     = config['zMin'] 
    zMax     = config['zMax'] 
    wMin     = config['wMin'] 
    wMax     = config['wMax'] 
    uMin     = config['uMin'] 
    uMax     = config['uMax'] 
    rhoMin   = config['rhoMin'] 
    rhoMax   = config['rhoMax'] 
    rMin     = config['rMin'] 
    rMax     = config['rMax'] 
    Tmin     = config['Tmin'] 
    Tmax     = config['Tmax'] 
    ageMin   = config['ageMin'] 
    ageMax   = config['ageMax'] 
    WMin     = config['WMin'] 
    WMax     = config['WMax'] 

    # plotting windows :
    Tb       = config['enthalpy']
    rhob     = config['density']
    wb       = config['velocity']
    ageb     = config['age']
             
    blist    = [Tb, rhob, wb, ageb]
    totb     = 100 + 10 * sum(blist)
    
    # convert to arrays :
    thetap  = model.theta.vector().array()
    rhop    = model.rho.vector().array()
    wp      = model.w.vector().array()
    agep    = model.age.vector().array()
    Tp      = model.T.vector().array()
    Wp      = model.W.vector().array()
    rp      = model.r.vector().array()
    pp      = model.p.vector().array()
    up      = model.u.vector().array()
    Smip    = model.Smi.vector().array()
    cp      = model.cif.vector().array()
    Ts      = thetap[0] / cp[0]

    # x-values :
    T      = Tp
    W      = Wp
    rho    = rhop
    w      = wp * model.spy(0)  # m/a
    u      = up * 1e2           # cm/s
    a      = agep /model.spy(0)
    Smi    = Smip
    r      = rp * 1000
    Ts     = Ts - 273.15
    rhos   = rho[0]
    adot   = model.adot.vector().array()[0]

    # y-value :
    z      = model.z
    zs     = z[0]
    zb     = z[-1]
    
    # original surface height :
    zo     = model.zo

    Th     = Tmin + 0.1*(Tmax - Tmin) / 2         # T height x-coord
    Tz     = zMax - 0.15*(zMax - zMin) / 2        # z-coord of Ts
    rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord
    wh     = wMin + 0.1*(wMax - wMin) / 2
    #kh    = kMin + 0.1*(kMax - kMin) / 2

    figx = 4.0 * sum(blist)
    i    = 1
    pur = '#880cbc'
    plt.ion()
    self.fig   = plt.figure(figsize=(figx,6))
    if Tb:
      # temperature axis :
      self.Tax   = self.fig.add_subplot(totb + i)
      self.Tax.axis([Tmin, Tmax, zMin, zMax])
      self.Tax.set_xlabel(r'$T\ [\degree \mathrm{C}]$')
      self.Tax.set_ylabel('Depth [m]')
      self.Tax.grid()
      
      # surface temp text :
      self.Tsurf    = self.Tax.text(Th, Tz, r'$T_S$: %.1E $\degree$C' % Ts)

      # temperature profile :
      self.phT,     = self.Tax.plot(T - 273.15, z, '0.3', lw=1.5,
                                    drawstyle='steps-pre')
      
      # firn surface :
      self.phTs,    = self.Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=3)
      
      # original (beginning) surface height :
      self.phTs_0,  = self.Tax.plot(Th, zo, 'go')

      # temperature surface boundary condition :
      self.phTs_dot,= self.Tax.plot(Ts, zs, 'ko')

      # grid spacing :
      self.phTsp,   = self.Tax.plot(Th*ones(len(z)), z, 'g+')
      
      # water content axis :
      self.Oax   = self.Tax.twiny()
      self.Oax.axis([WMin, WMax, zMin, zMax])
      self.Oax.set_xlabel(r'$W\ [\mathrm{m}^{3} \mathrm{m}^{-3}]$', color=pur)
      self.Oax.grid()
      for tl in self.Oax.get_xticklabels():
        tl.set_color(pur)

      # water content :
      self.phO,     = self.Oax.plot(W, z, pur, lw=1.5,
                                    drawstyle='steps-pre')

      # water content surface boundary condition :
      self.phO_dot, = self.Oax.plot(W[0], zs, color=pur, marker='o')

      # irreducible water content :
      self.phSmi,   = self.Oax.plot(Smi,   z, 'r--',   lw=1.5,
                                    drawstyle='steps-pre')

      # irreducible water content surface boundary condition :
      self.phSmi_dot, = self.Oax.plot(Smi[0], zs, 'ro')

      i += 1

    if rhob:
      # density axis :
      self.rhoax = self.fig.add_subplot(totb + i)
      self.rhoax.axis([rhoMin, rhoMax, zMin, zMax])
      self.rhoax.xaxis.set_major_formatter(FixedOrderFormatter(2))
      text = r'$\rho\ \left[\frac{\mathrm{kg}}{\mathrm{m}^3}\right]$'
      self.rhoax.set_xlabel(text)
      self.rhoax.grid()

      # surface density text :
      text = r'$\dot{a}$: %.1E i.e.$\frac{\mathrm{m}}{\mathrm{a}}$' % adot
      self.rhoSurf  = self.rhoax.text(rhoh, Tz, text)

      # density profile :
      self.phrho,   = self.rhoax.plot(rho, z, '0.3', lw=1.5,
                                      drawstyle='steps-pre')

      # surface height :
      self.phrhoS,  = self.rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=3)

      # density surface boundary condition :
      self.phrhoS_dot, = self.rhoax.plot(rho[0], zs, 'ko')

      #self.phrhoS_0,= self.rhoax.plot(rhoh, zo, 'ko')
      #self.phrhoSp, = self.rhoax.plot(rhoh*ones(len(z)), z, 'r+')
      
      # grain-size axis :
      self.rax   = self.rhoax.twiny()
      self.rax.axis([rMin, rMax, zMin, zMax])
      text = r'$r\ \left[\mathrm{mm}\right]$'
      self.rax.set_xlabel(text, color=pur)
      self.rax.grid()
      for tl in self.rax.get_xticklabels():
        tl.set_color(pur)
      
      # grain-size profile :
      self.phr,     = self.rax.plot(r, z, pur, lw=1.5,
                                    drawstyle='steps-pre')
      # grain-size surface boundary condition :
      self.phr_dot, = self.rax.plot(r[0], zs, color=pur, marker='o')

      i += 1

    if wb:
      # firn compaction velocity axis :
      self.wax   = self.fig.add_subplot(totb + i)
      self.wax.axis([wMin, wMax, zMin, zMax])
      self.wax.set_xlabel(r'$w\ \left[\frac{\mathrm{m}}{\mathrm{a}}\right]$')
      self.wax.grid()

      # surface accumulation text :
      text = r'$\rho_S$: %.1E $\frac{\mathrm{kg}}{\mathrm{m}^3}$' % rhos
      self.wSurf    = self.wax.text(wh, Tz, text)

      # compaction velocity profile :
      self.phw,     = self.wax.plot(w, z, '0.3', lw=1.5,
                                    drawstyle='steps-pre')
      
      # compaction velocity surface boundary condition :
      self.wS_dot,  = self.wax.plot(w[0], zs, 'ko')

      # surface height :
      self.phwS,    = self.wax.plot([wMin, wMax], [zs, zs], 'k-', lw=3)

      #self.phws_0,  = self.wax.plot(wh, zo, 'ko')
      #self.phwsp,   = self.wax.plot(wh*ones(len(z)), z, 'r+')
      
      # water velocity axis :
      self.uax   = self.wax.twiny()
      self.uax.axis([uMin, uMax, zMin, zMax])
      #self.uax.xaxis.set_major_formatter(FixedOrderFormatter(2))
      #self.uax.ticklabel_format(style='sci',scilimits=(uMin,uMax), axis='x')
      text = r'$u\ \left[\frac{\mathrm{cm}}{\mathrm{s}}\right]$'
      self.uax.set_xlabel(text, color = pur)
      self.uax.grid()
      for tl in self.uax.get_xticklabels():
        tl.set_color(pur)
      
      # water velocity profile :
      self.phu,     = self.uax.plot(u, z, pur, lw=1.5,
                                    drawstyle='steps-pre')

      # water velocity surface boundary condition :
      self.uS_dot,  = self.uax.plot(u[0], zs, color=pur, marker='o')

      i += 1

    if ageb:
      self.aax   = self.fig.add_subplot(totb + i)
      self.aax.axis([ageMin, ageMax, zMin, zMax])
      #self.aax.xaxis.set_major_formatter(FixedOrderFormatter(3))
      self.aax.grid()
      self.pha,     = self.aax.plot(a, z, '0.3', lw=1.5,
                                    drawstyle='steps-pre')
      self.phaS,    = self.aax.plot([ageMin, ageMax], [zs, zs], 'k-', lw=3)
      self.aax.set_title('Age')
      self.aax.set_xlabel(r'$a\ [\mathrm{a}]$')

      
    #self.phks_0,  = self.kax.plot(kh, zo, 'ko')
    #self.phksp,   = self.kax.plot(kh*ones(len(z)), z, 'r+')

    # formatting :
    self.fig.canvas.set_window_title('Time = 0.0 yr')
    plt.tight_layout()
    plt.show()

  def update(self):
    """
    Update the plot for each time step at time t.
    """
    model  = self.model
    config = self.config
    index  = model.index 
    
    # convert to arrays :
    thetap = model.theta.vector().array()
    rhop   = model.rho.vector().array()
    wp     = model.w.vector().array()
    agep   = model.age.vector().array()
    Tp     = model.T.vector().array()
    Wp     = model.W.vector().array()
    rp     = model.r.vector().array()
    pp     = model.p.vector().array()
    up     = model.u.vector().array()
    Smip   = model.Smi.vector().array()
    cp     = model.cif.vector().array()
    Ts     = thetap[0] / cp[0]
    T_w    = model.T_w(0)

    # x-values :
    T      = Tp
    W      = Wp
    rho    = rhop
    w      = wp * model.spy(0)  # m/a
    u      = up * 1e2           # cm/s
    a      = agep /model.spy(0)
    Smi    = Smip
    r      = rp * 1000
    Ts     = Ts - T_w
    adot   = model.adot.vector().array()[0]
    t      = model.t / model.spy(0)
    phi    = 1 - rho/917.0
    Smi    = 0.0057 / (1 - phi) + 0.017
    
    z      = model.z
    zo     = model.zo
    zs     = model.z[0]

    self.fig.canvas.set_window_title('Time = %.2f yr' % t)
   
    if config['enthalpy']: 
      self.Tsurf.set_text(r'$T_S$: %.1E $\degree$C' % Ts)
      self.phT.set_xdata(T - T_w)
      self.phT.set_ydata(z)
      self.phTs.set_ydata(zs)
      self.phTs_0.set_ydata(zo)
      self.phTsp.set_ydata(z)
      self.phTs_dot.set_xdata(Ts)
      self.phTs_dot.set_ydata(zs)
      self.phO.set_xdata(W)
      self.phO.set_ydata(z)
      self.phO_dot.set_xdata(W[0])
      self.phO_dot.set_ydata(zs)
      self.phSmi.set_xdata(Smi)
      self.phSmi.set_ydata(z)
      self.phSmi_dot.set_xdata(Smi[0])
      self.phSmi_dot.set_ydata(zs)
      
    if config['density']: 
      text = r'$\rho_S$: %.1E $\frac{\mathrm{kg}}{\mathrm{m}^3}$' % rho[0]
      self.rhoSurf.set_text(text)
      self.phrho.set_xdata(rho)
      self.phrho.set_ydata(z)
      self.phr.set_xdata(r)
      self.phr.set_ydata(z)
      self.phrhoS.set_ydata(zs)
      self.phrhoS_dot.set_xdata(rho[0])
      self.phrhoS_dot.set_ydata(zs)
      self.phr_dot.set_xdata(r[0])
      self.phr_dot.set_ydata(zs)
      
    if config['velocity']: 
      text = r'$\dot{a}$: %.1E i.e.$\frac{\mathrm{m}}{\mathrm{a}}$'
      self.wSurf.set_text(text % adot)
      self.phw.set_xdata(w)
      self.phw.set_ydata(z)
      self.phu.set_xdata(u)
      self.phu.set_ydata(z)
      self.phwS.set_ydata(zs)
      self.wS_dot.set_xdata(w[0])
      self.wS_dot.set_ydata(zs)
      self.uS_dot.set_xdata(u[0])
      self.uS_dot.set_ydata(zs)
     
    if config['age']: 
      self.pha.set_xdata(a)
      self.pha.set_ydata(z)
      self.phaS.set_ydata(zs)

    plt.draw()
    plt.pause(0.00000001)
    

  def plot_all(self, models, titles, colors):
    """
    Plot the data from a list of model objects with corresponding titles and
    colors array.
    """    
    Tmin   = -20                                 # T x-coord min
    Tmax   = 0                                   # T x-coord max
    Th     = Tmin + 0.1*(Tmax - Tmin) / 2        # T height x-coord

    rhoMin = 300                                 # rho x-coord min
    rhoMax = 1000                                # rho x-coord max
    rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord

    wMin   = -22.0e-6
    wMax   = -7.5e-6
    wh     = wMin + 0.1*(wMax - wMin) / 2

    kMin   = 0.0
    kMax   = 2.2
    kh     = kMin + 0.1*(kMax - kMin) / 2
    
    zMax   = models[0].zs + 20                    # max z-coord
    zMin   = models[0].zb                         # min z-coord

    fig    = figure(figsize=(16,6))
    Tax    = fig.add_subplot(141)
    rhoax  = fig.add_subplot(142)
    wax    = fig.add_subplot(143)
    kax    = fig.add_subplot(144)

    # format : [xmin, xmax, ymin, ymax]
    Tax.axis([Tmin, Tmax, zMin, zMax])
    Tax.grid()
    rhoax.axis([rhoMin, rhoMax, zMin, zMax])
    rhoax.grid()
    rhoax.xaxis.set_major_formatter(FixedOrderFormatter(2))
    wax.axis([wMin, wMax, zMin, zMax])
    wax.grid()
    wax.xaxis.set_major_formatter(FixedOrderFormatter(-6))
    kax.axis([kMin, kMax, zMin, zMax])
    kax.grid()

    # plots :
    for model, title, color in zip(models, titles, colors):
      i = model.index
      Tax.plot(model.T[i] - 273.15, model.z[i], color, label=title, lw=2)
      Tax.plot([Tmin, Tmax], [model.z[i][-1], model.z[i][-1]], color, lw=2)

      rhoax.plot(model.rho[i], model.z[i], color, lw=2)
      rhoax.plot([rhoMin, rhoMax],[model.z[i][-1], model.z[i][-1]], color, lw=2)

      wax.plot(model.w[i], model.z[i], color, lw=2)
      wax.plot([wMin, wMax], [model.z[i][-1], model.z[i][-1]], color, lw=2)

      kax.plot(model.k2[i], model.z[i], color, lw=2)
      kax.plot([kMin, kMax], [model.z[i][-1], model.z[i][-1]], color, lw=2)
    
    # formatting :
    fig_text = figtext(.85,.95,'Time = 0.0 yr')

    Tax.set_title('Temperature')
    Tax.set_xlabel(r'$T$ $[\degree C]$')
    Tax.set_ylabel(r'Depth $[m]$')

    rhoax.set_title('Density')
    rhoax.set_xlabel(r'$\rho$ $\left [\frac{kg}{m^3}\right ]$')
    #rhoax.set_ylabel(r'Depth $[m]$')

    wax.set_title('Velocity')
    wax.set_xlabel(r'$w$ $\left [\frac{mm}{s}\right ]$')
    #wax.set_ylabel(r'Depth $[m]$')

    kax.set_title('Thermal Conductivity')
    kax.set_xlabel(r'$k$ $\left [\frac{J}{m K s} \right ]$')
    #kax.set_ylabel(r'Depth $[m]$')
    
    # Legend formatting:
    leg    = Tax.legend(loc='upper right')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    setp(ltext, fontsize='small')
    frame.set_alpha(0)

    show()


  def plot_height(self, x, ht, origHt):
    """
    Plot the height history of a column of firn for times x, current height ht, 
    and original surface height origHt.
    """
    x /= self.model.spy

    # plot the surface height information :
    plot(x,               ht,     'k-',  lw=1.5, label=r'Surface Height')
    plot(x[:len(origHt)], origHt, 'k--', lw=1.5, label=r'Original Surface')
    xlabel('time [a]')
    ylabel('height [m]')
    title('Surface Height Changes')
    grid()
  
    # Legend formatting:
    leg = legend(loc='lower left')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    setp(ltext, fontsize='small')
    frame.set_alpha(0)
    show()


  def plot_all_height(self, xs, hts, origHts, titles, colors):
    """
    Plot the height history of a list of model objects for times array xs, 
    current height array hts, original surface heights origHts, with 
    corresponding titles and colors arrays.
    """
    zMin = min(min(origHts))
    zMax = max(max(hts))
    zMax = zMax + (zMax - zMin) / 16.0
    xMin = min(min(xs))
    xMax = max(max(xs))
    
    fig = figure(figsize=(11,8))
    ax  = fig.add_subplot(111)
    
    # format : [xmin, xmax, ymin, ymax]
    ax.axis([xMin, xMax, zMin, zMax])
    ax.grid()
    
    # plot the surface height information :
    for x, ht, origHt, title, color in zip(xs, hts, origHts, titles, colors):
      ax.plot(x, ht,     color + '-',  label=title + ' Surface Height')
      ax.plot(x, origHt, color + '--', label=title + ' Original Surface')
    
    ax.set_xlabel('time [a]')
    ax.set_ylabel('height [m]')
    ax.set_title('Surface Height Changes')
    ax.grid()
  
    # Legend formatting:
    leg    = ax.legend(loc='lower left')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    setp(ltext, fontsize='small')
    frame.set_alpha(0)
    show()




