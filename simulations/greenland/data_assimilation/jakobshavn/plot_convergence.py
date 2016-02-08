from numpy import *
import matplotlib.pyplot as plt
import matplotlib        as mpl
import os

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'medium'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']

in_dir = 'dump/jakob_small/rstrt_alpha_1e8_regularized/tmc/'

abs_err = loadtxt(in_dir + 'convergence_history/abs_err.txt')
tht_nrm = loadtxt(in_dir + 'convergence_history/theta_norm.txt')

Ds = []
Js = []
Rs = []
Is = []
ns = []

for d in next(os.walk(in_dir))[1]:
  try:
    i = int(d)
    dn = in_dir + d + '/objective_ftnls_history/'
    Is.append(i)
    Ds.append(loadtxt(dn + 'Ds.txt'))
    Js.append(loadtxt(dn + 'Js.txt'))
    Rs.append(loadtxt(dn + 'Rs.txt'))
    ns.append(len(Js[-1]))
  except ValueError:
    pass

Ds = array(Ds)
Js = array(Js)
Rs = array(Rs)
Is = array(Is)
ns = array(ns)

idx = argsort(Is)
Ds  = Ds[idx]
Js  = Js[idx]
Rs  = Rs[idx]
ns  = ns[idx]

xn  = cumsum(ns - 1)

Rmax = Rs.max()
Jmax = Js.max()
amax = abs_err.max()
tmax = tht_nrm.max()

#===============================================================================
# plotting :

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(211)
ax2 = ax1.twinx()
ax3 = fig.add_subplot(212)
ax4 = ax3.twinx()

k = 0

ax3.plot(xn, abs_err - amax, 'ko-', lw=2.0)
ax4.plot(xn, tht_nrm - tmax, 'ro-', lw=2.0)

for i in range(len(Is)):
  xi = arange(k, k + ns[i])
  ax1.plot(xi, Js[i] - Jmax, 'k-', lw=2.0)
  ax2.plot(xi, Rs[i] - Rmax, 'r-', lw=2.0)

  k += ns[i] - 1

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax3.set_xlim([0, xn.max()])
ax4.set_xlim([0, xn.max()])

ax1.set_xlabel('iteration')
ax3.set_xlabel('iteration')
ax1.set_ylabel(r'$\mathscr{J}$')
ax2.set_ylabel(r'$\mathscr{R}$')
ax3.set_ylabel(r'$\Vert \theta - \theta_n \Vert$')
ax4.set_ylabel(r'$\Vert \theta \Vert$')

ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
ax2.tick_params(axis='y', colors='r')
ax4.tick_params(axis='y', colors='r')

yloc1 = plt.MaxNLocator(4)
yloc2 = plt.MaxNLocator(4)
yloc3 = plt.MaxNLocator(4)
yloc4 = plt.MaxNLocator(4)
ax1.yaxis.set_major_locator(yloc1)
ax2.yaxis.set_major_locator(yloc2)
ax3.yaxis.set_major_locator(yloc3)
ax4.yaxis.set_major_locator(yloc4)

#ax2.set_yscale('log')
ax2.yaxis.get_offset_text().set_color('r')
ax2.yaxis.label.set_color('r')
ax4.yaxis.get_offset_text().set_color('r')
ax4.yaxis.label.set_color('r')

plt.tight_layout()
plt.show()



