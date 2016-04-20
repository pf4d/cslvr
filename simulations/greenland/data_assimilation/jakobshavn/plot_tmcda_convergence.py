from numpy import *
import matplotlib.pyplot as plt
import matplotlib        as mpl
import os

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'x-small'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']

# set the relavent directories :
in_dir  = 'dump/jakob_small/inversion_Wc_0.01/'
out_dir = in_dir

def get_data(direc):
  Js  = []
  J1s = []
  J2s = []
  Rs  = []
  Is  = []
  ns  = []
  
  for d in next(os.walk(direc))[1]:
    try:
      i = int(d)
      dn = direc + d + '/objective_ftnls_history/'
      Is.append(i)
      Js.append(loadtxt(dn  + 'Js.txt'))
      J1s.append(loadtxt(dn + 'J1s.txt'))
      J2s.append(loadtxt(dn + 'J2s.txt'))
      Rs.append(loadtxt(dn  + 'Rs.txt'))
      ns.append(len(Js[-1]))
    except ValueError:
      pass
  
  Js  = array(Js)
  J1s = array(J1s)
  J2s = array(J2s)
  Rs  = array(Rs)
  Is  = array(Is)
  ns  = array(ns)
  
  idx  = argsort(Is)
  Js   = Js[idx]
  J1s  = J1s[idx]
  J2s  = J2s[idx]
  Rs   = Rs[idx]
  ns   = ns[idx]
  xn   = cumsum(ns - 1)

  return (Is, Js, J1s, J2s, Rs, ns, xn)


out = get_data(in_dir)

Is, Js, J1s, J2s, Rs, ns, xn = out

fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(111)

k = 0

dkred  = '#820000'
pink   = '#ff8e8e'
ltpink = '#ffbaba'

for i in range(len(Is)):
  xi = arange(k, k + ns[i])
  
  if i == 0:
    ax1.plot(xi, Js[i],       '-',  c='k',   lw=2,
             label=r'$\mathscr{I}(\beta, \mathbf{u})$')
    ax1.plot(xi, 5000*J2s[i], '-',  c='0.6', lw=2,
             label=r'$\mathscr{I}_1(\mathbf{u}, \gamma_1)$')
    ax1.plot(xi, 0.01*J1s[i], '-',  c='0.4', lw=2,
             label=r'$\mathscr{I}_2(\mathbf{u}, \gamma_2)$')
    ax1.plot(xi, 10.0*Rs[i],  '--', c='r',   lw=2,
             label=r'$\mathscr{I}_3(\beta, \gamma_3)$')
  else:
    ax1.plot(xi, Js[i],       '-',  c='k',   lw=2)
    ax1.plot(xi, 5000*J2s[i], '-',  c='0.6', lw=2)
    ax1.plot(xi, 0.01*J1s[i], '-',  c='0.4', lw=2)
    ax1.plot(xi, 10.0*Rs[i],  '--', c='r',   lw=2)

  k += ns[i] - 1

leg1 = ax1.legend(loc='upper right', ncol=4)
leg1.get_frame().set_alpha(0.0)

ax1.grid()

ax1.set_ylim([1e11, 2e13])

ax1.set_xlabel('iteration')
ax1.set_ylabel(r'$\mathscr{I}$')

ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)

yloc1 = plt.MaxNLocator(4)
ax1.yaxis.set_major_locator(yloc1)
ax1.set_yscale('log')


plt.tight_layout()
plt.savefig(out_dir + 'convergence_plot.pdf')
plt.close(fig)



