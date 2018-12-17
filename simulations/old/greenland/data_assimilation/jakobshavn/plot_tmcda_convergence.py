from numpy import *
import matplotlib.pyplot as plt
import matplotlib        as mpl
import os

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'x-small'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']

# set the relavent directories :
in_dir  = 'dump/jakob_small/inversion_Wc_0.03/'
out_dir = in_dir

def get_data(direc):
  Js  = []
  J1s = []
  J2s = []
  Rs  = []
  R1s = []
  R2s = []
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
      R1s.append(loadtxt(dn + 'R1s.txt'))
      R2s.append(loadtxt(dn + 'R2s.txt'))
      ns.append(len(Js[-1]))
    except ValueError:
      pass
  
  Js  = array(Js)
  J1s = array(J1s)
  J2s = array(J2s)
  Rs  = array(Rs)
  R1s = array(R1s)
  R2s = array(R2s)
  Is  = array(Is)
  ns  = array(ns)
  
  idx  = argsort(Is)
  Js   = Js[idx]
  J1s  = J1s[idx]
  J2s  = J2s[idx]
  Rs   = Rs[idx]
  R1s  = R1s[idx]
  R2s  = R2s[idx]
  ns   = ns[idx]
  xn   = cumsum(ns - 1)

  return (Is, Js, J1s, J2s, Rs, R1s, R2s, ns, xn)


out = get_data(in_dir)

Is, Js, J1s, J2s, Rs, R1s, R2s, ns, xn = out

fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(111)

k = 0

dkred  = '#820000'
pink   = '#ff8e8e'
ltpink = '#ffbaba'

for i in range(len(Is)):
  xi = arange(k, k + ns[i])

  R_i = 0.1*R1s[i] + 10*R2s[i]
  
  if i == 0:
    #ax1.plot(xi, Js[i],       '-',  c='k',   lw=2,
    #         label=r'$\mathscr{I}(\mathbf{u})$')
    #ax1.plot(xi, R_i,         '-', c=dkred,   lw=2,
    #         label=r'$\mathscr{I}(\beta)$')
    ax1.plot(xi, 10*R2s[i],   '-',  c='r',   lw=2,
             label=r'$\mathscr{I}_4(\beta, \gamma_4)$')
    ax1.plot(xi, 0.1*R1s[i],  '-',  c=pink,  lw=2,
             label=r'$\mathscr{I}_3(\beta, \gamma_3)$')
    ax1.plot(xi, 5000*J2s[i], '-',  c='0.5', lw=2,
             label=r'$\mathscr{I}_1(\mathbf{u}, \gamma_1)$')
    ax1.plot(xi, 0.01*J1s[i], '-',  c='k',   lw=2,
             label=r'$\mathscr{I}_2(\mathbf{u}, \gamma_2)$')
  else:
    #ax1.plot(xi, Js[i],       '-',  c='k',    lw=2)
    #ax1.plot(xi, R_i,    '-',  c=dkred,  lw=2)
    ax1.plot(xi, 10*R2s[i],   '-',  c='r',    lw=2)
    ax1.plot(xi, 0.1*R1s[i],  '-',  c=pink,   lw=2)
    ax1.plot(xi, 5000*J2s[i], '-',  c='0.5',  lw=2)
    ax1.plot(xi, 0.01*J1s[i], '-',  c='k',    lw=2)

  k += ns[i] - 1

h_l, l_l = ax1.get_legend_handles_labels()

hdls = [h_l[2], h_l[3], h_l[1], h_l[0]]
lbls = [l_l[2], l_l[3], l_l[1], l_l[0]]

leg1 = ax1.legend(hdls, lbls, loc='upper right', ncol=4)
leg1.get_frame().set_alpha(0.0)

ax1.grid()

ax1.set_ylim([5e8, 1e14])

ax1.set_xlabel('iteration')
ax1.set_ylabel(r'$\mathscr{I}$')

ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)

yloc1 = plt.MaxNLocator(4)
ax1.yaxis.set_major_locator(yloc1)
ax1.set_yscale('log', basey=10)


plt.tight_layout()
plt.savefig(out_dir + 'convergence_plot.pdf')
plt.close(fig)



