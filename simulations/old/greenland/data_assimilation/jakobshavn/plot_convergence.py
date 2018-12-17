from numpy import *
import matplotlib.pyplot as plt
import matplotlib        as mpl
import os

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'xx-small'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']

# set the relavent directories :
in_dir_1 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_1/'
in_dir_2 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'
in_dir_3 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100' + \
           '_disc_kappa/'
in_dir_4 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_1' + \
           '_disc_kappa/'
out_dir  = 'dump/jakob_small/convergence/'

# plot convergence :
abs_err_1 = loadtxt(in_dir_1 + 'tmc/convergence_history/abs_err.txt')
tht_nrm_1 = loadtxt(in_dir_1 + 'tmc/convergence_history/theta_norm.txt')
abs_err_2 = loadtxt(in_dir_2 + 'tmc/convergence_history/abs_err.txt')
tht_nrm_2 = loadtxt(in_dir_2 + 'tmc/convergence_history/theta_norm.txt')
abs_err_3 = loadtxt(in_dir_3 + 'tmc/convergence_history/abs_err.txt')
tht_nrm_3 = loadtxt(in_dir_3 + 'tmc/convergence_history/theta_norm.txt')
abs_err_4 = loadtxt(in_dir_4 + 'tmc/convergence_history/abs_err.txt')
tht_nrm_4 = loadtxt(in_dir_4 + 'tmc/convergence_history/theta_norm.txt')


def get_data(direc):
  Ds = []
  Js = []
  Rs = []
  Is = []
  ns = []
  
  for d in next(os.walk(direc + 'tmc/'))[1]:
    try:
      i = int(d)
      dn = direc + 'tmc/' + d + '/objective_ftnls_history/'
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

  return (Is, Ds, Js, Rs, ns, xn)


out_1 = get_data(in_dir_1)
out_2 = get_data(in_dir_2)
out_3 = get_data(in_dir_3)
out_4 = get_data(in_dir_4)

Is_1, Ds_1, Js_1, Rs_1, ns_1, xn_1 = out_1
Is_2, Ds_2, Js_2, Rs_2, ns_2, xn_2 = out_2
Is_3, Ds_3, Js_3, Rs_3, ns_3, xn_3 = out_3
Is_4, Ds_4, Js_4, Rs_4, ns_4, xn_4 = out_4

Rmax_1 = Rs_1.max()
Jmax_1 = Js_1.max()
amax_1 = abs_err_1.max()
tmax_1 = tht_nrm_1.max()

Rmax_2 = Rs_2.max()
Jmax_2 = Js_2.max()
amax_2 = abs_err_2.max()
tmax_2 = tht_nrm_2.max()

Rmax_3 = Rs_3.max()
Jmax_3 = Js_3.max()
amax_3 = abs_err_3.max()
tmax_3 = tht_nrm_3.max()

Rmax_4 = Rs_4.max()
Jmax_4 = Js_4.max()
amax_4 = abs_err_4.max()
tmax_4 = tht_nrm_4.max()

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(211)
ax2 = ax1.twinx()
ax3 = fig.add_subplot(212)
ax4 = ax3.twinx()

#ax1 = fig.add_subplot(411)
#ax2 = fig.add_subplot(412)
#ax3 = fig.add_subplot(413)
#ax4 = fig.add_subplot(414)

k = 0

dkred  = '#820000'
pink   = '#ff8e8e'
ltpink = '#ffbaba'

ax3.plot(xn_3, abs_err_3, 'o-', c='0.8', lw=2.0,
         label=r'$\alpha_m = 100, k_0 = 1e3$')
ax4.plot(xn_3, tht_nrm_3, 'o-', c=ltpink,  lw=2.0,
         label=r'$\alpha_m = 100, k_0 = 1e3$')

ax3.plot(xn_4, abs_err_4, 'o-', c='0.6', lw=2.0,
         label=r'$\alpha_m = 1, k_0 = 1e3$')
ax4.plot(xn_4, tht_nrm_4, 'o-', c=pink,  lw=2.0,
         label=r'$\alpha_m = 1, k_0 = 1e3$')

ax3.plot(xn_2, abs_err_2, 'o-', c='0.4', lw=2.0,
         label=r'$\alpha_m = 100, k_0 = 1$')
ax4.plot(xn_2, tht_nrm_2, 'o-', c='r',   lw=2.0,
         label=r'$\alpha_m = 100, k_0 = 1$')

ax3.plot(xn_1, abs_err_1, 'o-', c='k',   lw=2.0,
         label=r'$\alpha_m = 1, k_0 = 1$')
ax4.plot(xn_1, tht_nrm_1, 'o-', c=dkred, lw=2.0,
         label=r'$\alpha_m = 1, k_0 = 1$')


for i in range(len(Is_1)):
  xi = arange(k, k + ns_1[i])
  
  ax1.plot(xi, Js_3[i], '-', c='0.8',  lw=2.0)
  ax2.plot(xi, Rs_3[i], '-', c=ltpink, lw=2.0)
  
  ax1.plot(xi, Js_4[i], '-', c='0.6', lw=2.0)
  ax2.plot(xi, Rs_4[i], '-', c=pink,  lw=2.0)
  
  ax1.plot(xi, Js_2[i], '-', c='0.4', lw=2.0)
  ax2.plot(xi, Rs_2[i], '-', c='r',   lw=2.0)
  
  ax1.plot(xi, Js_1[i], '-', c='k',   lw=2.0)
  ax2.plot(xi, Rs_1[i], '-', c=dkred, lw=2.0)

  k += ns_1[i] - 1

leg1 = ax3.legend(loc='upper center')
leg2 = ax4.legend(loc='upper right')
leg1.get_frame().set_alpha(0.0)
leg2.get_frame().set_alpha(0.0)

ax1.grid()
#ax2.grid()
ax3.grid()
#ax4.grid()

ax3.set_xlim([0, max(xn_1.max(), xn_2.max())])
ax4.set_xlim([0, max(xn_1.max(), xn_2.max())])
#ax1.set_ylim([9.75e18, 9.3e18])
ax2.set_ylim([1e6, 1e8])

ax1.set_xticklabels([])
#ax2.set_xticklabels([])
#ax3.set_xticklabels([])

#ax1.set_xlabel('iteration')
#ax2.set_xlabel('iteration')
ax3.set_xlabel('iteration')
#ax4.set_xlabel('iteration')
ax1.set_ylabel(r'$\mathscr{J}$')
ax2.set_ylabel(r'$\mathscr{R}$')
ax3.set_ylabel(r'$\Vert \theta_n - \theta_{n-1} \Vert$')
ax4.set_ylabel(r'$\Vert \theta_n \Vert$')

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

ax2.set_yscale('log')
ax3.set_yscale('log')
ax2.yaxis.get_offset_text().set_color('r')
ax2.yaxis.label.set_color('r')
ax4.yaxis.get_offset_text().set_color('r')
ax4.yaxis.label.set_color('r')

plt.tight_layout()
plt.savefig(out_dir + 'convergence_plot.pdf')
plt.close(fig)



