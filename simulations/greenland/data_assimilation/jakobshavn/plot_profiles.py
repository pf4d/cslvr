import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'small'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']

# set the relavent directories :
base_dir_1 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_1/'
base_dir_2 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100/'
base_dir_3 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_100' + \
             '_disc_kappa/tmc/10/'
base_dir_4 = 'dump/jakob_small/rstrt_alpha_1e8_regularized_FS_Tp_a_0_1' + \
             '_disc_kappa/tmc/10/'
out_dir    = 'dump/jakob_small/profile/'

T_1 = np.loadtxt(base_dir_1 + 'profile_data/T.txt')
W_1 = np.loadtxt(base_dir_1 + 'profile_data/W.txt')
z_1 = np.loadtxt(base_dir_1 + 'profile_data/z.txt')

T_2 = np.loadtxt(base_dir_2 + 'profile_data/T.txt')
W_2 = np.loadtxt(base_dir_2 + 'profile_data/W.txt')
z_2 = np.loadtxt(base_dir_2 + 'profile_data/z.txt')

T_3 = np.loadtxt(base_dir_3 + 'profile_data/T.txt')
W_3 = np.loadtxt(base_dir_3 + 'profile_data/W.txt')
z_3 = np.loadtxt(base_dir_3 + 'profile_data/z.txt')

T_4 = np.loadtxt(base_dir_4 + 'profile_data/T.txt')
W_4 = np.loadtxt(base_dir_4 + 'profile_data/W.txt')
z_4 = np.loadtxt(base_dir_4 + 'profile_data/z.txt')

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#ax3 = ax2.twiny()

#W_d_c = W_2 - W_3
#W_d_d = W_1 - W_4
#
#ax3.plot(W_d_c, z_1, 'b-',  lw=3.0)
#ax3.plot(W_d_d, z_1, 'g--', lw=3.0)
ax1.plot(T_1, z_1, 'r-', lw=3.0)
ax2.plot(W_1, z_1, 'r-', lw=3.0, label=r'$\alpha \approx 1.0, k_0=1$')
ax1.plot(T_4, z_4, 'g--', lw=3.0)
ax2.plot(W_4, z_4, 'g--', lw=3.0, label=r'$\alpha \approx 1.0, k_0=1e3$')
ax1.plot(T_2, z_2, 'k-', lw=3.0)
ax2.plot(W_2, z_2, 'k-', lw=3.0, label=r'$\alpha \approx 8.58, k_0=1$')
ax1.plot(T_3, z_3, 'c--', lw=3.0)
ax2.plot(W_3, z_3, 'c--', lw=3.0, label=r'$\alpha \approx 8.58, k_0=1e3$')

ax2.legend(loc='upper right')

#plt.subplots_adjust(wspace = 0.001)
ax2.set_yticklabels([])
ax2.set_xlim([0,0.16])

xloc1 = plt.MaxNLocator(4)
xloc2 = plt.MaxNLocator(4)
ax1.xaxis.set_major_locator(xloc1)
ax2.xaxis.set_major_locator(xloc2)

#ax3.set_xticks([-1.5e-3, 0.0, 0.5e-3])
ax1.set_yticks([0,-50,-100,-150,-200,-250,-280])
ax1.set_ylim([-280, 0])
ax2.set_ylim([-280, 0])

ax1.set_xlabel(r'$T$')
ax2.set_xlabel(r'$W$')
#ax3.set_xlabel(r'$W_{k} - W_{\kappa}$')
ax1.set_ylabel(r'depth')

#ax3.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1g'))
#ax3.ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useOffset=False)
#ax3.set_xticklabels(['-1.5e-3', '0', '5e-4'])
#ax3.tick_params(axis='x', colors='b')

#ax3.set_xscale('log')
#ax3.xaxis.get_offset_text().set_color('b')
#ax3.xaxis.label.set_color('b')

ax1.grid()
ax2.grid()
plt.tight_layout()
plt.savefig(out_dir + 'profile_plot.pdf')
plt.close(fig)



