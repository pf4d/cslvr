from varglas     import *
from pylab       import *
from fenics      import *
from numpy       import loadtxt, array
import os

out_dir = 'dump/stokes/fast_bfgs_honed/'

#hs = [1000, 2000, 4000, 8000, 16000, 32000]
#Hs = [250,  500,  750,  1000, 2000,  3000]
#Gs = [0.1,  0.25, 0.5,  1,    2,     4]
#
#for h in hs:
#  for H in Hs:
#    for g in Gs:
#      pass

n     = 25
h     = 1.0
g     = 0.5

H     = 1000.0
L     = n*h
alpha = g * pi / 180

p1    = Point(0.0, 0.0, 0.0)
p2    = Point(L,   L,   1)
mesh  = BoxMesh(p1, p2, 25, 25, 10)

model = D3Model(out_dir = out_dir + 'initial/')
model.set_mesh(mesh)
model.generate_function_spaces(use_periodic = True)

surface = Expression('- x[0] * tan(alpha)', alpha=alpha, 
                     element=model.Q.ufl_element())
bed     = Expression('- x[0] * tan(alpha) - H', alpha=alpha, H=H, 
                     element=model.Q.ufl_element())

model.calculate_boundaries()
model.deform_mesh_to_geometry(surface, bed)

model.init_S(surface)
model.init_B(bed)
model.init_mask(0.0)  # all grounded

lg      = LagrangeInterpolator()
bedmesh = model.get_bed_mesh()
srfmesh = model.get_surface_mesh()

Q_b     = FunctionSpace(bedmesh, 'CG', 1)
Q_s     = FunctionSpace(srfmesh, 'CG', 1)

beta_b  = Function(Q_b)
U_ob_s  = Function(Q_s)
U_s     = Function(Q_s)

model.init_beta(out_dir + 'initial/xml/beta_true.xml')
model.init_U(out_dir + 'initial/xml/u_true.xml',
             out_dir + 'initial/xml/v_true.xml',
             out_dir + 'initial/xml/w_true.xml')
model.assign_variable(model.U_ob, out_dir + 'initial/xml/U_ob.xml')

lg.interpolate(U_ob_s, model.U_ob)
lg.interpolate(beta_b, model.beta)
lg.interpolate(U_s,    model.U_mag)

umin = U_s.vector().min()
umax = U_s.vector().max()

betamin = beta_b.vector().min()
betamax = beta_b.vector().max()


plot_variable(beta_b, name='beta_true', direc=out_dir + 'plot/initial/', 
              cmap='gist_yarg', scale='lin', numLvls=12,
              umin=betamin, umax=betamax, tp=True, tpAlpha=0.5, show=False,
              hide_ax_tick_labels=False, label_axes=True, 
              title=r'$\beta$',
              use_colorbar=True, hide_axis=False, colorbar_loc='right')
plot_variable(U_s, name='U_true', direc=out_dir + 'plot/initial/', 
              cmap='gist_yarg', scale='lin', numLvls=12,
              umin=umin, umax=umax, tp=True, tpAlpha=0.5, show=False,
              hide_ax_tick_labels=False, label_axes=True, 
              title=r'$\Vert \mathbf{u}_{\mathrm{true}} \Vert$',
              use_colorbar=True, hide_axis=False, colorbar_loc='right')
plot_variable(U_ob_s, name='U_ob', direc=out_dir + 'plot/initial/', 
              cmap='gist_yarg', scale='lin', numLvls=12,
              umin=umin, umax=umax, tp=True, tpAlpha=0.5, show=False,
              hide_ax_tick_labels=False, label_axes=True, 
              title=r'$\Vert \mathbf{u}_{\mathrm{ob}} \Vert$',
              use_colorbar=True, hide_axis=False, colorbar_loc='right')

for d in os.listdir(out_dir + 'assimilated'):
  di = out_dir + 'assimilated/' + d + '/xml/'
  do = out_dir + 'plot/assimilated/'
  model.init_beta(di + 'beta_opt.xml')
  model.init_U(di + 'u_opt.xml',
               di + 'v_opt.xml',
               di + 'w_opt.xml')

  lg.interpolate(beta_b, model.beta)
  lg.interpolate(U_s,    model.U_mag)

  plot_variable(beta_b, name='beta_opt_' + d, direc=do + 'beta_opt/', 
                cmap='gist_yarg', scale='lin', numLvls=12,
                umin=betamin, umax=betamax, tp=True, tpAlpha=0.5, show=False,
                hide_ax_tick_labels=False, label_axes=True, 
                title=r'$\beta_{\mathrm{opt}}$',
                use_colorbar=True, hide_axis=False, colorbar_loc='right')
  plot_variable(U_s, name='U_opt_' + d, direc=do + 'U_opt/', 
                cmap='gist_yarg', scale='lin', numLvls=12,
                umin=umin, umax=umax, tp=True, tpAlpha=0.5, show=False,
                hide_ax_tick_labels=False, label_axes=True, 
                title=r'$\Vert \mathbf{u}_{\mathrm{opt}} \Vert$',
                use_colorbar=True, hide_axis=False, colorbar_loc='right')


di = out_dir + 'plot/'
do = out_dir + 'plot/L_curve/'

R  = loadtxt(di + 'Rs.txt')
J  = loadtxt(di + 'Js.txt')
a  = loadtxt(di + 'as.txt')


fig = figure()
ax  = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$\ln\left( \mathscr{R}\left(\alpha\right) \right)$')
ax.set_xlabel(r'$\ln\left( \mathscr{J}\left(\alpha\right) \right)$')

ax.plot(J, R, 'ko-', lw=2.0)
grid()
savefig(do + 'L_curve.png', dpi=200)
show()
