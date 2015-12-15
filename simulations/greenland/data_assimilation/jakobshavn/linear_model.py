from pylab             import *
from scipy.stats       import t, distributions, scoreatpercentile, \
                              distributions, probplot, chisquare
from scipy.special     import fdtrc
from scipy.sparse      import diags
from scipy.interpolate import interp1d

import sys
import os

from varglas           import *
#from fenics            import *
from pylab             import *

lognorm  = distributions.lognorm

def iqr(arr):
  arr           = sort(arr.copy())
  upperQuartile = scoreatpercentile(arr,.75)
  lowerQuartile = scoreatpercentile(arr,.25)
  iqr           = upperQuartile - lowerQuartile
  return iqr

def normal(x, mu, sigma):
  """ 
  Function which returns the normal value at <x> with mean <mu> and 
  standard deviation <sigma>.
  """
  return 1.0/(sigma * sqrt(2.0 * pi)) * exp(-(x - mu)**2 / (2.0 * sigma**2))

def glm(x,y,w=1.0):

  p,n    = shape(x)                    # sample size
  p     += 1                           # add one for intercept
  dof    = n - p                       # degrees of freedom
  
  sig    = var(y)                      # variance
  mu     = (y + mean(y))/2.0           # initial mean estimate
  eta    = log(mu)                     # initial predictor
  X      = vstack((ones(n), x)).T      # observed x-variable matrix

  # Newton-Raphson :
  converged = False
  rtol      = 1e-12
  dtol      = 1e-12
  lmbda     = 1.0
  nIter     = 0
  deviance  = 1
  D         = 1
  ahat      = zeros(p)   # initial parameters
  rel_res   = zeros(p)   # initial relative residual
  maxIter   = 350

  rel_a = []
  dev_a = []

  while not converged and nIter < maxIter:
    W       = diags(w*mu**2/sig, 0)         # compute weights
    z       = eta + (y - mu)/mu             # adjusted dependent variable

    WX      = W.dot(X)
    XTWX    = dot(X.T, WX)
    iXTWX   = inv(XTWX)
    Wz      = W.dot(z)

    ahat_n  = dot(iXTWX, dot(X.T, Wz))
    
    eta     = dot(X, ahat_n)               # compute estimates
    mu      = exp(eta)                     # linear predictor

    # calculate residual :
    rel_res  = norm(ahat - ahat_n, inf)
    rel_a.append(rel_res)
    ahat     = ahat_n

    D_n      = sum((y - mu)**2)
    deviance = abs(D_n - D)
    D        = D_n
    dev_a.append(deviance)
    
    if rel_res < rtol or deviance < dtol: converged = True
    nIter +=  1

    string = "Newton iteration %d: d (abs) = %.2e, (tol = %.2e) r (rel) = %.2e (tol = %.2e)"
    print string % (nIter, deviance, dtol, rel_res, rtol)
  
  # calculate statistics :
  varA   = diag(iXTWX)            # variance of alpha hat
  sea    = sqrt(varA)             # vector of standard errors for alpha hat
  t_a    = ahat / sea
  pval   = t.sf(abs(t_a), dof) * 2
  conf   = 0.95                        # 95% confidence interval
  tbonf  = t.ppf((1 - conf/p), dof)    # bonferroni corrected t-value
  ci     = tbonf*sea                   # confidence interval for ahat
  resid  = (y - mu)                    # 'working' residual
                                       
  RSS    = sum((y - mu)**2)            # residual sum of squares
  TSS    = sum((y - mean(y))**2)       # total sum of squares
  R2     = (TSS-RSS)/TSS               # R2
  F      = (TSS-RSS)/(p-1) * (n-p)/RSS # F-statistic
  F_p    = fdtrc(p-1, dof, F)          # F-Stat. p-value

  # log-likelihood :
  L      = sum((y*mu - mu**2/2)/(2*sig) - y**2/(2*sig) - 0.5*log(2*pi*sig))
  AIC    = (-2*L + 2*p)/n              # AIC statistic

  # estimated error variance :
  sighat = 1/(n-p) * RSS
                                        
  vara = { 'ahat'  : ahat,              
           'yhat'  : mu,                
           'sea'   : sea,               
           'ci'    : ci,                
           'dof'   : dof,               
           'resid' : resid,             
           'rel_a' : rel_a,
           'dev_a' : dev_a,
           'R2'    : R2,
           'F'     : F,
           'AIC'   : AIC,
           'sighat': sighat}
  return vara


#===============================================================================
# create directories and such :

mdl = sys.argv[1]
typ = sys.argv[2]

if typ == 'weighted':
  file_n = mdl + '/weighted/'

elif typ == 'limited':
  file_n = mdl + '/limited_weighted/'

else:
  file_n = mdl + '/normal/'

print file_n

g_fn = 'images/stats/' + file_n

dirs = [g_fn, 'dat/' + file_n]

for di in dirs:
  if not os.path.exists(di):
    os.makedirs(di)


#===============================================================================
# get the data from the model output on the bed :

# set the relavent directories :
dir_b   = 'dump/jakob_small/'
var_dir = 'dump/vars_jakobshavn_small/'       # directory from gen_vars.py
in_dir  = dir_b + '01/hdf5/'                  # input dir
out_dir = 'plot/01/'                          # base directory to save

# create HDF5 files for saving and loading data :
fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',       'r')
fthermo = HDF5File(mpi_comm_world(), in_dir  + 'thermo_01.h5',   'r')
finv    = HDF5File(mpi_comm_world(), in_dir  + 'inverted_01.h5', 'r')
fstress = HDF5File(mpi_comm_world(), in_dir  + 'stress_01.h5',   'r')

# not deformed mesh :
mesh    = Mesh('dump/meshes/jakobshavn_3D_small_block.xml.gz')

# create 3D model for stokes solves :
d3model = D3Model(mesh, out_dir)

# init subdomains and boundary meshes :
d3model.set_subdomains(fdata)

# initialize the 3D model vars :
d3model.init_S(fdata)
d3model.init_B(fdata)
d3model.init_mask(fdata)
d3model.init_T_surface(fdata)
d3model.init_adot(fdata)
d3model.init_U_ob(fdata, fdata)
d3model.init_U_mask(fdata)
d3model.init_beta(finv)
d3model.init_U(finv)
d3model.init_T(fthermo)
d3model.init_W(fthermo)
d3model.init_theta(fthermo)
d3model.init_Mb(fthermo)
d3model.init_tau_id(fstress)
d3model.init_tau_jd(fstress)
d3model.init_tau_ii(fstress)
d3model.init_tau_ij(fstress)
d3model.init_tau_iz(fstress)
d3model.init_tau_ji(fstress)
d3model.init_tau_jj(fstress)
d3model.init_tau_jz(fstress)


#===============================================================================
# retrieve the bed mesh :
bedmesh = d3model.get_bed_mesh()
srfmesh = d3model.get_surface_mesh()

# create 2D model for balance velocity :
d2model = D2Model(bedmesh, out_dir)

# 2D model gets balance-velocity appropriate variables initialized :
d2model.assign_submesh_variable(d2model.S,         d3model.S)
d2model.assign_submesh_variable(d2model.B,         d3model.B)
d2model.assign_submesh_variable(d2model.T_surface, d3model.T_surface)
d2model.assign_submesh_variable(d2model.adot,      d3model.adot)
d2model.assign_submesh_variable(d2model.u_ob,      d3model.u_ob)
d2model.assign_submesh_variable(d2model.v_ob,      d3model.v_ob)
d2model.assign_submesh_variable(d2model.U_ob,      d3model.U_ob)
d2model.assign_submesh_variable(d2model.beta,      d3model.beta)
d2model.assign_submesh_variable(d2model.U3,        d3model.U3)
d2model.assign_submesh_variable(d2model.U_mag,     d3model.U_mag)
d2model.assign_submesh_variable(d2model.T,         d3model.T)
d2model.assign_submesh_variable(d2model.W,         d3model.W)
d2model.assign_submesh_variable(d2model.Mb,        d3model.Mb)
d2model.assign_submesh_variable(d2model.tau_id,    d3model.tau_id)
d2model.assign_submesh_variable(d2model.tau_jd,    d3model.tau_jd)
d2model.assign_submesh_variable(d2model.tau_ii,    d3model.tau_ii)
d2model.assign_submesh_variable(d2model.tau_ij,    d3model.tau_ij)
d2model.assign_submesh_variable(d2model.tau_iz,    d3model.tau_iz)
d2model.assign_submesh_variable(d2model.tau_ji,    d3model.tau_ji)
d2model.assign_submesh_variable(d2model.tau_jj,    d3model.tau_jj)
d2model.assign_submesh_variable(d2model.tau_jz,    d3model.tau_jz)

# create a new 2D model for surface variables :
srfmodel = D2Model(srfmesh, out_dir)

# put the velocity on it :
d2model.assign_submesh_variable(srfmodel.U3,        d3model.U3)
d2model.assign_submesh_variable(srfmodel.U_mag,     d3model.U_mag)

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=5.0)
bv.solve(annotate=False)

d2model.Ubar5 = d2model.Ubar
d2model.Ubar5.rename('Ubar5', '')

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=10.0)
bv.solve(annotate=False)

d2model.Ubar10 = d2model.Ubar
d2model.Ubar10.rename('Ubar10', '')

# solve the balance velocity :
bv = BalanceVelocity(d2model, kappa=20.0)
bv.solve(annotate=False)

d2model.Ubar20 = d2model.Ubar
d2model.Ubar20.rename('Ubar20', '')


#===============================================================================
# greenland :

Q      = d2model.Q

u,v,w  = d2model.U3.split(True)
S      = d2model.S 
B      = d2model.B
adot   = d2model.adot
qgeo   = d2model.q_geo
beta   = d2model.beta
Mb     = d2model.Mb
Tb     = d2model.T
Ts     = d2model.T_surface
Ubar5  = d2model.Ubar5
Ubar10 = d2model.Ubar10
Ubar20 = d2model.Ubar20
U_ob   = d2model.U_ob
tau_id = d2model.tau_id
tau_jd = d2model.tau_jd
tau_ii = d2model.tau_ii
tau_ij = d2model.tau_ij
tau_iz = d2model.tau_iz
tau_ji = d2model.tau_ji
tau_jj = d2model.tau_jj
tau_jz = d2model.tau_jz
mask   = d2model.mask

dSdx   = project(S.dx(0), Q)
dSdy   = project(S.dx(1), Q)

dBdx   = project(B.dx(0), Q)
dBdy   = project(B.dx(1), Q)
                        
dHdx   = project((S - B).dx(0), Q)
dHdy   = project((S - B).dx(1), Q)

# vectors :
beta_v   = beta.vector().array()
S_v      = S.vector().array()
B_v      = B.vector().array()
adot_v   = adot.vector().array()
qgeo_v   = qgeo.vector().array()
Mb_v     = Mb.vector().array()
Tb_v     = Tb.vector().array()
Ts_v     = Ts.vector().array()
u_v      = u.vector().array()
v_v      = v.vector().array()
w_v      = w.vector().array()
Ubar5_v  = Ubar5.vector().array()
Ubar10_v = Ubar10.vector().array()
Ubar20_v = Ubar20.vector().array()
U_ob_v   = U_ob.vector().array()
tau_id_v = tau_id.vector().array()
tau_jd_v = tau_jd.vector().array()
tau_ii_v = tau_ii.vector().array()
tau_ij_v = tau_ij.vector().array()
tau_iz_v = tau_iz.vector().array()
tau_ji_v = tau_ji.vector().array()
tau_jj_v = tau_jj.vector().array()
tau_jz_v = tau_jz.vector().array()
mask_v   = mask.vector().array()

H_v    = S_v - B_v
dHdx_v = dHdx.vector().array()
dHdy_v = dHdy.vector().array()
gradH  = sqrt(dHdx_v**2 + dHdy_v**2 + 1e-16)
U_mag  = sqrt(u_v**2 + v_v**2 + 1e-16)
dSdx_v = dSdx.vector().array()
dSdy_v = dSdy.vector().array()
gradS  = sqrt(dSdx_v**2 + dSdy_v**2 + 1e-16)
dBdx_v = dBdx.vector().array()
dBdy_v = dBdy.vector().array()
gradB  = sqrt(dBdx_v**2 + dBdy_v**2 + 1e-16)
D      = zeros(len(B_v))
D[B_v < 0] = B_v[B_v < 0]

taux = -917.0 * 9.8 * H_v * dSdx_v
tauy = -917.0 * 9.8 * H_v * dSdy_v
tau_mag = sqrt(taux**2 + tauy**2 + 1e-16)

#if mdl == 'Ubar' or mdl == 'Ubar_temp':
#  uhat = taux / tau_mag
#  vhat = tauy / tau_mag
#
#elif mdl == 'U' or mdl == 'stress' or mdl == 'U_temp':
#  uhat = u_v / U_mag
#  vhat = v_v / U_mag
uhat = u_v / U_mag
vhat = v_v / U_mag

dBdi = dBdx_v * uhat + dBdy_v * vhat
dBdj = dBdx_v * vhat - dBdy_v * uhat

dSdi = dSdx_v * uhat + dSdy_v * vhat
dSdj = dSdx_v * vhat - dSdy_v * uhat

dHdi = dHdx_v * uhat + dHdy_v * vhat
dHdj = dHdx_v * vhat - dHdy_v * uhat

Ubar_avg = (Ubar5_v + Ubar10_v + Ubar20_v) / 3.0


if mdl == 'Ubar' or mdl == 'Ubar_temp':
  ini_i    = 917.0 * 9.8 * H_v * dSdi / (Ubar5_v + 0.1)
  ini_j    = 917.0 * 9.8 * H_v * dSdj / (Ubar5_v + 0.1)

elif mdl == 'U' or mdl == 'stress' or mdl == 'U_temp':
  ini_i    = 917.0 * 9.8 * H_v * dSdi / (U_mag + 0.1)
  ini_j    = 917.0 * 9.8 * H_v * dSdj / (U_mag + 0.1)

# areas of cells for weighting :
h_v  = project(CellSize(d2model.mesh), Q).vector().array()

# number of dofs :
n = len(beta_v)

#===============================================================================
# remove areas with garbage data :
valid  = where(mask_v < 1.0)[0]
valid  = intersect1d(valid, where(S_v > 0.0)[0])
#valid  = intersect1d(valid, where(beta_v < 1000)[0])
#valid  = intersect1d(valid, where(beta_v > 1e-4)[0])
if typ == 'limited':
  valid  = intersect1d(valid, where(U_mag > 20)[0])
else:
  valid  = intersect1d(valid, where(U_mag > 0)[0])
valid  = intersect1d(valid, where(U_ob_v > 1e-9)[0])
valid  = intersect1d(valid, where(Ts_v > 100)[0])
valid  = intersect1d(valid, where(h_v > 0)[0])
valid  = intersect1d(valid, where(S_v - B_v > 60)[0])
valid  = intersect1d(valid, where(adot_v > -100)[0])
#valid  = intersect1d(valid, where(gradS < 0.05)[0])
#valid  = intersect1d(valid, where(gradB < 0.2)[0])
#valid  = intersect1d(valid, where(Mb_v < 0.04)[0])
#valid  = intersect1d(valid, where(Mb_v > 0.0)[0])
#valid  = intersect1d(valid, where(adot_v < 1.2)[0])
#valid  = intersect1d(valid, where(adot_v > -1.0)[0])

#===============================================================================
# individual regions for plotting :

valid_f          = Function(Q)
valid_f_v        = valid_f.vector().array()
valid_f_v[valid] = 1.0
valid_f.vector().set_local(valid_f_v)
valid_f.vector().apply('insert')

drg     = DataFactory.get_rignot()

#===============================================================================
#
#plotIce(drg, valid_f, name='valid', direc=g_fn, basin='jakobshavn',
#        cmap='gist_yarg', scale='bool', numLvls=12, tp=False,
#        tpAlpha=0.5, show=False)
#
#dBdi_f = Function(Q)
#dBdj_f = Function(Q)
#dBdi_f.vector().set_local(dBdi)
#dBdi_f.vector().apply('insert')
#dBdj_f.vector().set_local(dBdj)
#dBdj_f.vector().apply('insert')
#
#plotIce(drg, dBdi_f, name='dBdi', direc=g_fn, basin='jakobshavn',
#        title=r'$\partial_i B$', cmap='RdGy', scale='lin', extend='max',
#        umin=-0.1, umax=0.1, numLvls=12, tp=False, tpAlpha=0.5, show=False)
#
#plotIce(drg, dBdj_f, name='dBdj', direc=g_fn, basin='jakobshavn',
#        title=r'$\partial_j B$', cmap='RdGy', scale='lin', extend='max',
#        umin=-0.1, umax=0.1, numLvls=12, tp=False, tpAlpha=0.5, show=False)
#
#dSdi_f = Function(Q)
#dSdj_f = Function(Q)
#dSdi_f.vector().set_local(dSdi)
#dSdi_f.vector().apply('insert')
#dSdj_f.vector().set_local(dSdj)
#dSdj_f.vector().apply('insert')
#
#plotIce(drg, dSdi_f, name='dSdi', direc=g_fn, basin='jakobshavn',
#        title=r'$\partial_i S$', cmap='RdGy', scale='lin', extend='max',
#        umin=-0.1, umax=0.1, numLvls=12, tp=False, tpAlpha=0.5, show=False)
#
#plotIce(drg, dSdj_f, name='dSdj', direc=g_fn, basin='jakobshavn',
#        title=r'$\partial_j S$', cmap='RdGy', scale='lin', extend='max',
#        umin=-0.1, umax=0.1, numLvls=12, tp=False, tpAlpha=0.5, show=False)
#
#===============================================================================
# cell declustering :
n      = len(valid)
h_v    = h_v[valid]
A      = sum(h_v)

wt     = n * h_v / A

#h_v      = h_v[valid]
#A        = sum(h_v)
#wt       = n * h_v / A
#beta_bar = 1.0/n * sum(beta_v[valid] * wt)

#===============================================================================
#data = [beta_v,  S_v,     B_v,    gradS,   gradB, 
#        H_v,     adot_v,  Ts_v,   Tb_v,    Mb_v,
#        Ubar5_v, u_v,     v_v,    w_v,     U_mag]
#names = [r'$\beta$',
#         r'$S$',
#         r'$D$',
#         r'$\Vert \nabla S \Vert$', 
#         r'$\Vert \nabla B \Vert$', 
#         r'$H$',
#         r'$\dot{a}$',
#         r'$T_S$', 
#         r'$T_B$', 
#         r'$M_B$', 
#         r'$\Vert \bar{\mathbf{u}}_{bv} \Vert$',
#         r'$u$', 
#         r'$v$', 
#         r'$w$', 
#         r'$\Vert \mathbf{u}_B \Vert$']
#
#fig = figure(figsize=(25,15))
#for k,(n,d) in enumerate(zip(names, data)):
#  ax = fig.add_subplot(4,4,k+1)
#  m, bins, pat = hist(d[valid], 1000, normed=1, histtype='stepfilled')
#  setp(pat, 'facecolor', 'b', 'alpha', 0.75)
#  ax.set_xlabel(n)
#  ax.set_ylabel(r'$n$')
#  ax.grid()
#fn = 'images/data.png'
##savefig(fn, dpi=100)
#show()

#===============================================================================
# data analysis :

mtx     = 8.0
betaMax = beta_v[valid].max()**(1/mtx)
y       = (beta_v[valid])**(1/mtx)

#===============================================================================
# do the glm :

v0   = S_v
v1   = Ts_v
v2   = gradS
v3   = D
v4   = gradB
v5   = H_v
v6   = qgeo_v
v7   = adot_v
v8   = Tb_v
v9   = Mb_v
v10  = u_v
v11  = v_v
v12  = w_v
v13  = log(Ubar5_v + DOLFIN_EPS)
v14  = log(Ubar10_v + DOLFIN_EPS)
v15  = log(Ubar20_v + DOLFIN_EPS)
v16  = log(U_mag + DOLFIN_EPS)
v17  = tau_id_v
v18  = tau_jd_v
v19  = tau_ii_v
v20  = tau_ij_v
v21  = tau_iz_v
v22  = tau_ji_v
v23  = tau_jj_v
v24  = tau_jz_v
v25  = ini_i
v26  = ini_j
v27  = dBdi
v28  = dBdj
v29  = dSdi
v30  = dSdj
v31  = gradH
v32  = dHdi
v33  = dHdj

x0   = v0[valid]
x1   = v1[valid]
x2   = v2[valid]
x3   = v3[valid]
x4   = v4[valid]
x5   = v5[valid]
x6   = v6[valid]
x7   = v7[valid]
x8   = v8[valid]
x9   = v9[valid]
x10  = v10[valid]
x11  = v11[valid]
x12  = v12[valid]
x13  = v13[valid]
x14  = v14[valid]
x15  = v15[valid]
x16  = v16[valid]
x17  = v17[valid]
x18  = v18[valid]
x19  = v19[valid]
x20  = v20[valid]
x21  = v21[valid]
x22  = v22[valid]
x23  = v23[valid]
x24  = v24[valid]
x25  = v25[valid]
x26  = v26[valid]
x27  = v27[valid]
x28  = v28[valid]
x29  = v29[valid]
x30  = v30[valid]
x31  = v31[valid]
x32  = v32[valid]
x33  = v33[valid]

#===============================================================================
# formulte design matrix and do some EDA :
names = [r'$S$', 
         r'$T_S$', 
         r'$\Vert \nabla S \Vert$', 
         r'$D$',
         r'$\Vert \nabla B \Vert$', 
         r'$H$',
         r'$q_{geo}$',
         r'$\dot{a}$',
         r'$T_B$', 
         r'$M_B$', 
         r'$u$', 
         r'$v$', 
         r'$w$', 
         r'$\ln\left( \Vert \bar{\mathbf{u}}_{5} \Vert \right)$',
         r'$\ln\left( \Vert \bar{\mathbf{u}}_{10} \Vert \right)$',
         r'$\ln\left( \Vert \bar{\mathbf{u}}_{20} \Vert \right)$',
         r'$\ln\left( \Vert \mathbf{u}_B \Vert \right)$',
         r'$\tau_{id}$',
         r'$\tau_{jd}$',
         r'$\tau_{ii}$',
         r'$\tau_{ij}$',
         r'$\tau_{iz}$',
         r'$\tau_{ji}$',
         r'$\tau_{jj}$',
         r'$\tau_{jz}$',
         r'ini$_i$',
         r'ini$_j$',
         r'$\partial_i B$',
         r'$\partial_j B$',
         r'$\partial_i S$',
         r'$\partial_j S$',
         r'$\Vert \nabla H \Vert$',
         r'$\partial_i H$',
         r'$\partial_j H$']

X      = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,
          x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33]
V      = [v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,
          v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33]

# with stress terms :
if mdl == 'stress':
  index  = [0,1,5,7,16,27,29,17,19,20,22,23]

# U instead of Ubar :
elif mdl == 'U':
  index  = [0,1,5,7,16,27,29]

elif mdl == 'U_temp':
  index  = [0,1,5,7,8,9,16,27,29]

# independent only :
elif mdl == 'Ubar':
  index  = [0,1,5,7,13,27,29]

# independent only :
elif mdl == 'Ubar_temp':
  index  = [0,1,5,7,8,9,13,27,29]

ii     = index
ii_int = []
ii_int.extend(ii)

for i,m in enumerate(ii):
  if mdl == 'U' or mdl == 'Ubar':
    k = i
  else:
    k = i+1
  for j,n in enumerate(ii[k:]):
    ii_int.append([m,n])

#fig = figure(figsize=(25,15))
Xt   = []
Vt   = []
ex_n = []

for k,i in enumerate(ii_int):
  
  if type(i) == list:
    n = ''
    x = 1.0
    v = 1.0
    for jj,j in enumerate(i):
      x *= X[j]
      v *= V[j]
      n += names[j]
      if jj < len(i) - 1:
        n += r'$ \star $'
  else:
    x = X[i]
    v = V[i]
    n = names[i]
    #ax = fig.add_subplot(3,4,k+1)
    #ax.plot(x, y, 'ko', alpha=0.1)
    #ax.set_xlabel(n)
    #ax.set_ylabel(r'$\beta$')
    #ax.grid()

  ex_n.append(n)
  Xt.append(x)
  Vt.append(v)

ex_n.insert(0, '$\mathbf{1}$')
ex_n = array(ex_n)
 
#show()

#==============================================================================
# plot beta distribution and lognorm fit :

ln_fit   = lognorm.fit(y)
g_x      = linspace(y.min(), y.max(), 1000)
ln_freq  = lognorm.pdf(g_x, *ln_fit)

fig      = figure()
ax       = fig.add_subplot(111)

ax.hist(y, 300, histtype='stepfilled', color='k', alpha=0.5, normed=True,
        label=r'$\beta$')
ax.plot(g_x, ln_freq, lw=2.0, color='r', label=r'$\mathrm{LogNorm}$')
#ax.set_xlim([0,200])
##ax.set_ylim([0,0.020])
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Frequency')
ax.legend(loc='upper right')
ax.grid()
tight_layout()
fn = 'images/stats/' + file_n + 'beta_distribution.png'
savefig(fn, dpi=100)
#show()
close(fig)

#===============================================================================
# fit the glm :

if typ == 'weighted' or typ == 'limited':
  out   = glm(array(Xt), y, wt)
else:
  out   = glm(array(Xt), y)
yhat  = out['yhat']
resid = out['resid']
ahat  = out['ahat']
ci    = out['ci']

yhat_f = Function(Q)
resi_f = Function(Q)
yhat_f.vector()[valid] = yhat**mtx
resi_f.vector()[valid] = resid

cmap = 'RdGy'

plotIce(drg, yhat_f, name='GLM_beta', direc=g_fn, basin='jakobshavn', 
        title=r'$\hat{\beta}$', cmap=cmap, scale='log', extend='max',
        umin=1e-4, umax=1e4, numLvls=12, tp=False, tpAlpha=0.5, show=False)

plotIce(drg, resi_f, name='GLM_resid', direc=g_fn, basin='jakobshavn',
        title=r'$d$', cmap='RdGy', scale='lin', extend='both',
        umin=-1, umax=1, numLvls=13, tp=False, tpAlpha=0.5, show=False)

#===============================================================================
# data analysis :

fig      = figure()
ax       = fig.add_subplot(111)

ax.hist(y,    50, histtype='step', color='k', lw=1.5, alpha=1.0, normed=True,
        label=r'$\beta$')
ax.hist(yhat, 50, histtype='step', color='r', lw=1.5, alpha=1.0, normed=True,
        label=r'$\hat{\beta}$')
ax.set_xlim([0,betaMax])
#ax.set_ylim([0,0.03])
ax.set_xlabel(r'$\hat{\beta}$')
ax.set_ylabel('Frequency')
ax.legend(loc='upper right')
ax.grid()
tight_layout()
fn = 'images/stats/' + file_n + 'GLM_beta_distributions.png'
savefig(fn, dpi=100)
#show()
close(fig)
  
#=============================================================================
# residual plot and normal quantile plot for residuals :
fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#rtol  = 30
#Xr    = array(X)
#Xr    = Xr[:, resid < rtol]
#yhat  = yhat[resid < rtol]
#resid = resid[resid < rtol]

# Normal quantile plot of residuals
((osm,osr), (m, b, r)) = probplot(resid)
interp = interp1d(osm, osr)
yl     = interp(-2.5)
yh     = interp(2.5)
ax1.plot(osm, m*osm + b, 'r-', lw=2.0, label=r'LS fit')
ax1.plot(osm, osr,       'k.', alpha=1.0, label='$\mathbf{d}$')
ax1.set_xlabel('Standard Normal Quantiles')
ax1.set_ylabel('Residuals')
#ax1.set_title('Normal Quantile Plot')
ax1.set_xlim([-2.5, 2.5])
ax1.set_ylim([yl,   yh])
ax1.legend(loc='lower right')
ax1.grid()

ax2.plot(yhat, resid, 'k.', alpha=0.10)
ax2.set_xlabel(r'$\hat{\beta}$')
ax2.set_ylabel('Residuals')
#ax2.set_title('Residual Plot')
ax2.set_xlim([0,  betaMax])
ax2.set_ylim([yl, yh])
ax2.grid()

tight_layout()
fn = 'images/stats/' + file_n + 'GLM_resid_NQ.png'
savefig(fn, dpi=100)
#show()
close(fig)


#=============================================================================
# plot newton residuals :
fig = figure()
ax  = fig.add_subplot(111)

ax.plot(out['rel_a'], 'k-', lw=2.0,
        label=r'$\Vert \alpha - \alpha_n \Vert^2$')
ax.plot(out['dev_a'], 'r-', lw=2.0,
        label=r'$\Vert \mathbf{d} - \mathbf{d}_n \Vert^2$')
ax.set_xlabel(r'Iteration')
ax.set_yscale('log')
ax.set_xlim([0, len(out['dev_a'])-1])
ax.grid()
ax.legend()
fn = 'images/stats/' + file_n + 'GLM_newton_resid.png'
tight_layout()
savefig(fn, dpi=100)
#show()
close(fig)

##=============================================================================
## create partial-residual plot :
#fig  = figure(figsize=(25,15))
#s1   = int(ceil(sqrt(len(ii_int))))
#s2   = int(floor(sqrt(len(ii_int))))
#
#for k,i in enumerate(ii_int):
#  
#  if type(i) == list:
#    n = ''
#    x = 1.0
#    v = 1.0
#    for jj,j in enumerate(i):
#      x *= X[j]
#      v *= V[j]
#      n += names[j]
#      if jj < len(i) - 1:
#        n += r'$ \star $'
#  else:
#    x = X[i]
#    v = V[i]
#    n = names[i]
#   
#  alpha = ahat[k+1]
#  eta   = alpha*x
#  
#  ax = fig.add_subplot(s1,s1,k+1)
#  ax.plot(x, resid + eta, 'ko', alpha=0.1)
#  ax.set_xlabel(n)
#  ax.set_ylabel(r'Residual')
#  ax.grid()
#
#tight_layout()
#
#fn = 'images/stats/' + file_n + 'GLM_partial_residual.png'
#savefig(fn, dpi=100)
#show()
#close(fig)

#===============================================================================
# create tables :
n        = len(valid)

mu       = mean(y)                      # mean
med      = median(y)                    # median
sigma    = std(y)                       # standard deviation
fe_iqr   = iqr(y)                       # IQR
v_m_rat  = sigma**2 / mu                # variance-to-mean ratio
stats_y  = [mu, med, sigma**2, fe_iqr, v_m_rat]

mu       = mean(yhat)                  # mean
med      = median(yhat)                # median
sigma    = std(yhat)                   # standard deviation
fe_iqr   = iqr(yhat)                   # IQR
v_m_rat  = sigma**2 / mu               # variance-to-mean ratio
stats_yh = [mu, med, sigma**2, fe_iqr, v_m_rat, 
            out['R2'], out['F'], out['AIC'], out['sighat']]

#srt = argsort(abs(ahat))[::-1]
f   = open('dat/' + file_n + 'alpha.dat', 'wb')
#for n, a, c in zip(ex_n[srt], ahat[srt], ci[srt]):
for n, a, c in zip(ex_n, ahat, ci):
  al = a-c
  ah = a+c
  if sign(al) != sign(ah):
    strng = '\\color{red}%s & \\color{red}%.1e & \\color{red}%.1e & ' + \
            '\\color{red}%.1e \\\\\n'
  else:
    strng = '%s & %.1e & %.1e & %.1e \\\\\n'
  f.write(strng % (n, al, a, ah))
f.write('\n')
f.close()

names = ['$\mu$', 'median', '$\sigma^2$', 'IQR',   '$\sigma^2 / \mu$',
         '$R^2$', 'F',      'AIC',        '$\hat{\sigma}^2$']

f = open('dat/' + file_n + 'stats.dat', 'wb')
for n, s_yh in zip(names, stats_yh):
  strng = '%s & %g \\\\\n' % (n, s_yh)
  f.write(strng)
f.write('\n')
f.close()

#===============================================================================
sys.exit(0)
#===============================================================================
# reduce the model to explanitory variables with meaning :

ex_a   = array(ex_n)
ahat_n = ahat.copy()
ci_n   = ci.copy()
X_i    = array(Xt)

# find out how many to eliminate first:
v = []
for i,(a,c) in enumerate(zip(ahat_n, ci_n)):
  al = a-c
  ah = a+c
  if sign(al) == sign(ah):
    v.append(i)
v = array(v)
exterminated = len(ahat_n) - len(v)
print "eliminated %i fields" % exterminated

while exterminated > 0:

  ex_a  = ex_a[v]
  X_i   = X_i[v[1:]-1]
  
  out_n = glm(X_i, y)
  
  yhat_n  = out_n['yhat']
  resid_n = out_n['resid']
  ahat_n  = out_n['ahat']
  ci_n    = out_n['ci']

  yhat_f = Function(Q)
  resi_f = Function(Q)
  yhat_f.vector()[valid] = yhat_n
  resi_f.vector()[valid] = resid_n
  
  plotIce(drg, yhat_f, name='GLM_beta_reduced', direc=g_fn, basin='jakobshavn', 
          title=r'$\hat{\beta}$', cmap='gist_yarg', scale='log', 
          umin=1.0, umax=betaMax, numLvls=12, tp=False, tpAlpha=0.5, show=False)
  
  plotIce(drg, resi_f, name='GLM_resid_reduced', direc=g_fn, 
          title=r'$d$', cmap='RdGy', scale='lin', basin='jakobshavn', 
          umin=-50, umax=50, numLvls=13, tp=False, tpAlpha=0.5, show=False)
    
  #=============================================================================
  # data analysis :
  
  fig      = figure()
  ax       = fig.add_subplot(111)
  
  ax.hist(y,      300, histtype='step', color='k', lw=1.5, alpha=1.0,
          normed=True, label=r'$\beta$')
  ax.hist(yhat_n, 300, histtype='step', color='r', lw=1.5, alpha=1.0, 
          normed=True, label=r'$\hat{\beta}$')
  ax.set_xlim([0,200])
  #ax.set_ylim([0,0.03])
  ax.set_xlabel(r'$\hat{\beta}$')
  ax.set_ylabel('Frequency')
  ax.legend(loc='upper right')
  ax.grid()
  tight_layout()
  fn = 'images/stats/'+file_n+'GLM_beta_distributions_reduced.png'
  savefig(fn, dpi=100)
  #show()
  close(fig)
    
  #=============================================================================
  # residual plot and normal quantile plot for residuals :
  fig = figure(figsize=(12,5))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)
  
  #rtol    = 30
  #Xr      = X_i[:, resid_n < rtol]
  #yhat_n  = yhat_n[resid_n < rtol]
  #resid_n = resid_n[resid_n < rtol]
  
  # Normal quantile plot of residuals
  ((osm,osr), (m, b, r)) = probplot(resid_n)
  interp = interp1d(osm, osr)
  yl     = interp(-2.5)
  yh     = interp(2.5)
  ax1.plot(osm, m*osm + b, 'r-', lw=2.0, label=r'LS fit')
  ax1.plot(osm, osr,       'k.', alpha=1.0, label='$\mathbf{d}$')
  ax1.set_xlabel('Standard Normal Quantiles')
  ax1.set_ylabel('Residuals')
  #ax1.set_title('Normal Quantile Plot')
  ax1.set_xlim([-2.5, 2.5])
  ax1.set_ylim([yl,   yh])
  ax1.legend(loc='lower right')
  ax1.grid()
  
  ax2.plot(yhat_n, resid_n, 'k.', alpha=0.10)
  ax2.set_xlabel(r'$\hat{\beta}$')
  ax2.set_ylabel('Residuals')
  #ax2.set_title('Residual Plot')
  ax2.set_xlim([0,  150])
  ax2.set_ylim([yl, yh])
  ax2.grid()
  
  tight_layout()
  fn = 'images/stats/'+file_n+'GLM_resid-NQ_reduced.png'
  savefig(fn, dpi=100)
  #show()
  close(fig)

  #=============================================================================
  # plot newton residuals :
  fig = figure()
  ax  = fig.add_subplot(111)
  
  ax.plot(out_n['rel_a'], 'k-', lw=2.0,
          label=r'$\Vert \alpha - \alpha_n \Vert^2$')
  ax.plot(out_n['dev_a'], 'r-', lw=2.0,
          label=r'$\Vert \mathbf{d} - \mathbf{d}_n \Vert^2$')
  ax.set_xlabel(r'Iteration')
  ax.set_yscale('log')
  ax.set_xlim([0, len(out_n['dev_a'])-1])
  ax.grid()
  ax.legend()
  fn = 'images/stats/' + file_n + 'GLM_newton_resid_reduced.png'
  tight_layout()
  savefig(fn, dpi=100)
  #show()
  close(fig)

  #=============================================================================
  # save tables :
  #srt = argsort(abs(ahat_n))[::-1]
  fn  = open('dat/'+file_n+'alpha_reduced.dat', 'wb')
  #for n, a, c in zip(ex_a[srt], ahat_n[srt], ci_n[srt]):
  for n, a, c in zip(ex_a, ahat_n, ci_n):
    al = a-c
    ah = a+c
    if sign(al) != sign(ah):
      strng = '\\color{red}%s & \\color{red}%.1e & \\color{red}%.1e & ' + \
              '\\color{red}%.1e \\\\\n'
    else:
      strng = '%s & %.1e & %.1e & %.1e \\\\\n'
    fn.write(strng % (n, al, a, ah))
  fn.write('\n')
  fn.close()

  mu       = mean(yhat_n)                  # mean
  med      = median(yhat_n)                # median
  sigma    = std(yhat_n)                   # standard deviation
  fe_iqr   = iqr(yhat_n)                   # IQR
  v_m_rat  = sigma**2 / mu                 # variance-to-mean ratio
  stats_yh = [mu, med, sigma**2, fe_iqr, v_m_rat, 
              out_n['R2'], out_n['F'], out_n['AIC'], out_n['sighat']]

  names = ['$\mu$', 'median', '$\sigma^2$', 'IQR',   '$\sigma^2 / \mu$',
           '$R^2$', 'F',      'AIC',        '$\hat{\sigma}^2$']
  
  fn = open('dat/' + file_n + 'stats_reduced.dat', 'wb')
  for n, s_yh in zip(names, stats_yh):
    strng = '%s & %g \\\\\n' % (n, s_yh)
    fn.write(strng)
  fn.write('\n')
  fn.close()

  v = []
  for i,(a,c) in enumerate(zip(ahat_n, ci_n)):
    al = a-c
    ah = a+c
    if sign(al) == sign(ah):
      v.append(i)
  v = array(v)
  exterminated = len(ahat_n) - len(v)
  print "eliminated %i fields" % exterminated



