"""

EISMINT model intercomparison:

http://homepages.vub.ac.be/~phuybrec/eismint.html

Solves the thermo-mechanically-coupled three-dimensional model and
upper-free-surface equation

Written by Evan Cummings in his free time during May--August, 2018.

"""

from cslvr import *

# directories for loading or saving data :
var_dir = './dump/vars/' # directory where the ``gen_vars.py`` data are located
out_dir = './dump/A/'    # directory to save results
mdl_odr = 'BP'           # the order of the momentum model

thklim  = 1.0            # [m] thickness limit

fdata   = HDF5File(mpi_comm_world(), var_dir + 'state.h5',  'r')
fmeshes = HDF5File(mpi_comm_world(), var_dir + 'meshes.h5', 'r')

# initialize the model :
model = D3Model(fmeshes, out_dir)

model.set_subdomains(fdata)
model.set_srf_mesh(fmeshes)

# form a 2D model using the upper-surface mesh :
srfmodel = D2Model(model.srfmesh,
                   out_dir      = out_dir,
                   use_periodic = False,
                   kind         = 'submesh')

# initialize the 3D model variables :
model.assign_variable(model.S,         fdata)
model.assign_variable(model.B,         fdata)
model.assign_variable(model.sigma,     fdata)
model.assign_variable(model.mask,      fdata)
model.assign_variable(model.U_mask,    fdata)
model.assign_variable(model.S_ring,    fdata)
model.assign_variable(model.T_surface, fdata)
model.assign_variable(model.T,         model.T_surface)
model.assign_variable(model.q_geo,     fdata)
model.assign_variable(model.W,         0.0)
model.assign_variable(model.k_0,       1e-3)
model.assign_variable(model.beta,      1e9)
model.assign_variable(model.A,         1e-16)
#model.init_beta_stats()
#model.solve_hydrostatic_pressure()
#model.form_energy_dependent_rate_factor()

# update the 2D model variables that we'll need to compute the mass balance :
model.assign_to_submesh_variable(u = model.S,      u_sub = srfmodel.S)
model.assign_to_submesh_variable(u = model.B,      u_sub = srfmodel.B)
model.assign_to_submesh_variable(u = model.S_ring, u_sub = srfmodel.S_ring)

# we can choose any of these to solve our 3D-momentum problem :
if mdl_odr == 'BP':
	mom = MomentumBP(model, use_pressure_bc=False)
elif mdl_odr == 'BP_duk':
	mom = MomentumDukowiczBP(model, use_pressure_bc=False)
elif mdl_odr == 'RS':
	mom = MomentumDukowiczStokesReduced(model, use_pressure_bc=False)
elif mdl_odr == 'FS_duk':
	mom = MomentumDukowiczStokes(model, use_pressure_bc=False)
elif mdl_odr == 'FS_stab':
	mom = MomentumNitscheStokes(model, use_pressure_bc=False, stabilized=True)
elif mdl_odr == 'FS_th':
	mom = MomentumNitscheStokes(model, use_pressure_bc=False, stabilized=False)

mom.solve_params['solver']['newton_solver']['relaxation_parameter'] = 0.7

#nrg = Enthalpy(model, mom,
#               transient  = True,
#               use_lat_bc = False)
mass = UpperFreeSurface(srfmodel,
                        thklim              = thklim,
                        lump_mass_matrix    = False)

# create a function to be called at the end of each iteration :
U_file  = XDMFFile(out_dir + 'U.xdmf')
p_file  = XDMFFile(out_dir + 'p.xdmf')
S_file  = XDMFFile(out_dir + 'S.xdmf')
dS_file = XDMFFile(out_dir + 'dSdt.xdmf')
#T_file  = XDMFFile(out_dir + 'T.xdmf')

def cb_ftn(t):
	#nrg.solve()                               # quasi-thermo-mechanical couple
	model.save_xdmf(model.u,         'U3',    f = U_file,  t = t)
	model.save_xdmf(model.p,          'p',     f = p_file,  t = t)
	srfmodel.save_xdmf(srfmodel.S,    'S',     f = S_file,  t = t)
	srfmodel.save_xdmf(srfmodel.dSdt, 'dSdt',  f = dS_file, t = t)
	#model.save_xdmf(model.T,  'T',  f = T_file)

# run the transient simulation :
model.transient_solve(mom, mass,
                      t_start    = 0.0,
                      t_end      = 10.0,#5000.0
                      time_step  = 1.0,
                      tmc_kwargs = None,
                      adaptive   = False,
                      annotate   = False,
                      callback   = cb_ftn)

U_file.close()
p_file.close()
S_file.close()
dS_file.close()



