import sys
import varglas.solvers            as solvers
import varglas.physical_constants as pc
import varglas.model              as model
from varglas.mesh.mesh_factory    import MeshFactory
from varglas.data.data_factory    import DataFactory
from varglas.helper               import default_nonlin_solver_params
from varglas.utilities            import DataInput, DataOutput
from fenics                       import *
from time                         import time
from termcolor                    import colored, cprint

t0 = time()

out_dir = './stress_balance_stokes_5H/'
in_dir  = './stokes_5H/'

set_log_active(True)

thklim = 200.0

# collect the raw data :
bamber   = DataFactory.get_bamber(thklim = thklim)

# define the meshes :
#mesh    = Mesh('meshes/greenland_3D_5H.xml')
mesh     = MeshFactory.get_greenland_3D_5H()

# create data objects to use with varglas :
dbm     = DataInput(bamber, mesh=mesh)

# get the expressions used by varglas :
Surface = dbm.get_spline_expression('H')
Bed     = dbm.get_spline_expression('B')

model = model.Model(out_dir = out_dir)
model.set_mesh(mesh)
model.set_geometry(Surface, Bed, deform=True)
model.set_parameters(pc.IceParameters())
model.calculate_boundaries()
model.initialize_variables()
parameters['form_compiler']['quadrature_degree'] = 2

File(in_dir + 'u.xml')     >>  model.u
File(in_dir + 'v.xml')     >>  model.v
File(in_dir + 'w.xml')     >>  model.w
File(in_dir + 'beta2.xml') >>  model.beta2
File(in_dir + 'eta.xml')   >>  model.eta

config = {'output_path' : out_dir}

F = solvers.StokesBalanceSolver(model, config)
F.solve()


#===============================================================================
## calculate the cartesian "stokes-balance" stress fields :
#out  = model.component_stress_stokes_c()

#XDMFFile(mesh.mpi_comm(), out_dir + 'mesh.xdmf')   << model.mesh
#
## functionality of HDF5 not completed by fenics devs :
#f = HDF5File(out_dir + '3D_5H_stokes.h5', 'w')
#f.write(model.mesh,  'mesh')
#f.write(model.beta2, 'beta2')
#f.write(model.T,     'T')
#f.write(model.S,     'S')
#f.write(model.B,     'B')
#f.write(model.u,     'u')
#f.write(model.v,     'v')
#f.write(model.w,     'w')
#f.write(model.eta,   'eta')


#===============================================================================
## calculate the "stress-balance" stress fields :
#out  = model.component_stress()
#tau_lon = out[0]
#tau_lat = out[1]
#tau_bas = out[2]
#tau_drv = out[3]
#
#tau_drv_p_bas = project(tau_bas + tau_drv)
#tau_lat_p_lon = project(tau_lat + tau_lon)
#tau_tot       = project(tau_lat + tau_lon - tau_bas - tau_drv)
#
## output "stress-balance" :
#File(out_dir + 'tau_tot_s.pvd')      << tau_tot
#File(out_dir + 'tau_lat_p_lon.pvd')  << tau_lat_p_lon
#File(out_dir + 'tau_drv_p_bas.pvd')  << tau_drv_p_bas

tf = time()

# calculate total time to compute
s = tf - t0
m = s / 60.0
h = m / 60.0
s = s % 60
m = m % 60
if model.MPI_rank == 0:
  s    = "Total time to compute: %02d:%02d:%02d" % (h,m,s)
  text = colored(s, 'red', attrs=['bold'])
  print text



