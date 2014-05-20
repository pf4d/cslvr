import sys
import os
src_directory = '../../../'
sys.path.append(src_directory)

import src.solvers            as solvers
import src.physical_constants as pc
import src.model              as model
from meshes.mesh_factory  import MeshFactory
from data.data_factory    import DataFactory
from src.utilities        import DataInput, DataOutput
from src.helper           import *
from dolfin               import *
from time                 import time

out_dir = './stress_balance/'
in_dir  = './results_high_fo/05/'

set_log_active(True)
#set_log_level(PROGRESS)

thklim = 200.0

# collect the raw data :
bamber   = DataFactory.get_bamber(thklim = thklim)

# define the meshes :
mesh      = Mesh('meshes/mesh_high_new.xml')
flat_mesh = Mesh('meshes/mesh_high_new.xml')
#mesh      = Mesh('meshes/mesh_low.xml')
#flat_mesh = Mesh('meshes/mesh_low.xml')
mesh.coordinates()[:,2]      /= 100000.0
flat_mesh.coordinates()[:,2] /= 100000.0

# create data objects to use with varglas :
dbm     = DataInput(None, bamber, mesh=mesh)

# get the expressions used by varglas :
Surface = dbm.get_spline_expression('h')
Bed     = dbm.get_spline_expression('b')

model = model.Model()
model.set_geometry(Surface, Bed)
model.set_mesh(mesh, flat_mesh=flat_mesh, deform=True)
model.set_parameters(pc.IceParameters())
model.initialize_variables()
parameters['form_compiler']['quadrature_degree'] = 2

File(in_dir + 'u.xml')     >>  model.u
File(in_dir + 'v.xml')     >>  model.v
File(in_dir + 'w.xml')     >>  model.w
File(in_dir + 'beta2.xml') >>  model.beta2
File(in_dir + 'eta.xml')   >>  model.eta

model.u.update()
model.v.update()
model.w.update()
model.beta2.update()
model.eta.update()

#===============================================================================
# calculate the "stokes-balance" stress fields :
out     = model.component_stress_stokes()
tau_lon = out[1]
tau_lat = out[2]
tau_bas = out[3]
tau_drv = out[4]

tau_lat_p_lon = project(sqrt(tau_lon**2 + tau_lat**2))
dpb           = tau_drv + tau_bas
tau_drv_p_bas = project(sqrt(inner(dpb, dpb)))
tau_tot       = project(tau_drv_p_bas - tau_lat_p_lon)
U             = as_vector([model.u, model.v, model.w])
intDivU       = model.vert_integrate(div(U))
intU          = model.vert_integrate(sqrt(inner(U,U)))
w_bas_e       = model.extrude(model.w, 3, 2)
tau_drv       = project(sqrt(tau_drv[0]**2 + tau_drv[1]**2 + tau_drv[2]**2))
tau_bas       = project(sqrt(tau_bas[2]**2 + tau_bas[1]**2 + tau_bas[2]**2))

w_bas_e.update()
intDivU.update()

# output diagnostics :
File(out_dir + 'w_bas_e.pvd')        << project(w_bas_e)
File(out_dir + 'intDivU.pvd')        << project(intDivU)
File(out_dir + 'intU.pvd')           << project(intU)

# output "stokes-balance" :
File(out_dir + 'tau_drv.pvd')        << tau_drv
File(out_dir + 'tau_bas.pvd')        << tau_bas
File(out_dir + 'tau_lon.pvd')        << tau_lon
File(out_dir + 'tau_lat.pvd')        << tau_lat
File(out_dir + 'tau_tot.pvd')        << tau_tot
File(out_dir + 'tau_lat_p_lon.pvd')  << tau_lat_p_lon
File(out_dir + 'tau_drv_p_bas.pvd')  << tau_drv_p_bas

#===============================================================================
# calculate the "stress-balance" stress fields :
#out     = model.component_stress()
#tau_lon = out[0]
#tau_lat = out[1]
#tau_vrt = out[2]
#tau_bas = out[3]
#tau_drv = out[4]
#
#tau_lat_p_lon = project(sqrt(tau_lat**2 + tau_lon**2))
#tau_tot       = project(tau_drv_p_bas - tau_lat_p_lon)
#
## output "stress-balance" :
#File(out_dir + 'tau_lon_s.pvd')        << tau_lon
#File(out_dir + 'tau_lat_s.pvd')        << tau_lat
#File(out_dir + 'tau_vrt_s.pvd')        << tau_vrt
#File(out_dir + 'tau_tot_s.pvd')        << tau_tot
#File(out_dir + 'tau_lat_p_lon_s.pvd')  << tau_lat_p_lon


