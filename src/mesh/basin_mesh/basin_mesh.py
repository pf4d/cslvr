from varglas.meshing    import MeshGenerator, MeshRefiner
from varglas.io         import DataInput
from varglas.data.data_factory import DataFactory
from varglas.io         import GetBasin
import os


# SET THE FOLLOWING BEFORE RUNNING THE SCRIPT
output_path = 'meshes/'
mesh_name = 'NEGIS_1H'
edge_res = 1000   # How closely will edges be resolved (m)?
Hmin = 800.    # minimum ice thickness in terms of how mesh is created
ice_thicknesses = 1.0 # How many ice thicknesses to create mesh

# Get data needed for operation:
bamber_factory = DataFactory.get_bamber()
rignot_factory = DataFactory.get_gre_rignot()
#bedmap2_factory = DataFactory.get_bedmap2()                   # get bedmap2 data

# process the data :
bamber = DataInput(bamber_factory, gen_space=False)
#rignot = DataInput(rignot_factory, gen_space=True)
#rignot.change_projection(bamber)

#bedmap2 = DataInput(bedmap2_factory,gen_space=False)
#dbm.set_data_min('U_ob', boundary=0.0,   val=0.0)     # Set the minimum vel.

gb = GetBasin(bamber,where="Greenland",edge_resolution=edge_res)
gb.extend_edge(5000)
contour = gb.get_xy_contour()

gb.plot_xycoords_buf(Show=False)

if not os.path.exists(output_path):
      os.makedirs(output_path)

m = MeshGenerator(bamber,mesh_name,output_path)
m.set_contour(contour)
m.plot_contour()
#m.write_gmsh_contour(boundary_extend=False)
#m.close_file()

## Mesh according to some field
#
## A transform of speed:
##rignot.data['background_field'] = (0.05 + 1/(1 + rignot.data['U_ob'])) * 25000.
#
## Thickness based:
#bamber.set_data_min('H', boundary=Hmin,   val=Hmin)     # Set the minimum vel.
#bamber.data['background_field'] =  ice_thicknesses * bamber.data['H']
#
#mr = MeshRefiner(bamber,'background_field',gmsh_file_name=output_path+mesh_name)
#a,aid = mr.add_static_attractor()
#mr.set_background_field(aid)
#mr.finish(gui=False,out_file_name=output_path+mesh_name)
