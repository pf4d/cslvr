from varglas.meshing    import MeshGenerator, MeshRefiner, GetBasin
from varglas.io         import DataInput
from varglas.data.data_factory import DataFactory
import os


# SET THE FOLLOWING BEFORE RUNNING THE SCRIPT
output_path = 'meshes/test/'
mesh_name = 'TEST'
edge_res = 1000   # How closely will edges be resolved (m)?
Hmin = 1000.    # minimum ice thickness in terms of how mesh is created
ice_thicknesses = 2.0 # How many ice thicknesses to create mesh

# Get data needed for operation:
bamber_factory = DataFactory.get_bamber()

# process the data :
bamber = DataInput(bamber_factory, gen_space=False)

# create file if it does not exist
if not os.path.exists(output_path):
      os.makedirs(output_path)

mb = MeshGenerator(bamber,mesh_name,output_path)
mb.create_contour('H',1,1)

gb = GetBasin(bamber,basin='6.2', edge_resolution=edge_res)
gb.extend_edge(20000)
gb.intersection(mb.longest_cont)
contour = gb.get_xy_contour()

gb.plot_xycoords_buf(Show=True,other=mb.longest_cont)

m = MeshGenerator(bamber,mesh_name,output_path)
m.set_contour(contour)
m.plot_contour()
m.write_gmsh_contour(boundary_extend=False)
m.close_file()

# A transform of speed:
#rignot.data['background_field'] = (0.05 + 1/(1 + rignot.data['U_ob'])) * 25000.

# Thickness based:
bamber.set_data_min('H', boundary=Hmin,   val=Hmin) # Set the minimum thickness.
bamber.data['background_field'] =  ice_thicknesses * bamber.data['H']

mr = MeshRefiner(bamber,'background_field',gmsh_file_name=output_path+mesh_name)
a,aid = mr.add_static_attractor()
mr.set_background_field(aid)
mr.finish(gui=False,out_file_name=output_path+mesh_name)
m.convert_msh_to_xml(mesh_name, mesh_name)
