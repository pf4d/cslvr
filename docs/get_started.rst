Hello CSLVR
===========

To get started with CSLVR, we begin with an example that does not require any external data; the "`Ice Sheet Model Intercomparison Project for Higher-Order Models <http://homepages.ulb.ac.be/~fpattyn/ismip/>`_".

Set up the model
----------------

First, import CSLVR::

  from cslvr import *

Next, we make a simple three-dimensional box mesh with 15 cells in the :math:`x` and :math:`y` directions, and 5 cells in the :math:`z` direction, and a width of 8 km::

  L     = 8000                          # width of domain
  p1    = Point(0.0, 0.0, 0.0)          # origin
  p2    = Point(L,   L,   1)            # x, y, z corner 
  mesh  = BoxMesh(p1, p2, 15, 15, 5)    # a box to fill the void 


We have a three-dimensional problem here, with periodic lateral boundaries. Thus we need to instantiate a :class:`~d3model.D3Model`, with periodic lateral boundaries::

  model = D3Model(mesh, out_dir = './results/', use_periodic = True)

We now mark the exterior facets and interior cells appropriately by calling :func:`~model.Model.calculate_boundaries`::

  model.calculate_boundaries()

The ISMIP-HOM experiment "A" geometry are created using the FEniCS class :class:`~fenics.Expression`.  This class requires the specification of the type of finite-element used by the ``model`` -- defined by the :class:`~fenics.FunctionSpace` created in the instantiation above -- accessed by ``model.Q``.  With the surface and bed functions defined, the mesh may be deformed to the desired geometry through the use of :func:`~d3model.D3Model.deform_mesh_to_geometry`::

  a       = 0.5 * pi / 180     # surface slope in radians
  surface = Expression('- x[0] * tan(a)', a=a,
                       element=model.Q.ufl_element())
  bed     = Expression(  '- x[0] * tan(a) - 1000.0 + 500.0 * ' \
                       + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                       a=a, L=L, element=model.Q.ufl_element())
  model.deform_mesh_to_geometry(surface, bed)

Solve the momentum balance
--------------------------

We can now set the desired isothermal flow-rate factor :math:`A` and constant basal traction coefficient :math:`\beta` through the appropriate ``init_`` method of the abstract class :class:`~model.Model`::

  model.init_beta(1000)                       # really high friction
  model.init_A(1e-16)                         # cold, isothermal rate-factor

For three-dimensional momentum problems, we can solve either the first-order :class:`~momentumbp.MomentumDukowiczBP` physics, ::

  mom = MomentumDukowiczBP(model)

the reformulated-Stokes :class:`~momentumstokes.MomentumDukowiczStokesReduced` physics, ::

  mom = MomentumDukowiczStokesReduced(model)

or the full-Stokes :class:`~momentumstokes.MomentumDukowiczStokes` physics, ::

  mom = MomentumDukowiczStokes(model)

Once this choice is made, you can solve the momentum balance::

  mom.solve()

Let's investigate the resulting velocity divergence :math:`\nabla \cdot \mathbf{u}` by projecting the 3D velocity resulting from the momentum balance that is saved to ``model.U3`` with the FEniCS function :func:`~fenics.project` and the UFL divergence function :func:`~fenics.div`::

  divU = project(div(model.U3))

Plot the results
----------------

Now we can save the resulting velocity ``model.U3``, pressure ``model.p`` and our calculated ``divU`` functions to xdmf files for use with `paraview <http://www.paraview.org/>`_::

  model.save_xdmf(model.p,  'p')
  model.save_xdmf(model.U3, 'U')
  model.save_xdmf(divU,     'divU')
  
Additionally, we can plot the :class:`~fenics.Function`\ s over the surface or bed by creating surface and bed meshes associated with the 3D model::

  model.form_srf_mesh()
  model.form_bed_mesh()
  
These functions save the surface mesh to ``model.srfmesh`` and bed mesh to ``model.bedmesh``.  With these created, we can instantiate 2D models with the :class:`~d2model.D2Model` class::

  srfmodel = D2Model(model.srfmesh)
  bedmodel = D2Model(model.bedmesh)
  
We don't have a function for ``divU`` included in the ``model`` instance, so we have to make one ourselves::

  divU_b   = Function(bedmodel.Q)
  
Next, we interpolate from the 3D mesh to the 2D mesh using the Lagrange interpolation method :func:`~model.Model.assign_submesh_variable`::

  bedmodel.assign_submesh_variable(divU_b, divU)
  srfmodel.assign_submesh_variable(srfmodel.U3, model.U3)
  bedmodel.assign_submesh_variable(bedmodel.p,  model.p)

To plot :math:`\mathbf{u}`, we need to calculate the velocity magnitude::
  
  srfmodel.init_U_mag(srfmodel.U3)

Now we figure out some nice-looking contour levels::

  U_min  = srfmodel.U_mag.vector().min()
  U_max  = srfmodel.U_mag.vector().max()
  U_lvls = array([84, 86, 88, 90, 92, 94, 96, 98, 100])
  
  p_min  = bedmodel.p.vector().min()
  p_max  = bedmodel.p.vector().max()
  p_lvls = array([4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 1.1e7, 1.2e7, p_max])
  
  d_min  = divU_b.vector().min()
  d_max  = divU_b.vector().max()
  d_lvls = array([d_min, -5e-3, -2.5e-3, -1e-3, 
                  1e-3, 2.5e-3, 5e-3, d_max])
  
and finally plot the variables using :func:`~helper.plot_variable`::

  plot_variable(u = srfmodel.U3, name = 'U_mag', direc = plt_dir,
                levels              = U_lvls,
                cmap                = 'viridis',
                tp                  = True,
                show                = False,
                extend              = 'both',#'neither',
                cb_format           = '%g')
  
  plot_variable(u = bedmodel.p, name = 'p', direc = plt_dir,
                levels              = p_lvls,
                cmap                = 'viridis',
                tp                  = True,
                show                = False,
                extend              = 'min',
                cb_format           = '%.1e')
  
  plot_variable(u = divU_b, name = 'divU', direc = plt_dir,
                cmap                = 'RdGy',
                levels              = d_lvls,
                tp                  = True,
                show                = False,
                extend              = 'neither',
                cb_format           = '%.1e')
