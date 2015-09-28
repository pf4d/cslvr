VarGlaS
=======

Forked from the source code repository for the Variational Glacier Simulator (VarGlaS) code:

https://github.com/douglas-brinkerhoff/VarGlaS

utilizing the FEniCS project:

http://fenicsproject.org/

and Dolfin-Adjoint:

http://dolfin-adjoint-doc.readthedocs.org/en/latest/

installation:
-------------

Latest Python packages and misc. dependencies:

```bash
sudo apt-get install python-pip python-vtk python-dev build-essential libatlas-base-dev gfortran libfreetype6-dev python-gdal;
  sudo pip install numpy shapely matplotlib scipy colored termcolor ipython sympy netcdf ply mpi4py pyproj --upgrade;
```

FEniCS 1.6 :

```bash
sudo add-apt-repository ppa:fenics-packages/fenics;
sudo apt-get update;
sudo apt-get install blt dolfin-bin dolfin-doc dvipng fonts-lyx  libblacs-mpi-dev libblacs-mpi1 libboost-serialization-dev libdolfin-dev libdolfin1.6 libfftw3-mpi-dev libfftw3-mpi3 libhdf5-mpi-dev libmshr-dev libmshr1.6 libmumps-4.10.0 libmumps-dev libpetsc3.4.2 libpetsc3.4.2-dev libscalapack-mpi-dev libscalapack-mpi1 libslepc3.4.2 libslepc3.4.2-dev libspooles-dev libspooles2.2 libwebpmux1 mshr-demos pyro python-dateutil python-decorator python-dolfin python-ffc python-fiat python-gnuplot python-imaging python-instant python-matplotlib-data python-mshr python-petsc4py python-pexpect python-pil python-pmw python-pyparsing python-pyx python-scitools python-simplegeneric python-tz;
sudo apt-get dist-upgrade;
```

also ```Gmsh_dynamic``` and ```GmshPy``` from

http://geuz.org/gmsh/

**NOTE:** GMSH is required to be installed in order to use the mesh generation facilities located in ```meshing.py```.  Instructions on installing Gmsh_dynamic can be found on the [Qssi wiki ](http://qssi.cs.umt.edu/wiki/index.php/Setup).

To install the program by editing your .bashrc file with
```bash
export PYTHONPATH="<PATH TO VARGLAS>:$PYTHONPATH"
```

After this, install Dolfin-Adjoint as described [here](http://dolfin-adjoint-doc.readthedocs.org/en/latest/download/index.html).

Somewhat dated documentation and install directions may be found at

http://qssi.cs.umt.edu/wiki/index.php/Main_Page

Even older documentation is located here:

http://cas.umt.edu/feismd/

Please note that this is old and some descriptions may have changed for this fork of VarGlaS (previously called UM-FEISM).

Data download :
---------------

You may like to download some data and pre-made meshes.  To do this, go into the ``scripts/`` directory and run the ``download_*.py`` files there, dependending on your needs.

The data may be accessed via the ``DataFactory`` class, and the meshes via the ``MeshFactory`` class.

