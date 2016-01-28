cslvr : [c]ryospheric problem [s]o[lv]e[r]
=======

Built from elements of [Var]iational [Gl]acier [S]imulator (VarGlaS) codes:

https://github.com/douglas-brinkerhoff/VarGlaS

and the [U] of [M] [F]irn [D]ensification [M]odel (UM-FDM):

https://github.com/pf4d/um-fdm

utilizing the FEniCS project:

http://fenicsproject.org/

and Dolfin-Adjoint:

http://dolfin-adjoint-doc.readthedocs.org/en/latest/

installation (verified with ubuntu 15.04) :
-------------------------------------------

Latest Python packages and misc. dependencies:

```bash
sudo apt-get install python-pip python-vtk python-dev build-essential libatlas-base-dev gfortran libfreetype6-dev;
sudo pip install numpy shapely matplotlib scipy colored termcolor ipython sympy netcdf ply mpi4py pyproj --upgrade;
```

FEniCS 1.6 :

```bash
sudo add-apt-repository ppa:fenics-packages/fenics;
sudo apt-get update;
sudo apt-get install blt dolfin-bin dolfin-doc dvipng fonts-lyx  libblacs-mpi-dev libblacs-mpi1 libboost-serialization-dev libdolfin-dev libdolfin1.6 libfftw3-mpi-dev libfftw3-mpi3 libhdf5-mpi-dev libmshr-dev libmshr1.6 libmumps-4.10.0 libmumps-dev libpetsc3.4.2 libpetsc3.4.2-dev libscalapack-mpi-dev libscalapack-mpi1 libslepc3.4.2 libslepc3.4.2-dev libspooles-dev libspooles2.2 libwebpmux1 mshr-demos pyro python-dateutil python-decorator python-dolfin python-ffc python-fiat python-gnuplot python-imaging python-instant python-matplotlib-data python-mshr python-petsc4py python-pexpect python-pil python-pmw python-pyparsing python-pyx python-scitools python-simplegeneric python-tz python-gdal;
sudo apt-get dist-upgrade;
```


After this, install Dolfin-Adjoint as described [here](http://dolfin-adjoint-doc.readthedocs.org/en/latest/download/index.html), but utilizing my cslvr-modified fork located [here](https://github.com/pf4d/dolfin-adjoint).

also ```Gmsh_dynamic``` and ```GmshPy``` from

http://geuz.org/gmsh/

**NOTE:** GMSH is required to be installed in order to use the mesh generation facilities located in ```meshing.py```.  Instructions on installing Gmsh_dynamic can be found on the [Qssi wiki ](http://qssi.cs.umt.edu/wiki/index.php/Setup).

For a generic install (of gmsh-dynamic 2.10.1) to ``/usr/local``, use

```bash
wget http://geuz.org/gmsh/bin/Linux/gmsh-svn-Linux64-dynamic.tgz;
tar -xzvf gmsh-svn-Linux64-dynamic.tgz;
cd gmsh-2.10.1-dynamic-svn-Linux;
sudo cp -r bin/ /usr/local/;
sudo cp -r include/ /usr/local/;
sudo cp -r lib/ /usr/local/;
sudo cp -r share/ /usr/local/;
cd gmshpy;
sudo python setup.py install;
sudo ldconfig;

```

And basemap:

http://matplotlib.org/basemap/users/installing.html

for a generic install to ``/usr/local/``, use
```bash
wget http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz;
tar -xzvf basemap-1.0.7.tar.gz;
cd basemap-1.0.7/geos-3.3.3/;
export GEOS_DIR=/usr/local/;
./configure --prefix=$GEOS_DIR;
make;
sudo make install;
cd ..;
sudo python setup.py install;
```

Install the program by editing your .bashrc file with
```bash
export PYTHONPATH="<PATH TO VARGLAS>:$PYTHONPATH"
```

Test your installation py entering in an ``ipython`` terminal

```python
from cslvr import *
```

Data download :
---------------

You may like to download some data and pre-made meshes.  To do this, go into the ``scripts/`` directory and run the ``download_*.py`` files there, dependending on your needs.

The data may be accessed via the ``DataFactory`` class, and the meshes via the ``MeshFactory`` class.

