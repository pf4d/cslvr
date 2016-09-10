cslvr : [c]ryospheric problem [s]o[lv]e[r]
=======

Built from elements of the [Var]iational [Gl]acier [S]imulator (VarGlaS) codes:

https://github.com/douglas-brinkerhoff/VarGlaS

and the [U] of [M] [F]irn [D]ensification [M]odel (UM-FDM):

https://github.com/pf4d/um-fdm

utilizing the FEniCS project:

http://fenicsproject.org/

and Dolfin-Adjoint:

http://dolfin-adjoint-doc.readthedocs.org/en/latest/

installation (verified with ubuntu 15.04) :
-------------------------------------------

FEniCS 2016.1.0 :

```bash
sudo add-apt-repository ppa:fenics-packages/fenics;
sudo apt-get update;
sudo apt-get install fenics;
sudo apt-get dist-upgrade;
```

After this, install Dolfin-Adjoint as described [here](http://dolfin-adjoint-doc.readthedocs.org/en/latest/download/index.html)

Latest Python packages and misc. dependencies:

```bash
sudo pip install shapely colored termcolor pyproj;
```

also ```Gmsh_dynamic``` and ```GmshPy``` from

http://geuz.org/gmsh/

**NOTE:** GMSH is required to be installed in order to use the mesh generation facilities located in ```meshing.py```.  Instructions on installing Gmsh_dynamic can be found on the [Qssi wiki ](http://qssi.cs.umt.edu/wiki/index.php/Setup).

For a generic install (of gmsh-dynamic 2.10.1) to ``/usr/local``, use

```bash
wget http://geuz.org/gmsh/bin/Linux/gmsh-svn-Linux64-dynamic.tgz;
tar -xzvf gmsh-svn-Linux64-dynamic.tgz;
cd gmsh-2.11.0-dynamic-svn-Linux;
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
export PYTHONPATH="<PATH TO CSLVR>:$PYTHONPATH"
```

Test your installation py entering in an ``ipython`` terminal

```python
from cslvr import *
```

Data download :
---------------

You may like to download some data and pre-made meshes.  To do this, go into the ``scripts/`` directory and run the ``download_*.py`` files there, dependending on your needs.

The data may be accessed via the ``DataFactory`` class, and the meshes via the ``MeshFactory`` class.

