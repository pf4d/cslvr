Installation
=======================


From source
------------------------

FEniCS 1.6::

  sudo add-apt-repository ppa:fenics-packages/fenics;
  sudo apt-get update;
  sudo apt-get install fenics;
  sudo apt-get dist-upgrade;

After this, install `Dolfin-Adjoint <http://dolfin-adjoint-doc.readthedocs.org/en/latest/download/index.html>`_.

Latest Python packages and misc. dependencies::

  sudo pip install shapely colored termcolor pyproj;

also `gmsh_dynamic and gmshpy <http://geuz.org/gmsh/>`_.

**NOTE:** GMSH is required to be installed in order to use the mesh generation facilities located in ```meshing.py```.  Instructions on installing gmsh_dynamic can be found on the `QSSI wiki <http://qssi.cs.umt.edu/wiki/index.php/Setup>`_.  For a generic install (of gmsh-dynamic 2.10.1) to ``/usr/local``, use::

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

And `basemap <http://matplotlib.org/basemap/users/installing.html>`_.  For a generic install to ``/usr/local/``, use::

  wget http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz;
  tar -xzvf basemap-1.0.7.tar.gz;
  cd basemap-1.0.7/geos-3.3.3/;
  export GEOS_DIR=/usr/local/;
  ./configure --prefix=$GEOS_DIR;
  make;
  sudo make install;
  cd ..;
  sudo python setup.py install;

Install the program by editing your .bashrc file with::
  
  export PYTHONPATH="<PATH TO CSLVR>:$PYTHONPATH"

Test your installation py entering in an ``ipython`` terminal::

  from cslvr import *


Using Docker
------------------------

If you have `docker <https://www.docker.com/>`_ installed, you can install CSLVR with::

  sudo docker pull pf4d/cslvr

Then run it like this::

  sudo docker run -ti pf4d/cslvr



