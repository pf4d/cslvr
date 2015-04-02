VarGlaS
=======

Forked from the source code repository for the Variational Glacier Simulator (VarGlaS) code:

https://github.com/douglas-brinkerhoff/VarGlaS

utilizing the FEniCS project:

http://fenicsproject.org/

installation:
-------------

dependencies:

```bash
sudo apt-get install python python-scipy python-pyproj \
                     python-mpltoolkits.basemap \
                     python-termcolor python-gdal \
                     python-shapely fenics python-pip
```

and you'll need to install the ```colored``` package,

```bash
pip install colored
```


also ```Gmsh_dynamic``` and ```GmshPy``` from

http://geuz.org/gmsh/

**NOTE:** GMSH is *only* required to be installed in order to use the mesh generation facilities located in ```meshing.py```.  Instructions on installing Gmsh_dynamic can be found on the [Qssi wiki ](http://qssi.cs.umt.edu/wiki/index.php/Setup).

To install the program and download program data to the your home directory,

```bash
python setup.py install --user
python download_all_data_and_meshes.py
```

Partial documentation and install directions may be found at

http://qssi.cs.umt.edu/wiki/index.php/Main_Page

Old documentation is located here:

http://cas.umt.edu/feismd/

Please note that this is old and some descriptions may have changed for this fork of VarGlaS (previously called UM-FEISM).

