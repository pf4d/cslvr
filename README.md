VarGlaS
=======

Forked from the source code repository for the Variational Glacier Simulator (VarGlaS) code:

https://github.com/douglas-brinkerhoff/VarGlaS

utilizing the FEniCS project:

ttp://fenicsproject.org/

installation:
-------------

dependencies:

```bash
sudo apt-get install python python-scipy python-pyproj \
                     python-mpltoolkits.basemap \
                     python-termcolor python-gdal fenics
```

also ```Gmsh_dynamic``` from

http://geuz.org/gmsh/


To install the program in the ```usr``` directory and download program data,

```bash
python setup.py install --user
python download_all_data_and_meshes.py
```


Partial documentation and install directions may be found at

http://qssi.cs.umt.edu/wiki/index.php/Main_Page

