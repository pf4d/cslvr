from distutils.core import setup

setup( name        ='varglas',
       version     ='1.0',
       description ='Variational Glacier Simulator (VarGlaS)',
       author      ='Douglas Brinkerhoff',
       url         ='https://github.com/douglas-brinkerhoff/VarGlaS',
       packages    =['', 
                     'varglas.plot', 
                     'varglas.mesh', 
                     'varglas.data'],
       package_dir ={''             : 'src',
                     'varglas.plot' : 'src/plot', 
                     'varglas.mesh' : 'src/mesh', 
                     'varglas.data' : 'src/data'},
     )
