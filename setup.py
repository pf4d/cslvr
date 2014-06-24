from distutils.core import setup

setup( name        ='varglas',
       version     ='1.0',
       description ='Variational Glacier Simulator (VarGlaS)',
       author      ='Douglas Brinkerhoff',
       url         ='https://github.com/douglas-brinkerhoff/VarGlaS',
       packages    =['varglas', 
                     'varglas.plot', 
                     'varglas.mesh', 
                     'varglas.data',
                     'varglas.data.greenland',
                     'varglas.data.antarctica'],
       package_dir ={'varglas'                 : 'src',
                     'varglas.plot'            : 'src/plot', 
                     'varglas.mesh'            : 'src/mesh', 
                     'varglas.data'            : 'src/data',
                     'varglas.data.greenland'  : 'src/data/greenland',
                     'varglas.data.antarctica' : 'src/data/antarctica'}
     )
