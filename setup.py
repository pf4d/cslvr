from distutils.core import setup

setup
(      
  name        ='cslvr',
  version     ='2016.1.0',
  description ='Cryospheric Problem Solver',
  author      ='Evan Cummings, Douglas Brinkerhoff, Jesse Johnson',
  url         ='https://github.com/pf4d/clsvr',
  packages    =['cslvr', 
                'tifffile'],
  package_dir ={'cslvr'                 : 'cslvr',
                'tifffile'              : 'cslvr/ext_scripts'}
)
