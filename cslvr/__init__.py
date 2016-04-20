__version__    = '1.6'
__author__     = 'Evan Cummings, Douglas Brinkerhoff, Jesse Johnson'
__license__    = 'LGPL-3'
__maintainer__ = 'Evan Cummings'
__email__      = 'evan.cummings@umontana.edu'

__all__ = []

import pkgutil
import inspect
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'medium'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']
#mpl.rcParams['contour.negative_linestyle']   = 'solid'

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
  module = loader.find_module(name).load_module(name)
  for name, value in inspect.getmembers(module):
    if name.startswith('__'):
      continue

    globals()[name] = value
    __all__.append(name)

from age                 import *
from balancevelocity     import *
from d1model             import *
from d2model             import *
from d3model             import *
from datafactory         import *
from energy              import *
from helper              import *
from hybridmodel         import *
from io                  import *
from mass                import *
from meshing             import *
from model               import *
from momentumbp          import *
from momentumfirn        import *
from momentumhybrid      import *
from momentumplanestrain import *
from momentumstokes      import *
from momentum            import *
from monolithic          import *
from physics             import *
from stressbalance       import *
from surfacemassbalance  import *
