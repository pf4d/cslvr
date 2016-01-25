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

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
  module = loader.find_module(name).load_module(name)
  for name, value in inspect.getmembers(module):
    if name.startswith('__'):
      continue

    globals()[name] = value
    __all__.append(name)

#from age                import Age
#from balancevelocity    import BalanceVelocity 
#from d2model            import D2Model
#from d3model            import D3Model
#from energy             import Energy, Enthalpy, EnergyHybrid
#from helper             import plotIce, plot_variable 
#from io                 import DataInput, DataOutput, print_min_max, \
#                               print_text, get_text
#from mass               import FreeSurface
#from meshing            import MeshGenerator, MeshRefiner, GetBasin 
#from momentumbp         import MomentumBP
#from momentumhybrid     import MomentumHybrid
#from momentumstokes     import MomentumStokes, MomentumDukowiczStokesReduced, \
#                               MomentumDukowiczStokes
#from physics            import Physics
#from stressbalance      import SSA_Balance, BP_Balance
#from surfacemassbalance import SurfaceMassBalance
