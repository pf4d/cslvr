from helper         import raiseNotDefined
from fenics         import *
from dolfin_adjoint import *
from cslvr.io       import print_text


class Physics(object):
  """
  This abstract class outlines the structure of a physics calculation.
  """

  def __new__(self, model, *args, **kwargs):
    """
    Creates and returns a new Physics object.
    """
    instance = object.__new__(self)
    instance.model = model
    return instance
  
  def color(self):
    """
    return the default color for this class.
    """
    return 'white'
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance
    """
    params  = {'solver' : 'mumps'}
    return params

  def get_solve_params(self):
    """
    Returns the solve parameters.
    """
    return self.default_solve_params()

  def form_reg_ftn(self, c, integral, kind='TV', alpha=1.0):
    """
    Formulates, and returns the regularization functional for use 
    with adjoint, saved to self.R.
    """
    self.alpha = alpha   # need to save this for printing values.
    model = self.model

    # differentiate between regularization over cells or facets :
    if integral in [model.OMEGA_GND, model.OMEGA_FLT]:
      dR = model.dx(integral)
    else:
      dR = model.ds(integral)
    
    kinds = ['TV', 'Tikhonov', 'square', 'abs']
    
    # form regularization term 'R' :
    if kind not in kinds:
      s    =   ">>> VALID REGULARIZATIONS ARE 'TV', 'Tikhonov', 'square', " + \
               " or 'abs' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    elif kind == 'TV':
      R  = alpha * sqrt(inner(grad(c), grad(c)) + 1e-15) * dR
      Rp = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR
    elif kind == 'Tikhonov':
      R  = alpha * 0.5 * inner(grad(c), grad(c)) * dR
      Rp = 0.5 * inner(grad(c), grad(c)) * dR
    elif kind == 'square':
      R  = alpha * 0.5 * c**2 * dR
      Rp = 0.5 * c**2 * dR
    elif kind == 'abs':
      R  = alpha * abs(c) * dR
      Rp = abs(c) * dR
    s   = "::: forming %s regularization functional with parameter" + \
          " alpha = %.2E :::"
    print_text(s % (kind, alpha), self.color())
    s = "    - integrated over %s -" % model.boundaries[integral]
    print_text(s, self.color())
    self.R  = R
    self.Rp = Rp  # needed for L-curve

  def solve(self):
    """
    Solves the physics calculation.
    """
    raiseNotDefined()



