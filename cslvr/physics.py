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

  def form_reg_ftn(self, c, integral, kind='TV', alpha=1.0,
                   alpha_tik=1e-7, alpha_tv=10):
    """
    Formulates, and returns the regularization functional for use 
    with adjoint, saved to self.R.
    """
    self.alpha = alpha   # need to save this for printing values.
    model = self.model
    self.reg_ftn_type = kind     # need to save this for printing values.

    # differentiate between regularization over cells or facets :
    if integral in [model.OMEGA_GND, model.OMEGA_FLT]:
      dR = model.dx(integral)
    else:
      dR = model.ds(integral)
    
    kinds = ['TV', 'Tikhonov', 'TV_Tik_hybrid', 'square', 'abs']
    
    # form regularization term 'R' :
    if kind not in kinds:
      s    =   ">>> VALID REGULARIZATIONS ARE 'TV', 'Tikhonov', 'square', " + \
               " 'abs', or 'TV_Tik_hybrid' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)
    elif kind == 'TV':
      R  = alpha * sqrt(inner(grad(c), grad(c)) + 1e-15) * dR
      Rp = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR
      s  = "::: forming 'TV' regularization functional with parameter" + \
           " alpha = %.2E :::" % alpha
    elif kind == 'Tikhonov':
      R  = alpha * 0.5 * inner(grad(c), grad(c)) * dR
      Rp = 0.5 * inner(grad(c), grad(c)) * dR
      s  = "::: forming 'Tikhonov' regularization functional with parameter" + \
           " alpha = %.2E :::" % alpha
    elif kind == 'TV_Tik_hybrid':
      self.R1  = alpha_tik * 0.5 * inner(grad(c), grad(c)) * dR
      self.R2  = alpha_tv * sqrt(inner(grad(c), grad(c)) + 1e-15) * dR 
      self.R1p = 0.5 * inner(grad(c), grad(c)) * dR
      self.R2p = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR
      R  = self.R1 + self.R2
      Rp = self.R1p + self.R2p
      s   = "::: forming Tikhonov/TV hybrid regularization with alpha_tik = " \
            "%.1e and alpha_tv = %.1e :::" % (alpha_tik, alpha_tv)
    elif kind == 'square':
      R  = alpha * 0.5 * c**2 * dR
      Rp = 0.5 * c**2 * dR
      s  = "::: forming 'square' regularization functional with parameter" + \
           " alpha = %.2E :::" % alpha
    elif kind == 'abs':
      R  = alpha * abs(c) * dR
      Rp = abs(c) * dR
      s  = "::: forming 'abs' regularization functional with parameter" + \
           " alpha = %.2E :::" % alpha
    print_text(s, self.color())
    s = "    - integrated over %s -" % model.boundaries[integral]
    print_text(s, self.color())
    self.R  = R
    self.Rp = Rp  # needed for L-curve

  def solve(self):
    """
    Solves the physics calculation.
    """
    raiseNotDefined()



