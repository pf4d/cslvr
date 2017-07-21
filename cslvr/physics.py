from helper            import raiseNotDefined
from fenics            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import print_text
from cslvr.d2model     import D2Model


class Physics(object):
  """
  This abstract class outlines the structure of a physics calculation.

  :param model: the model instance for this physics problem
  :type model:  :class:`~model.Model`
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

    :rtype: string
    """
    return 'white'
  
  def default_solve_params(self):
    """ 
    Returns a set of default solver parameters that yield good performance

    :rtype: dict
    """
    params  = {'solver' : 'mumps'}
    return params

  def form_reg_ftn(self, c, integral, kind='TV', alpha=1.0,
                   alpha_tik=1e-7, alpha_tv=10):
    r"""
    Formulates the regularization functional for with regularization 
    parameter :math:`\alpha` given by ``alpha`` for use 
    with optimization of the control parameter :math:`c` given by ``c`` 
    over the integral ``integral``.

    The choices for ``kind`` are :

    1. ``Tikhonov`` -- Tikhonov regularization

    .. math::

      \mathscr{R}(c) = \frac{\alpha}{2} \int_{\Gamma} \nabla c \cdot \nabla c\ d\Gamma

    2. ``TV`` -- total variation regularization

    .. math::

      \mathscr{R}(c) = \alpha \int_{\Gamma} \left( \nabla c \cdot \nabla c + c_0 \right)^{\frac{1}{2}}\ d\Gamma,

    3. ``square`` -- squared regularization

    .. math::

      \mathscr{R}(c) = \frac{\alpha}{2} \int_{\Gamma} c^2\ d\Gamma,

    4. ``abs`` -- absolute regularization

    .. math::
    
      \mathscr{R}(c) = \alpha \int_{\Gamma} |c|\ d\Gamma,

    5. ``TV_Tik_hybrid`` -- Tikhonov/total-variation hybrid

    .. math::

      \mathscr{R}(c) = \frac{\alpha_{\text{tik}}}{2} \int_{\Gamma} \nabla c \cdot \nabla c\ d\Gamma + \alpha_{\text{tv}} \int_{\Gamma} \left( \nabla c \cdot \nabla c + c_0 \right)^{\frac{1}{2}}\ d\Gamma,

    This saves the regularization parameter :math:`\alpha` to ``self.alpha``, 
    ``kind`` to ``self.reg_ftn_type``, regularization function to ``self.R``, 
    and regularization functional without multiplying by :math:`\alpha` 
    to ``self.Rp``.

    :param c:         the control variable
    :param integral:  measure over which to integrate 
                      (see :func:`~model.calculate_boundaries`)
    :param kind:      kind of regularization to use
    :param alpha:     regularization parameter
    :param alpha_tik: Tikhonov regularization parameter (if ``kind`` is
                      ``TV_Tik_hybrid``)
    :param alpha_tv:  TV regularization parameter (if ``kind`` is 
                      ``TV_Tik_hybrid``)
    :type c:          :class:`~fenics.Function`
    :type integral:   int 
    :type kind:       string
    :type alpha:      float
    :type alpha_tik:  float
    :type alpha_tv:   float
    """
    self.alpha = alpha   # need to save this for printing values.
    model = self.model
    self.reg_ftn_type = kind     # need to save this for printing values.

    # if there are a list of integration measures, iterate through them:
    if hasattr(integral, '__iter__'):
      meas = []
      for i in integral:
        # the 2D model will have surface boundaries are in Omega :
        if i in [model.OMEGA_GND, model.OMEGA_FLT] or type(model) == D2Model:
          meas.append(model.dx)
        # otherwise use surface integrals :
        else:
          meas.append(model.ds)
      bndrys = model.boundaries[integral[0]]   # for printing to screen
      dR     = meas[0](integral[0])            # reg. integration measure
      for i,m in zip(integral[1:], meas[1:]):
        dR     += m(i)
        bndrys += ' and ' + model.boundaries[i]
    # if there is only one integration measure :
    else:
      # the 2D model will have surface boundaries are in Omega :
      if integral in [model.OMEGA_GND, model.OMEGA_FLT] \
         or type(model) == D2Model:
        dR   = model.dx(integral)
      # otherwise use surface integrals :
      else:
        dR   = model.ds(integral)
      bndrys = model.boundaries[integral]
   
    # the various possible regularization types : 
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
      R1p      = 0.5 * inner(grad(c), grad(c)) * dR
      R2p      = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR 
      R        = self.R1 + self.R2
      Rp       = R1p + R2p
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
    s = "    - integrated over %s -" % bndrys
    print_text(s, self.color())
    self.R  = R
    self.Rp = Rp  # needed for L-curve

  def solve(self):
    """
    Solves the physics calculation.
    """
    raiseNotDefined()



