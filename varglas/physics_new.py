from helper import raiseNotDefined


class Physics(object):
  """
  This abstract class outlines the structure of a physics calculation.
  """

  def __new__(self, model):
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

  def solve(self):
    """
    Solves the physics calculation.
    """
    raiseNotDefined()



