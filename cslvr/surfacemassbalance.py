from cslvr.physics import Physics


class SurfaceMassBalance(Physics):
  
  def __init__(self, model, config):
    """
    """ 
    s    = "::: INITIALIZING SURFACE-MASS-BALANCE PHYSICS :::"
    print_text(s, self.color())


    self.model  = model
    self.config = config
    
    Q           = model.Q
    S           = model.S
    B           = model.B
    H           = S - B
    Mb          = model.Mb
    ubar        = model.ubar
    vbar        = model.vbar
    wbar        = model.wbar
   
    phi  = TestFunction(Q)
    adot = TrialFunction(Q)

    Ubar = as_vector([ubar, vbar, wbar])
    
    self.B = (div(Ubar*H) + Mb) * phi * dx
    self.a = adot * phi * dx
  
  def solve(self):
    """
    Solve for the surface mass balance.
    """
    model = self.model

    s    = "::: solving for surface mass balance :::"
    print_text(s, self.color())
    solve(self.a == self.B, model.adot)
    print_min_max(model.adot, 'adot')
