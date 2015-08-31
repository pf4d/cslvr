from fenics              import *
from dolfin_adjoint      import *
from varglas.physics_new import Physics
from varglas.d2model     import D2Model
from varglas.io          import print_text, print_min_max
import numpy as np
import sys

class BalanceVelocity(Physics):
  
  def __init__(self, model, kappa=5.0):
    """
    """ 
    s    = "::: INITIALIZING VELOCITY-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    if type(model) != D2Model:
      s = ">>> BalanceVelocity REQUIRES A 'D2Model' INSTANCE, NOT %s <<<"
      print_text(s % type(model) , 'red', 1)
      sys.exit(1)

    Q      = model.Q
    g      = model.g
    rho    = model.rhoi
    S      = model.S
    B      = model.B
    H      = S - B
    h      = model.h
    dSdx   = model.dSdx
    dSdy   = model.dSdy
    d_x    = model.d_x
    d_y    = model.d_y
    adot   = model.adot
        
    #===========================================================================
    # form to calculate direction of flow (down driving stress gradient) :
    phi  = TestFunction(Q)
    Ubar = TrialFunction(Q)
    Nx   = TrialFunction(Q)
    Ny   = TrialFunction(Q)
    
    # calculate horizontally smoothed driving stress :
    a_dSdx = + Nx * phi * dx \
             + (kappa*H)**2 * (phi.dx(0)*Nx.dx(0) + phi.dx(1)*Nx.dx(1)) * dx
    L_dSdx = rho * g * H * dSdx * phi * dx \
    
    a_dSdy = + Ny * phi * dx \
             + (kappa*H)**2 * (phi.dx(0)*Ny.dx(0) + phi.dx(1)*Ny.dx(1)) * dx
    L_dSdy = rho * g * H * dSdy * phi*dx \
    
    # SUPG method :
    #if model.mesh.ufl_cell().topological_dimension() == 3:
    #  dS      = as_vector([d_x, d_y, 0.0])
    dS      = as_vector([d_x, d_y])
    phihat  = phi + h/(2*H) * ((H*dS[0]*phi).dx(0) + (H*dS[1]*phi).dx(1))
    #phihat  = phi + h/(2*H) * (H*dS[0]*phi.dx(0) + H*dS[1]*phi.dx(1))
    
    def L(u, uhat):
      #if model.mesh.ufl_cell().topological_dimension() == 3:
      #  return div(uhat)*u + dot(grad(u), uhat)
      #elif model.mesh.ufl_cell().topological_dimension() == 2:
      l = + (uhat[0].dx(0) + uhat[1].dx(1))*u \
          + u.dx(0)*uhat[0] + u.dx(1)*uhat[1]
      return l
    
    B = L(Ubar*H, dS) * phihat * dx
    a = adot * phihat * dx

    self.kappa  = kappa
    self.a_dSdx = a_dSdx
    self.a_dSdy = a_dSdy
    self.L_dSdx = L_dSdx
    self.L_dSdy = L_dSdy
    self.B      = B
    self.a      = a
  
  def solve(self, annotate=True):
    """
    Solve the balance velocity.
    """
    model = self.model
    
    s    = "::: solving BalanceVelocity :::"
    print_text(s, self.color())
    
    s    = "::: calculating surface gradient :::"
    print_text(s, self.color())
    
    dSdx   = project(model.S.dx(0), model.Q, annotate=annotate)
    dSdy   = project(model.S.dx(1), model.Q, annotate=annotate)
    model.assign_variable(model.dSdx, dSdx)
    model.assign_variable(model.dSdy, dSdy)
    print_min_max(model.dSdx, 'dSdx')
    print_min_max(model.dSdy, 'dSdy')
    
    # update velocity direction from driving stress :
    s    = "::: solving for smoothed x-component of driving stress " + \
           "with kappa = %f :::" % self.kappa
    print_text(s, self.color())
    solve(self.a_dSdx == self.L_dSdx, model.Nx, annotate=annotate)
    print_min_max(model.Nx, 'Nx')
    
    s    = "::: solving for smoothed y-component of driving stress :::"
    print_text(s, self.color())
    solve(self.a_dSdy == self.L_dSdy, model.Ny, annotate=annotate)
    print_min_max(model.Ny, 'Ny')
    
    # normalize the direction vector :
    s    =   "::: calculating normalized velocity direction" \
           + " from driving stress :::"
    print_text(s, self.color())
    d_x_v = model.Nx.vector().array()
    d_y_v = model.Ny.vector().array()
    d_n_v = np.sqrt(d_x_v**2 + d_y_v**2 + 1e-16)
    model.assign_variable(model.d_x, -d_x_v / d_n_v)
    model.assign_variable(model.d_y, -d_y_v / d_n_v)
    print_min_max(model.d_x, 'd_x')
    print_min_max(model.d_y, 'd_y')
    
    # calculate balance-velocity :
    s    = "::: solving velocity balance magnitude :::"
    print_text(s, self.color())
    solve(self.B == self.a, model.Ubar, annotate=annotate)
    print_min_max(model.Ubar, 'Ubar')
    
    # enforce positivity of balance-velocity :
    s    = "::: removing negative values of balance velocity :::"
    print_text(s, self.color())
    Ubar_v = model.Ubar.vector().array()
    Ubar_v[Ubar_v < 0] = 0
    model.assign_variable(model.Ubar, Ubar_v)
    print_min_max(model.Ubar, 'Ubar')



