from physics_new import Physics 

class FreeSurface(Physics):
  """
  """  
  def init_free_surface(self):
    """
    """
    s    = "::: INITIALIZING FREE-SURFACE PHYSICS :::"
    print_text(s, self.D3Model_color)
    
    # sigma coordinate :
    self.sigma = project((self.x[2] - self.B) / (self.S - self.B),
                         annotate=False)
    print_min_max(self.sigma, 'sigma')

    Q      = self.Q
    Q_flat = self.Q_flat

    phi    = TestFunction(Q_flat)
    dS     = TrialFunction(Q_flat)
    
    Shat   = Function(Q_flat) # surface elevation velocity 
    ahat   = Function(Q_flat) # accumulation velocity
    uhat   = Function(Q_flat) # horizontal velocity
    vhat   = Function(Q_flat) # horizontal velocity perp. to uhat
    what   = Function(Q_flat) # vertical velocity
    mhat   = Function(Q_flat) # mesh velocity
    dSdt   = Function(Q_flat) # surface height change
    M      = Function(Q_flat) # mass
    ds     = self.ds_flat
    dSurf  = ds(2)
    dBase  = ds(3)
    
    self.static_boundary = DirichletBC(Q, 0.0, self.ff_flat, 4)
    h = CellSize(self.flat_mesh)

    # upwinded trial function :
    unorm       = sqrt(self.uhat**2 + self.vhat**2 + 1e-1)
    upwind_term = h/(2.*unorm)*(self.uhat*phi.dx(0) + self.vhat*phi.dx(1))
    phihat      = phi + upwind_term

    mass_matrix = dS * phihat * dSurf
    lumped_mass = phi * dSurf

    stiffness_matrix = - self.uhat * self.Shat.dx(0) * phihat * dSurf \
                       - self.vhat * self.Shat.dx(1) * phihat * dSurf\
                       + (self.what + self.ahat) * phihat * dSurf
    
    # Calculate the nonlinear residual dependent scalar
    term1            = self.Shat.dx(0)**2 + self.Shat.dx(1)**2 + 1e-1
    term2            = + self.uhat*self.Shat.dx(0) \
                       + self.vhat*self.Shat.dx(1) \
                       - (self.what + self.ahat)
    C                = 10.0*h/(2*unorm) * term1 * term2**2
    diffusion_matrix = C * dot(grad(phi), grad(self.Shat)) * dSurf
    
    # Set up the Galerkin-least squares formulation of the Stokes' functional
    A_pro         = - phi.dx(2)*dS*dx - dS*phi*dBase + dSdt*phi*dSurf 
    M.vector()[:] = 1.0
    self.M        = M*dx

    self.newz                   = Function(self.Q)
    self.mass_matrix            = mass_matrix
    self.stiffness_matrix       = stiffness_matrix
    self.diffusion_matrix       = diffusion_matrix
    self.lumped_mass            = lumped_mass
    self.A_pro                  = A_pro
    self.Shat                   = Shat
    self.ahat                   = ahat
    self.uhat                   = uhat
    self.vhat                   = vhat
    self.what                   = what
    self.mhat                   = mhat
    self.dSdt                   = dSdt
    
  def solve_free_surface(self):
    """
    """
    config = self.config
   
    self.assign_variable(self.Shat, self.S) 
    self.assign_variable(self.ahat, self.adot) 
    self.assign_variable(self.uhat, self.u) 
    self.assign_variable(self.vhat, self.v) 
    self.assign_variable(self.what, self.w) 

    m = assemble(self.mass_matrix,      keep_diagonal=True)
    r = assemble(self.stiffness_matrix, keep_diagonal=True)

    s    = "::: solving free-surface :::"
    print_text(s, self.D3Model_color)
    if config['free_surface']['lump_mass_matrix']:
      m_l = assemble(self.lumped_mass)
      m_l = m_l.get_local()
      m_l[m_l==0.0]=1.0
      m_l_inv = 1./m_l

    if config['free_surface']['static_boundary_conditions']:
      self.static_boundary.apply(m,r)

    if config['free_surface']['use_shock_capturing']:
      k = assemble(self.diffusion_matrix)
      r -= k
      print_min_max(r, 'D')

    if config['free_surface']['lump_mass_matrix']:
      self.assign_variable(self.dSdt, m_l_inv * r.get_local())
    else:
      m.ident_zeros()
      solve(m, self.dSdt.vector(), r, annotate=False)

    A = assemble(lhs(self.A_pro))
    p = assemble(rhs(self.A_pro))
    q = Vector()  
    solve(A, q, p, annotate=False)
    self.assign_variable(self.dSdt, q)
