from physics_new import Physics
from varglas.io  import print_text, print_min_max
from fenics      import *

class SSA_Balance(Physics):

  def __init__(self, model):
    """
    """
    s    = "::: INITIALIZING SSA-BALANCE PHYSICS :::"
    print_text(s, self.color())

    mesh     = model.mesh
    Q2       = model.Q2
    ubar     = model.ubar
    vbar     = model.vbar
    S        = model.S
    B        = model.B
    H        = S - B
    beta     = model.beta
    rhoi     = model.rhoi
    g        = model.g
    etabar   = model.etabar
    
    Phi      = TestFunction(Q2)
    phi, psi = split(Phi)
    dU       = TrialFunction(Q2)
    du, dv   = split(dU)
    
    U        = as_vector([ubar, vbar])
    U_nm     = model.normalize_vector(U)
    U_n      = as_vector([U_nm[0],  U_nm[1]])
    U_t      = as_vector([U_nm[1], -U_nm[0]])
    
    s        = dot(dU, U_n)
    t        = dot(dU, U_t)
    U_s      = as_vector([s, t])
    grads    = as_vector([s.dx(0), s.dx(1)])
    gradt    = as_vector([t.dx(0), t.dx(1)])
    dsdi     = dot(grads, U_n)
    dsdj     = dot(grads, U_t)
    dtdi     = dot(gradt, U_n)
    dtdj     = dot(gradt, U_t)
    gradphi  = as_vector([phi.dx(0), phi.dx(1)]) 
    gradpsi  = as_vector([psi.dx(0), psi.dx(1)]) 
    gradS    = as_vector([S.dx(0),   S.dx(1)  ]) 
    dphidi   = dot(gradphi, U_n)
    dphidj   = dot(gradphi, U_t)
    dpsidi   = dot(gradpsi, U_n)
    dpsidj   = dot(gradpsi, U_t)
    dSdi     = dot(gradS,   U_n)
    dSdj     = dot(gradS,   U_t)
    gradphi  = as_vector([dphidi,    dphidj])
    gradpsi  = as_vector([dpsidi,    dpsidj])
    gradS    = as_vector([dSdi,      dSdj  ])
    
    epi_1  = as_vector([2*dsdi + dtdj, 
                        0.5*(dsdj + dtdi) ])
    epi_2  = as_vector([0.5*(dsdj + dtdi),
                        2*dtdj + dsdi     ])
    
    tau_id = phi * rhoi * g * H * gradS[0] * dx
    tau_jd = psi * rhoi * g * H * gradS[1] * dx

    tau_ib = - beta**2 * s * phi * dx
    tau_jb = - beta**2 * t * psi * dx

    tau_1  = - 2 * etabar * H * dot(epi_1, gradphi) * dx
    tau_2  = - 2 * etabar * H * dot(epi_2, gradpsi) * dx
    
    delta_1  = tau_1 + tau_ib - tau_id
    delta_2  = tau_2 + tau_jb - tau_jd
    
    delta  = delta_1 + delta_2
    U_s    = Function(Q2)

    # make the variables available to solve :
    self.delta = delta
    self.U_s   = U_s
    self.U_n   = U_n
    self.U_t   = U_t
    
  def solve(self):
    """
    """
    model = self.model
    
    s    = "::: solving 'SSA_Balance' for flow direction :::"
    print_text(s, self.color())
    solve(lhs(self.delta) == rhs(self.delta), self.U_s)
    u_s, u_t = self.U_s.split(True)
    model.assign_variable(model.u_s, u_s)
    model.assign_variable(model.u_t, u_t)
    print_min_max(model.u_s, 'u_s')
    print_min_max(model.u_t, 'u_t')
    
    # solve for the individual stresses now that the corrected velocities 
    # are found.
    self.solve_component_stress()

  def solve_component_stress(self):  
    """
    """
    model  = self.model
    
    s      = "::: solving 'SSA_Balance' for stresses :::" 
    print_text(s, self.color())

    Q       = model.Q
    beta    = model.beta
    S       = model.S
    B       = model.B
    H       = S - B
    rhoi    = model.rhoi
    g       = model.g
    etabar  = model.etabar
    
    Q       = model.Q
    U_s     = self.U_s
    U_n     = self.U_n
    U_t     = self.U_t

    phi     = TestFunction(Q)
    dtau    = TrialFunction(Q)
    
    s       = dot(U_s, U_n)
    t       = dot(U_s, U_t)
    grads   = as_vector([s.dx(0), s.dx(1)])
    gradt   = as_vector([t.dx(0), t.dx(1)])
    dsdi    = dot(grads, U_n)
    dsdj    = dot(grads, U_t)
    dtdi    = dot(gradt, U_n)
    dtdj    = dot(gradt, U_t)
    gradphi = as_vector([phi.dx(0), phi.dx(1)]) 
    gradS   = as_vector([S.dx(0),   S.dx(1)  ]) 
    dphidi  = dot(gradphi, U_n)
    dphidj  = dot(gradphi, U_t)
    dSdi    = dot(gradS,   U_n)
    dSdj    = dot(gradS,   U_t)
    gradphi = as_vector([dphidi, dphidj])
    gradS   = as_vector([dSdi,   dSdj  ])
    
    epi_1  = as_vector([2*dsdi + dtdj, 
                        0.5*(dsdj + dtdi) ])
    epi_2  = as_vector([0.5*(dsdj + dtdi),
                        2*dtdj + dsdi     ])
    
    tau_id_s = phi * rhoi * g * H * gradS[0] * dx
    tau_jd_s = phi * rhoi * g * H * gradS[1] * dx

    tau_ib_s = - beta**2 * s * phi * dx
    tau_jb_s = - beta**2 * t * phi * dx

    tau_ii_s = - 2 * etabar * H * epi_1[0] * gradphi[0] * dx
    tau_ij_s = - 2 * etabar * H * epi_1[1] * gradphi[1] * dx

    tau_ji_s = - 2 * etabar * H * epi_2[0] * gradphi[0] * dx
    tau_jj_s = - 2 * etabar * H * epi_2[1] * gradphi[1] * dx
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # solve the linear system :
    solve(M, model.tau_id.vector(), assemble(tau_id_s))
    print_min_max(model.tau_id, 'tau_id')
    solve(M, model.tau_jd.vector(), assemble(tau_jd_s))
    print_min_max(model.tau_jd, 'tau_jd')
    solve(M, model.tau_ib.vector(), assemble(tau_ib_s))
    print_min_max(model.tau_ib, 'tau_ib')
    solve(M, model.tau_jb.vector(), assemble(tau_jb_s))
    print_min_max(model.tau_jb, 'tau_jb')
    solve(M, model.tau_ii.vector(), assemble(tau_ii_s))
    print_min_max(model.tau_ii, 'tau_ii')
    solve(M, model.tau_ij.vector(), assemble(tau_ij_s))
    print_min_max(model.tau_ij, 'tau_ij')
    solve(M, model.tau_ji.vector(), assemble(tau_ji_s))
    print_min_max(model.tau_ji, 'tau_ji')
    solve(M, model.tau_jj.vector(), assemble(tau_jj_s))
    print_min_max(model.tau_jj, 'tau_jj')
   

class BP_Balance(Physics):

  def __init__(self, model, config):
    """
    """
    s    = "::: INITIALIZING BP-BALANCE PHYSICS :::"
    print_text(s, self.color())

    self.model  = model
    self.config = config

    mesh     = model.mesh
    Q        = model.Q
    Q2       = model.Q2
    u        = model.u
    v        = model.v
    S        = model.S
    B        = model.B
    H        = S - B
    beta     = model.beta
    rhoi     = model.rhoi
    rhow     = model.rhow
    g        = model.g
    x        = model.x
    N        = model.N
    D        = model.D
    eta      = model.eta
    
    dx       = model.dx
    dx_s     = dx(1)
    dx_g     = dx(0)
    dx       = dx(1) + dx(0) # entire internal
    ds       = model.ds  
    dGnd     = ds(3)         # grounded bed
    dFlt     = ds(5)         # floating bed
    dSde     = ds(4)         # sides
    dBed     = dGnd + dFlt   # bed
    
    f_w      = rhoi*g*(S - x[2]) + rhow*g*D
    
    Phi      = TestFunction(Q2)
    phi, psi = split(Phi)
    dU       = TrialFunction(Q2)
    du, dv   = split(dU)
    
    U        = as_vector([u, v])
    U_nm     = model.normalize_vector(U)
    U_n      = as_vector([U_nm[0],  U_nm[1]])
    U_t      = as_vector([U_nm[1], -U_nm[0]])
    
    s        = dot(dU, U_n)
    t        = dot(dU, U_t)
    U_s      = as_vector([s,       t      ])
    grads    = as_vector([s.dx(0), s.dx(1)])
    gradt    = as_vector([t.dx(0), t.dx(1)])
    dsdi     = dot(grads, U_n)
    dsdj     = dot(grads, U_t)
    dsdz     = s.dx(2)
    dtdi     = dot(gradt, U_n)
    dtdj     = dot(gradt, U_t)
    dtdz     = t.dx(2)
    gradphi  = as_vector([phi.dx(0), phi.dx(1)])
    gradpsi  = as_vector([psi.dx(0), psi.dx(1)])
    gradS    = as_vector([S.dx(0),   S.dx(1)  ])
    dphidi   = dot(gradphi, U_n)
    dphidj   = dot(gradphi, U_t)
    dpsidi   = dot(gradpsi, U_n)
    dpsidj   = dot(gradpsi, U_t)
    dSdi     = dot(gradS,   U_n)
    dSdj     = dot(gradS,   U_t)
    gradphi  = as_vector([dphidi,    dphidj,  phi.dx(2)])
    gradpsi  = as_vector([dpsidi,    dpsidj,  psi.dx(2)])
    gradS    = as_vector([dSdi,      dSdj,    S.dx(2)  ])
    
    epi_1  = as_vector([2*dsdi + dtdj, 
                        0.5*(dsdj + dtdi),
                        0.5*dsdz             ])
    epi_2  = as_vector([0.5*(dsdj + dtdi),
                             dsdi + 2*dtdj,
                        0.5*dtdz             ])
    
    F_id = phi * rhoi * g * gradS[0] * dx
    F_jd = psi * rhoi * g * gradS[1] * dx
    
    F_ib = - beta**2 * s * phi * dBed
    F_jb = - beta**2 * t * psi * dBed
    
    F_ip = f_w * N[0] * phi * dSde
    F_jp = f_w * N[1] * psi * dSde
    
    F_1  = - 2 * eta * dot(epi_1, gradphi) * dx
    F_2  = - 2 * eta * dot(epi_2, gradpsi) * dx
    
    delta_1  = F_1 + F_ib + F_ip - F_id
    delta_2  = F_2 + F_jb + F_jp - F_jd
    
    delta  = delta_1 + delta_2
    U_s    = Function(Q2)

    # make the variables available to solve :
    self.delta = delta
    self.U_nm  = U_nm
    self.U_s   = U_s
    self.U_n   = U_n
    self.U_t   = U_t
    self.f_w   = f_w
    
  def solve(self):
    """
    """
    model = self.model
    model.calc_eta()
    
    s    = "::: solving 'BP_Balance' for flow direction :::"
    print_text(s, self.color())
    solve(lhs(self.delta) == rhs(self.delta), self.U_s)
    u_s, u_t = self.U_s.split(True)
    model.assign_variable(model.u_s, u_s)
    model.assign_variable(model.u_t, u_t)
    print_min_max(model.u_s, 'u_s')
    print_min_max(model.u_t, 'u_t')

  def solve_component_stress(self):  
    """
    """
    model  = self.model
    config = self.config
    
    s    = "solving 'BP_Balance' for internal forces :::" 
    print_text(s, self.color())

    Q       = model.Q
    N       = model.N
    beta    = model.beta
    S       = model.S
    B       = model.B
    H       = S - B
    rhoi    = model.rhoi
    g       = model.g
    eta     = model.eta
    
    dx      = model.dx
    dx_s    = dx(1)
    dx_g    = dx(0)
    dx      = dx(1) + dx(0) # entire internal
    ds      = model.ds  
    dGnd    = ds(3)         # grounded bed
    dFlt    = ds(5)         # floating bed
    dSde    = ds(4)         # sides
    dBed    = dGnd + dFlt   # bed
    
    # solve with corrected velociites :
    model   = self.model
    config  = self.config

    Q       = model.Q
    f_w     = self.f_w
    U_s     = self.U_s
    U_n     = self.U_n
    U_t     = self.U_t
    U_nm    = self.U_nm

    phi     = TestFunction(Q)
    dtau    = TrialFunction(Q)
    
    s       = dot(U_s, U_n)
    t       = dot(U_s, U_t)
    U_s     = as_vector([s,       t      ])
    grads   = as_vector([s.dx(0), s.dx(1)])
    gradt   = as_vector([t.dx(0), t.dx(1)])
    dsdi    = dot(grads, U_n)
    dsdj    = dot(grads, U_t)
    dsdz    = s.dx(2)
    dtdi    = dot(gradt, U_n)
    dtdj    = dot(gradt, U_t)
    dtdz    = t.dx(2)
    dwdz    = -(dsdi + dtdj)
    gradphi = as_vector([phi.dx(0), phi.dx(1)])
    gradS   = as_vector([S.dx(0),   S.dx(1)  ])
    dphidi  = dot(gradphi, U_n)
    dphidj  = dot(gradphi, U_t)
    dSdi    = dot(gradS,   U_n)
    dSdj    = dot(gradS,   U_t)
    gradphi = as_vector([dphidi, dphidj, phi.dx(2)])
    gradS   = as_vector([dSdi,   dSdj,   S.dx(2)  ])
    
    epi_1   = as_vector([dsdi, 
                         0.5*(dsdj + dtdi),
                         0.5*dsdz             ])
    epi_2   = as_vector([0.5*(dtdi + dsdj),
                         dtdj,
                         0.5*dtdz             ])
    
    F_id_s = + phi * rhoi * g * gradS[0] * dx \
             - 2 * eta * dwdz * dphidi * dx #\
    #         + 2 * eta * dwdz * phi * N[0] * U_n[0] * ds
    F_jd_s = + phi * rhoi * g * gradS[1] * dx \
             - 2 * eta * dwdz * dphidj * dx #\
    #         + 2 * eta * dwdz * phi * N[1] * U_n[1] * ds
    
    F_ib_s = - beta**2 * s * phi * dBed
    F_jb_s = - beta**2 * t * phi * dBed
    
    F_ip_s = f_w * N[0] * phi * dSde
    F_jp_s = f_w * N[1] * phi * dSde
    
    F_pn_s = f_w * N[0] * phi * dSde
    F_pt_s = f_w * N[1] * phi * dSde
     
    F_ii_s = - 2 * eta * epi_1[0] * gradphi[0] * dx# \
    #         + 2 * eta * epi_1[0] * phi * N[0] * U_n[0] * ds
    #         + f_w * N[0] * phi * U_n[0] * dSde
    F_ij_s = - 2 * eta * epi_1[1] * gradphi[1] * dx# \
    #         + 2 * eta * epi_1[1] * phi * N[1] * U_n[1] * ds
    #         + f_w * N[1] * phi * U_n[1] * dSde
    F_iz_s = - 2 * eta * epi_1[2] * gradphi[2] * dx + F_ib_s #\
    #        + 2 * eta * epi_1[2] * phi * N[2] * ds
     
    F_ji_s = - 2 * eta * epi_2[0] * gradphi[0] * dx# \
    #         + 2 * eta * epi_2[0] * phi * N[0] * U_t[0] * ds
    #         + f_w * N[0] * phi * U_t[0] * dSde
    F_jj_s = - 2 * eta * epi_2[1] * gradphi[1] * dx# \
    #         + 2 * eta * epi_2[0] * phi * N[1] * U_t[1] * ds
    #         + f_w * N[1] * phi * U_t[1] * dSde
    F_jz_s = - 2 * eta * epi_2[2] * gradphi[2] * dx + F_jb_s #\
    #         + 2 * eta * epi_2[2] * phi * N[2] * ds
    
    # mass matrix :
    M = assemble(phi*dtau*dx)
    
    # solve the linear system :
    solve(M, model.F_id.vector(), assemble(F_id_s))
    print_min_max(model.F_id, 'F_id')
    solve(M, model.F_jd.vector(), assemble(F_jd_s))
    print_min_max(model.F_jd, 'F_jd')
    solve(M, model.F_ib.vector(), assemble(F_ib_s))
    print_min_max(model.F_ib, 'F_ib')
    solve(M, model.F_jb.vector(), assemble(F_jb_s))
    print_min_max(model.F_jb, 'F_jb')
    solve(M, model.F_ip.vector(), assemble(F_ip_s))
    print_min_max(model.F_ip, 'F_ip')
    solve(M, model.F_jp.vector(), assemble(F_jp_s))
    print_min_max(model.F_jp, 'F_jp')
    solve(M, model.F_ii.vector(), assemble(F_ii_s))
    print_min_max(model.F_ii, 'F_ii')
    solve(M, model.F_ij.vector(), assemble(F_ij_s))
    print_min_max(model.F_ij, 'F_ij')
    solve(M, model.F_iz.vector(), assemble(F_iz_s))
    print_min_max(model.F_iz, 'F_iz')
    solve(M, model.F_ji.vector(), assemble(F_ji_s))
    print_min_max(model.F_ji, 'F_ji')
    solve(M, model.F_jj.vector(), assemble(F_jj_s))
    print_min_max(model.F_jj, 'F_jj')
    solve(M, model.F_jz.vector(), assemble(F_jz_s))
    print_min_max(model.F_jz, 'F_jz')
   
    if config['stokes_balance']['vert_integrate']: 
      s    = "::: vertically integrating 'BP_Balance' internal forces :::"
      print_text(s, self.color())
      
      tau_ii   = model.vert_integrate(model.F_ii, d='down')
      tau_ij   = model.vert_integrate(model.F_ij, d='down')
      tau_iz   = model.vert_integrate(model.F_iz, d='down')
                                                
      tau_ji   = model.vert_integrate(model.F_ji, d='down')
      tau_jj   = model.vert_integrate(model.F_jj, d='down')
      tau_jz   = model.vert_integrate(model.F_jz, d='down')
                                                
      tau_id   = model.vert_integrate(model.F_id, d='down')
      tau_jd   = model.vert_integrate(model.F_jd, d='down')
                                                
      tau_ip   = model.vert_integrate(model.F_ip, d='down')
      tau_jp   = model.vert_integrate(model.F_jp, d='down')
      
      tau_ib   = model.vert_extrude(model.F_ib, d='up')
      tau_jb   = model.vert_extrude(model.F_jb, d='up')
     
      model.assign_variable(model.tau_id, tau_id)
      model.assign_variable(model.tau_jd, tau_jd)
      model.assign_variable(model.tau_ib, tau_ib)
      model.assign_variable(model.tau_jb, tau_jb)
      model.assign_variable(model.tau_ip, tau_ip)
      model.assign_variable(model.tau_jp, tau_jp)
      model.assign_variable(model.tau_ii, tau_ii)
      model.assign_variable(model.tau_ij, tau_ij)
      model.assign_variable(model.tau_iz, tau_iz)
      model.assign_variable(model.tau_ji, tau_ji)
      model.assign_variable(model.tau_jj, tau_jj)
      model.assign_variable(model.tau_jz, tau_jz)
    
      print_min_max(model.tau_id, 'tau_id')
      print_min_max(model.tau_jd, 'tau_jd')
      print_min_max(model.tau_ib, 'tau_ib')
      print_min_max(model.tau_jb, 'tau_jb')
      print_min_max(model.tau_ip, 'tau_ip')
      print_min_max(model.tau_jp, 'tau_jp')
      print_min_max(model.tau_ii, 'tau_ii')
      print_min_max(model.tau_ij, 'tau_ij')
      print_min_max(model.tau_iz, 'tau_iz')
      print_min_max(model.tau_ji, 'tau_ji')
      print_min_max(model.tau_jj, 'tau_jj')
      print_min_max(model.tau_jz, 'tau_jz')
