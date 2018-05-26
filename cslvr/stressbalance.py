from dolfin            import *
from cslvr.physics     import Physics
from cslvr.inputoutput import print_text, print_min_max


class StressBalance(Physics):

  def __init__(self, model, momentum):
    """
    """
    s    = "::: INITIALIZING STRESS-BALANCE PHYSICS :::"
    print_text(s, self.color())
    
    # calculate viscosity in model.eta :
    U      = momentum.velocity()
    epsdot = momentum.effective_strain_rate(U) + model.eps_reg
    model.calc_eta(epsdot)
    
    # stress tensor :
    tau   = momentum.deviatoric_stress_tensor(U, model.eta)

    # rotate about the z-axis : 
    rad_xy = model.get_xy_velocity_angle(U) 
    Rz     = model.z_rotation_matrix(rad_xy)
    tau_r  = model.rotate_tensor(tau, Rz)
    
    # get surface stress :
    tau_ii_S = model.vert_extrude(tau_r[0,0], d='down', Q=model.Q_non_periodic)
    tau_ij_S = model.vert_extrude(tau_r[0,1], d='down', Q=model.Q_non_periodic)
    tau_iz_S = model.vert_extrude(tau_r[0,2], d='down', Q=model.Q_non_periodic)
    tau_ji_S = model.vert_extrude(tau_r[1,0], d='down', Q=model.Q_non_periodic)
    tau_jj_S = model.vert_extrude(tau_r[1,1], d='down', Q=model.Q_non_periodic)
    tau_jz_S = model.vert_extrude(tau_r[1,2], d='down', Q=model.Q_non_periodic)
    tau_zi_S = model.vert_extrude(tau_r[2,0], d='down', Q=model.Q_non_periodic)
    tau_zj_S = model.vert_extrude(tau_r[2,1], d='down', Q=model.Q_non_periodic)
    tau_zz_S = model.vert_extrude(tau_r[2,2], d='down', Q=model.Q_non_periodic)
    
    # get basal stress :
    tau_ii_B = model.vert_extrude(tau_r[0,0], d='up', Q=model.Q_non_periodic)
    tau_ij_B = model.vert_extrude(tau_r[0,1], d='up', Q=model.Q_non_periodic)
    tau_iz_B = model.vert_extrude(tau_r[0,2], d='up', Q=model.Q_non_periodic)
    tau_ji_B = model.vert_extrude(tau_r[1,0], d='up', Q=model.Q_non_periodic)
    tau_jj_B = model.vert_extrude(tau_r[1,1], d='up', Q=model.Q_non_periodic)
    tau_jz_B = model.vert_extrude(tau_r[1,2], d='up', Q=model.Q_non_periodic)
    tau_zi_B = model.vert_extrude(tau_r[2,0], d='up', Q=model.Q_non_periodic)
    tau_zj_B = model.vert_extrude(tau_r[2,1], d='up', Q=model.Q_non_periodic)
    tau_zz_B = model.vert_extrude(tau_r[2,2], d='up', Q=model.Q_non_periodic)
    
    # vertically integrate deviatoric stress (membrane stress) :
    t_ii = model.vert_integrate(tau_r[0,0], d='up', Q=model.Q_non_periodic)
    t_ij = model.vert_integrate(tau_r[0,1], d='up', Q=model.Q_non_periodic)
    t_iz = model.vert_integrate(tau_r[0,2], d='up', Q=model.Q_non_periodic)
    t_ji = model.vert_integrate(tau_r[1,0], d='up', Q=model.Q_non_periodic)
    t_jj = model.vert_integrate(tau_r[1,1], d='up', Q=model.Q_non_periodic)
    t_jz = model.vert_integrate(tau_r[1,2], d='up', Q=model.Q_non_periodic)
    t_zi = model.vert_integrate(tau_r[2,0], d='up', Q=model.Q_non_periodic)
    t_zj = model.vert_integrate(tau_r[2,1], d='up', Q=model.Q_non_periodic)
    t_zz = model.vert_integrate(tau_r[2,2], d='up', Q=model.Q_non_periodic)

    # extrude the integral down the vertical :
    N_ii = model.vert_extrude(t_ii, d='down', Q=model.Q_non_periodic)
    N_ij = model.vert_extrude(t_ij, d='down', Q=model.Q_non_periodic)
    N_iz = model.vert_extrude(t_iz, d='down', Q=model.Q_non_periodic)
    N_ji = model.vert_extrude(t_ji, d='down', Q=model.Q_non_periodic)
    N_jj = model.vert_extrude(t_jj, d='down', Q=model.Q_non_periodic)
    N_jz = model.vert_extrude(t_jz, d='down', Q=model.Q_non_periodic)
    N_zi = model.vert_extrude(t_zi, d='down', Q=model.Q_non_periodic)
    N_zj = model.vert_extrude(t_zj, d='down', Q=model.Q_non_periodic)
    N_zz = model.vert_extrude(t_zz, d='down', Q=model.Q_non_periodic)

    # save the membrane stresses :
    model.init_N_ii(N_ii)
    model.init_N_ij(N_ij)
    model.init_N_iz(N_iz)
    model.init_N_ji(N_ji)
    model.init_N_jj(N_jj)
    model.init_N_jz(N_jz)
    model.init_N_zi(N_zi)
    model.init_N_zj(N_zj)
    model.init_N_zz(N_zz)
    
    # get the components of horizontal velocity :
    u,v,w    = U.split(True)
    U        = as_vector([u, v])
    U_hat    = model.normalize_vector(U)
    U_n      = as_vector([U_hat[0],  U_hat[1], 0.0])
    U_t      = as_vector([U_hat[1], -U_hat[0], 0.0])
    S        = model.S
    B        = model.B

    # directional derivative in direction of flow :
    def d_di(u):  return dot(grad(u), U_n)

    # directional derivative in direction across flow :
    def d_dj(u):  return dot(grad(u), U_t)

    # form components :
    phi      = TestFunction(model.Q_non_periodic)
    dtau     = TrialFunction(model.Q_non_periodic)
    
    # mass matrix :
    self.M = assemble(phi*dtau*dx)

    # integrated stress-balance using Leibniz Theorem :
    self.M_ii = (d_di(N_ii) + tau_ii_S*d_di(S) - tau_ii_B*d_di(B)) * phi * dx
    self.M_ij = (d_dj(N_ij) + tau_ij_S*d_dj(S) - tau_ij_B*d_dj(B)) * phi * dx
    self.M_iz = (tau_iz_S - tau_iz_B) * phi * dx
    self.M_ji = (d_di(N_ji) + tau_ji_S*d_di(S) - tau_ji_B*d_di(B)) * phi * dx
    self.M_jj = (d_dj(N_jj) + tau_jj_S*d_dj(S) - tau_jj_B*d_dj(B)) * phi * dx
    self.M_jz = (tau_jz_S - tau_jz_B) * phi * dx
    self.M_zi = (d_di(N_zi) + tau_zi_S*d_di(S) - tau_zi_B*d_di(B)) * phi * dx
    self.M_zj = (d_dj(N_zj) + tau_zj_S*d_dj(S) - tau_zj_B*d_dj(B)) * phi * dx
    self.M_zz = (tau_zz_S - tau_zz_B) * phi * dx

  def solve(self):
    """
    """
    s    = "::: solving 'Stress_Balance' :::"
    print_text(s, self.color())
    
    model = self.model

    # solve the linear system :
    solve(self.M, model.M_ii.vector(), assemble(self.M_ii))
    print_min_max(model.M_ii, 'M_ii')
    solve(self.M, model.M_ij.vector(), assemble(self.M_ij))
    print_min_max(model.M_ij, 'M_ij')
    solve(self.M, model.M_iz.vector(), assemble(self.M_iz))
    print_min_max(model.M_iz, 'M_iz')
    solve(self.M, model.M_ji.vector(), assemble(self.M_ji))
    print_min_max(model.M_ji, 'M_ji')
    solve(self.M, model.M_jj.vector(), assemble(self.M_jj))
    print_min_max(model.M_jj, 'M_jj')
    solve(self.M, model.M_jz.vector(), assemble(self.M_jz))
    print_min_max(model.M_jz, 'M_jz')
    solve(self.M, model.M_zi.vector(), assemble(self.M_zi))
    print_min_max(model.M_zi, 'm_zi')
    solve(self.M, model.M_zj.vector(), assemble(self.M_zj))
    print_min_max(model.M_zj, 'M_zj')
    solve(self.M, model.M_zz.vector(), assemble(self.M_zz))
    print_min_max(model.M_zz, 'M_zz')



