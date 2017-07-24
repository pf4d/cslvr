from helper            import raiseNotDefined
from fenics            import *
from dolfin_adjoint    import *
from cslvr.inputoutput import get_text, print_text, print_min_max
import numpy               as np
import matplotlib.pyplot   as plt
import os


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

  def form_reg_ftn(self, c, integral, kind='TV'):
    r"""
    Formulates a regularization functional for use 
    with optimization of the control parameter :math:`c` given by ``c`` 
    over the integral ``integral``.

    The choices for ``kind`` are :

    1. ``Tikhonov`` -- Tikhonov regularization

    .. math::

      \mathscr{R}(c) = \frac{1}{2} \int_{\Gamma} \nabla c \cdot \nabla c\ d\Gamma

    2. ``TV`` -- total variation regularization

    .. math::

      \mathscr{R}(c) = \int_{\Gamma} \left( \nabla c \cdot \nabla c + c_0 \right)^{\frac{1}{2}}\ d\Gamma,

    3. ``square`` -- squared regularization

    .. math::

      \mathscr{R}(c) = \frac{1}{2} \int_{\Gamma} c^2\ d\Gamma,

    4. ``abs`` -- absolute regularization

    .. math::
    
      \mathscr{R}(c) = \int_{\Gamma} |c|\ d\Gamma,

    :param c:         the control variable
    :param integral:  measure over which to integrate 
                      (see :func:`~model.calculate_boundaries`)
    :param kind:      kind of regularization to use
    :type c:          :class:`~fenics.Function`
    :type integral:   int 
    :type kind:       string
    """
    s  = "::: forming '%s' regularization functional for variable " + \
         " '%s'integrated over the %s :::"
    print_text(s % (kind, c.name(), integral.description), self.color())

    dR = integral()
   
    # the various possible regularization types : 
    kinds = ['TV', 'Tikhonov', 'square', 'abs']
    
    # form regularization term 'R' :
    if kind not in kinds:
      s    =   ">>> VALID REGULARIZATIONS ARE 'TV', 'Tikhonov', 'square'," + \
               " or 'abs' <<<"
      print_text(s, 'red', 1)
      sys.exit(1)

    elif kind == 'TV':
      R = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR

    elif kind == 'Tikhonov':
      R = 0.5 * inner(grad(c), grad(c)) * dR

    elif kind == 'square':
      R = 0.5 * c**2 * dR

    elif kind == 'abs':
      R = abs(c) * dR

    return R

  def form_obj_ftn(self, u, u_ob, integral, kind='log', eps=0.01):
    """
    Forms and returns an objective functional for use with adjoint.
    """
    s  = "::: forming '%s' objective functional integrated over %s :::"
    print_text(s % (kind, integral.description), self.color())

    dJ = integral()

    if type(u) != list and type(u_ob) != list:
      u    = [u]
      u_ob = [u_ob]
    # for a UFL expression for the norm and difference :
    U     = 0
    U_ob  = 0
    U_err = 0
    for Ui, U_obi in zip(u, u_ob):
      U     += Ui**2
      U_ob  += U_obi**2
      U_err += (Ui - U_obi)**2
    U    = sqrt(U)
    U_ob = sqrt(U_ob)

    if kind == 'log':
      J  = 0.5 * ln( (U + eps) / (U_ob + eps) )**2 * dJ 

    elif kind == 'l2':
      J  = 0.5 * U_err * dJ

    elif kind == 'ratio':
      J  = 0.5 * (1 -  (U + eps) / (U_ob + eps))**2 * dJ

    elif kind == 'abs': 
      J  = abs(U_err) * dJ

    else:
      s = ">>> ADJOINT OBJECTIVE FUNCTIONAL MAY BE 'l2', " + \
          "'log', 'ratio', OR 'abs', NOT '%s' <<<" % kind
      print_text(s, 'red', 1)
      sys.exit(1)
    return J

  def calc_functionals(self, u, u_ob, control, J_measure, R_measure, eps=0.01):
    """
    Used to facilitate printing the objective function in adjoint solves.
    """
    s   = "::: calculating functionals :::"
    print_text(s, cls=self)

    color = '208'

    # ensure we can iterate :
    if type(u) != list and type(u_ob) != list:
      u    = [u]
      u_ob = [u_ob]
    if type(control) != list:    control = [control]
    if type(R_measure) != list:  R_measure = [R_measure]

    # for a UFL expression for the norm and difference :
    U     = 0
    U_ob  = 0
    U_err = 0
    for Ui, U_obi in zip(u, u_ob):
      U     += Ui**2
      U_ob  += U_obi**2
      U_err += (Ui - U_obi)**2
    U    = sqrt(U)
    U_ob = sqrt(U_ob)

    dJ = J_measure()

    J_log   = 0.5 * ln( (U + eps) / (U_ob + eps) )**2 * dJ 
    J_l2    = 0.5 * U_err * dJ
    J_rat   = 0.5 * (1 -  (U + eps) / (U_ob + eps))**2 * dJ
    J_abs   = abs(U_err) * dJ

    J_log_a = assemble(J_log, annotate=False)
    J_l2_a  = assemble(J_l2,  annotate=False)
    J_rat_a = assemble(J_rat, annotate=False)
    J_abs_a = assemble(J_abs, annotate=False)
      
    print_min_max(J_log_a, 'J_log \t\t', color=color)
    print_min_max(J_l2_a,  'J_l2  \t\t', color=color)
    print_min_max(J_rat_a, 'J_rat \t\t', color=color)
    print_min_max(J_abs_a, 'J_abs \t\t', color=color)

    J_a = [J_log_a, J_l2_a,  J_rat_a, J_abs_a]

    R_a = []
    for c, R_m in zip(control, R_measure):
      dR      = R_m()
      R_tv    = sqrt(inner(grad(c), grad(c)) + 1e-15) * dR
      R_tik   = 0.5 * inner(grad(c), grad(c)) * dR
      R_sq    = 0.5 * c**2 * dR
      R_abs   = abs(c) * dR
      R_tv_a  = assemble(R_tv,  annotate=False)
      R_tik_a = assemble(R_tik, annotate=False)
      R_sq_a  = assemble(R_sq,  annotate=False)
      R_abs_a = assemble(R_abs, annotate=False)
      print_min_max(R_tv_a,  'R_tv  : %s \t' % c.name(), color=color)
      print_min_max(R_tik_a, 'R_tik : %s \t' % c.name(), color=color)
      print_min_max(R_sq_a,  'R_sq  : %s \t' % c.name(), color=color)
      print_min_max(R_abs_a, 'R_abs : %s \t' % c.name(), color=color)
      R_a.append([R_tv_a, R_tik_a, R_sq_a, R_abs_a])
    
    return [J_a, R_a]

  def plot_ftnls(self, t0, tf, J, R, control):
    """
    Save all the objective functional values with rudimentary plot.
    """
    model = self.model
    d     = model.out_dir + 'objective_ftnls_history/'
    
    s    = '::: saving objective functionals to %s :::'
    print_text(s % d, cls=self)

    # for images :
    J_lab = r'$\mathscr{J}_{\mathrm{%s}}$'
    R_lab = r'$\mathscr{R}_{\mathrm{%s}}\left( %s \right)$'

    # for subscripts of file names :
    J_sub = ['log', 'l2',  'rat', 'abs']
    R_sub = ['tv',  'tik', 'sq',  'abs']
    
    if model.MPI_rank==0:
      if not os.path.exists(d):
        os.makedirs(d)
      for i,sub in enumerate(J_sub):
        np.savetxt(d + 'J_%s.txt' % sub, J[:,i])
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_ylabel(J_lab % sub)
        ax.set_xlabel(r'iteration')
        ax.plot(J[:,i], 'r-', lw=2.0)
        plt.grid()
        plt.savefig(d + 'J_%s.png' % sub, dpi=100)
        plt.close(fig)
      for j, c in enumerate(control):
        for i, sub in enumerate(R_sub):
          np.savetxt(d + 'R_%s_%s.txt' % (sub, c.name()), R[:,j][:,i]) 
          fig = plt.figure()
          ax  = fig.add_subplot(111)
          ax.set_yscale('log')
          ax.set_ylabel(R_lab % (sub, c.name()))
          ax.set_xlabel(r'iteration')
          ax.plot(R[:,j][:,i], 'r-', lw=2.0)
          plt.grid()
          plt.savefig(d + 'R_%s_%s.png' % (sub, c.name()), dpi=100)
          plt.close(fig)

      # save the total time to compute :
      np.savetxt(d + 'time.txt',  np.array([tf - t0]))
      
  def optimize(self, u, u_ob, I, control, J_measure, R_measure, bounds,
               method            = 'ipopt',
               max_iter          = 100,
               adj_save_vars     = None,
               adj_callback      = None,
               post_adj_callback = None):
    """
    """
    s    = "::: solving optimal control to minimize ||u - u_ob|| with " + \
           "control parameter%s :::"
    if type(control) != list:
      control = [control]
    tx = 's '
    for i in control:
      tx += "'" + i.name() + "'"
      if i != control[-1]: tx += ' and '
    print_text(s % tx, cls=self)

    model = self.model

    # reset entire dolfin-adjoint state :
    adj_reset()

    # starting time :
    t0   = time()

    # need this for the derivative callback :
    global counter
    counter = 0 
    
    # functional lists to be populated :
    global J_a, R_a
    J_a, R_a = [],[]
    # solve the physics with annotation enabled :
    s    = '::: solving forward problem :::'
    print_text(s, cls=self)
    self.solve(annotate=True)
    
    # now solve the control optimization problem : 
    s    = "::: starting adjoint-control optimization with method '%s' :::"
    print_text(s % method, cls=self)

    # objective function callback function : 
    def eval_cb(I, c):
      s    = '::: adjoint objective eval post callback function :::'
      print_text(s, cls=self)
      print_min_max(I,    'I')
      for ci in c:
        print_min_max(ci,    'control: ' + ci.name())
    
    # objective gradient callback function :
    def deriv_cb(I, dI, c):
      global counter, J_a, R_a
      if method == 'ipopt':
        s0    = '>>> '
        s1    = 'iteration %i (max %i) complete'
        s2    = ' <<<'
        text0 = get_text(s0, 'red', 1)
        text1 = get_text(s1 % (counter, max_iter), 'red')
        text2 = get_text(s2, 'red', 1)
        if MPI.rank(mpi_comm_world())==0:
          print text0 + text1 + text2
        counter += 1
      s    = '::: adjoint obj. gradient post callback function :::'
      print_text(s, cls=self)
      for (dIi,ci) in zip(dI,c):
        print_min_max(dIi,    'dI/control: ' + ci.name())
        self.model.save_xdmf(dIi, 'dI_control_' + ci.name())
        self.model.save_xdmf(ci, 'control_' + ci.name())
      
      # update the DA current velocity to the model for evaluation 
      # purposes only;
      u_opt = DolfinAdjointVariable(u).tape_value()

      for i in range(len(control)):
        control[i].assign(c[i], annotate=False)

      # print functional values :
      ftnls = self.calc_functionals(u_opt, u_ob, control, J_measure, R_measure)

      # add to the list of functional evaluations :
      J_a.append(ftnls[0])
      R_a.append(ftnls[1])

      # call that callback, if you want :
      if adj_callback is not None:
        adj_callback(I, dI, c)
   
    # define the control parameter :
    m = []
    for i in control:
      m.append(Control(i, value=i))
    
    # create the reduced functional to minimize :
    F = ReducedFunctional(Functional(I), m, eval_cb_post=eval_cb,
                          derivative_cb_post=deriv_cb)

    # optimize with scipy's fmin_l_bfgs_b :
    if method == 'l_bfgs_b': 
      out = minimize(F, method="L-BFGS-B", tol=1e-9, bounds=bounds,
                     options={"disp"    : True,
                              "maxiter" : max_iter,
                              "gtol"    : 1e-5})
      b_opt = out
    
    # or optimize with IPOpt (preferred) :
    elif method == 'ipopt':
      try:
        import pyipopt
      except ImportError:
        info_red("""You do not have IPOPT and/or pyipopt installed.
                    When compiling IPOPT, make sure to link against HSL,
                    as it is a necessity for practical problems.""")
        raise
      problem = MinimizationProblem(F, bounds=bounds)
      parameters = {"tol"                : 1e-8,
                    "acceptable_tol"     : 1e-6,
                    "maximum_iterations" : max_iter,
                    "print_level"        : 5,
                    "ma97_order"         : "metis",
                    "linear_solver"      : "ma97"}
      solver = IPOPTSolver(problem, parameters=parameters)
      b_opt  = solver.solve()

    # make the optimal control parameter available :
    for c,b in zip(control, b_opt):
      model.assign_variable(c, b)
    
    # call the post-adjoint callback function if set :
    if post_adj_callback is not None:
      s    = '::: calling optimize_ubar() post-adjoined callback function :::'
      print_text(s, cls=self)
      post_adj_callback()
    
    # save state to unique hdf5 file :
    if isinstance(adj_save_vars, list):
      s    = '::: saving variables in list arg adj_save_vars :::'
      print_text(s, cls=self)
      out_file = model.out_dir + 'u_opt.h5'
      foutput  = HDF5File(mpi_comm_world(), out_file, 'w')
      
      for var in adj_save_vars:
        model.save_hdf5(var, f=foutput)
      
      foutput.close()

    # calculate total time to compute
    tf = time()
    s  = tf - t0
    m  = s / 60.0
    h  = m / 60.0
    s  = s % 60
    m  = m % 60
    text = "time to optimize ||u - u_ob||: %02d:%02d:%02d" % (h,m,s)
    print_text(text, 'red', 1)

    # convert lists to arrays :
    J_a = np.array(J_a)
    R_a = np.array(R_a)

    # save a rudimentary plot of functionals
    self.plot_ftnls(t0, tf, J_a, R_a, control)

  def solve(self):
    """
    Solves the physics calculation.
    """
    raiseNotDefined()



