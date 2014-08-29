from varglas.model              import Model
from varglas.solvers            import TransientSolver
from varglas.physical_constants import IceParameters
from varglas.helper             import default_nonlin_solver_params, \
                                       default_config
from fenics                     import set_log_active, File, Expression, pi, \
                                       sin, tan, cos, exp

set_log_active(True)

theta = -3.0 * pi / 180
L     = 100000.0
H     = 1000.0
a0    = 100
sigma = 10000

nx = 50
ny = 50
nz = 10

model = Model()
model.generate_uniform_mesh(nx, ny, nz, xmin=0, xmax=L, ymin=0, ymax=L,
                            generate_pbcs = True)

Surface = Expression('tan(theta) * x[0]', theta=theta,
                     element=model.Q.ufl_element())
class Bed(Expression):
  def __init__(self, element=None):
    pass
  def eval(self, values, x):
    y_0       = -H + a0 * (exp(-((x[0]-L/2.)**2 + (x[1]-L/2.)**2) / sigma**2))
    values[0] = sin(theta)/cos(theta) * (x[0] + sin(theta)*y_0) \
                + cos(theta)*y_0
Bed = Bed(element=model.Q.ufl_element())

SMB = Expression('0.0', element=model.Q.ufl_element())

model.set_geometry(Surface, Bed, deform=True)

model.set_parameters(IceParameters())
model.calculate_boundaries()
model.initialize_variables()
model.n = 1.0

nonlin_solver_params = default_nonlin_solver_params()
nonlin_solver_params['newton_solver']['relaxation_parameter'] = 1.0
nonlin_solver_params['newton_solver']['relative_tolerance']   = 1.0
nonlin_solver_params['newton_solver']['linear_solver']        = 'mumps'

config = default_config()
config['mode']                         = 'transient'
config['output_path']                  = './results/'
config['t_start']                      = 0.0
config['t_end']                        = 500.0
config['time_step']                    = 2.0
config['periodic_boundary_conditions'] = True
config['velocity']['newton_params']    = nonlin_solver_params
config['velocity']['A0']               = 2.140373e-7
config['velocity']['r']                = 1.0
config['velocity']['approximation']    = 'stokes'
config['free_surface']['on']           = True
config['free_surface']['observed_smb'] = SMB

T = TransientSolver(model, config)
T.solve()



