from cslvr   import D2Model, MomentumHybrid, plot_variable
from fenics  import Point, RectangleMesh, Expression, sqrt, pi
from numpy   import array

# output directiories :
out_dir = './ISMIP_HOM_A_results/hybrid/'
plt_dir = '../../images/momentum/ISMIP_HOM_A/hybrid/'

alpha = 0.5 * pi / 180 
L     = 8000

p1    = Point(0.0, 0.0)
p2    = Point(L,   L)
mesh  = RectangleMesh(p1, p2, 25, 25)

model = D2Model(mesh, out_dir = out_dir, use_periodic = True, kind = 'hybrid')

surface = Expression('- x[0] * tan(alpha)', alpha=alpha,
                     element=model.Q.ufl_element())
bed     = Expression(  '- x[0] * tan(alpha) - 1000.0 + 500.0 * ' \
                     + ' sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)',
                     alpha=alpha, L=L, element=model.Q.ufl_element())

model.init_S(surface)
model.init_B(bed)
model.init_beta(1000)
model.init_A(1e-16)

mom = MomentumHybrid(model)
mom.solve()

# this function allow the plotting of an arbitrary FEniCS function or 
# vector that reside on a two-dimensional mesh :
plot_variable(u = model.U3_s, name = 'U_mag', direc = plt_dir,
              ext                 = '.pdf',
              title               = r'$\underline{u} |_S$',
              levels              = None,#U_lvls,
              cmap                = 'viridis',
              tp                  = True,
              show                = False,
              extend              = 'neither',
              cb_format           = '%g')



