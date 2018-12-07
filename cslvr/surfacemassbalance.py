from cslvr.physics import Physics


class SurfaceMassBalance(Physics):

	def __init__(self, model, config):
		"""
		This class computes the surface accumulation/ablation function
		``model.S_ring`` from balance velocity, lower surface-mass balance
		``model.Mb``, and ice-sheet thickness ``model.H``.
		"""
		s    = "::: INITIALIZING SURFACE-MASS-BALANCE PHYSICS :::"
		print_text(s, cls=self)


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

		phi    = TestFunction(Q)
		S_ring = TrialFunction(Q)

		Ubar   = as_vector([ubar, vbar, wbar])

		self.B = (div(Ubar*H) + Mb) * phi * dx
		self.a = S_ring * phi * dx

	def solve(self):
		"""
		Solve for the surface mass balance.
		"""
		model = self.model

		s    = "::: solving for surface mass balance :::"
		print_text(s, cls=self)
		solve(self.a == self.B, model.S_ring)
		print_min_max(model.S_ring, 'S_ring')
