from dolfin import Mesh
import inspect
import os
import sys

class MeshFactory(object):

	global home
	filename = inspect.getframeinfo(inspect.currentframe()).filename
	home     = os.path.dirname(os.path.abspath(filename)) + '/../meshes'

	@staticmethod
	def get_circle():
		global home
		return Mesh(home + '/test/circle_mesh.xml.gz')



