from cslvr import download_file
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
home     = os.path.dirname(os.path.abspath(filename)) + '/../meshes'

# circle mesh :
fldr = 'test'
url  = 'https://www.dropbox.com/s/cj2cwkz22tzqtqh/circle_mesh.xml.gz?dl=1'
download_file(url, home, fldr, extract=False)




