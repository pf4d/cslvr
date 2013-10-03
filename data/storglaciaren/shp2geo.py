#!/usr/bin/env python
# Copyright (C) 2013 Andy Aschwanden
#

from argparse import ArgumentParser
import numpy as np


def read_shapefile(filename):
    '''
    Reads lat / lon from a ESRI shape file.

    Paramters
    ----------
    filename: filename of ESRI shape file.

    Returns
    -------
    lat, lon: array_like coordinates
    
    '''
    import ogr
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(filename, 0)
    layer = data_source.GetLayer(0)
    srs=layer.GetSpatialRef()
    cnt = layer.GetFeatureCount()
    x = []
    y = []
    c = []
    ## for pt in range(0, cnt):
    ##     feature = layer.GetFeature(pt)
    ##     geometry = feature.GetGeometryRef()
    ##     points = geometry.GetPoints()
    ##     for point in points:
    ##         x.append(point[0])
    ##         y.append(point[1])
    ##         c.append(pt)
    pt = 177
    feature = layer.GetFeature(pt)
    geometry = feature.GetGeometryRef()
    points = geometry.GetPoints()
    for point in points:
        x.append(point[0])
        y.append(point[1])
        c.append(pt)
    return np.asarray(y), np.asarray(x), np.asarray(c)


# Set up the option parser
description = '''A script to convert a shapefile into gmsh geo format.'''
parser = ArgumentParser()
parser.description = description
parser.add_argument("FILE", nargs=2)

options = parser.parse_args()
args = options.FILE

filename = args[0]
outfilename = args[1]

y, x, c = read_shapefile(filename)
data = np.vstack((y, x))

f = open(outfilename, 'w')
print >> f, 'Mesh.Algorithm=5; \n'
for k in range(0,len(y)):
    out = 'Point(%g)={%14.7e,%14.7e,0.0,%g};' % (k+1 ,y[k], x[k], 18.0)
    print >> f, out


s = str(range(1, len(y)+1)).split('[')[1].split(']')[0]
out = 'Spline(1)={%s,1};' % s
print >> f, out
print >> f, 'Line Loop(2)={1};'
print >> f, 'Plane Surface(3) = {2};'
print >> f, 'Physical Line(4) = {1};'
print >> f, 'Physical Surface(5) = {3};'
f.close()

import pylab as plt
plt.plot(x, y)
