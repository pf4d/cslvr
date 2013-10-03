#!/bin/bash

srcnodata=1.70141000000000007e+38
dstnodata=-2000000000
epsg='EPSG:3021'
datadir='.'
for dem in "surf" "bed"; do
    gdalwarp -overwrite -t_srs $epsg -of GTiff -srcnodata $srcnodata -dstnodata $dstnodata $datadir/stor_${dem}.grd $datadir/stor_${dem}.tif
done

# Ideally wwe would create a csv file directly, but GEOMETRY=AS_XY does not work
# with the LineString format that gdal_contour produces.
dem='surf'
contour_file=stor_contour
gdal_contour -3d -inodata -fl -1999999999 $datadir/stor_${dem}.tif $datadir/$contour_file.shp
python shp2geo.py $datadir/$contour_file.shp $datadir/$contour_file.geo

# Create contour mesh
gmsh $datadir/$contour_file.geo -1 -2
