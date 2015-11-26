import pandas as pd
import os
import shapefile
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
from mpl_toolkits.basemap import Basemap

# Note that it works but not so efficient as arcGIS.
working_path = os.path.join(os.path.expanduser("~"), 'Projects/scenario_Sydney')
data_path = os.path.join(working_path, 'data')

# Read the admin boundary of Greater Sydney
sf = shapefile.Reader(os.path.join(data_path, 'aust_polygon/GCCSA_2011_AUST.shp'))

shapes = sf.shapes()
#records = sf.records()

sydney_boundary = Polygon(shapes[2].points)

# read SA1
nsw = shapefile.Reader(os.path.join(data_path, 'sa1_polygon/SA1_2011_AUST.shp'))
nsw_shapes = nsw.shapes()
nsw_fields = nsw.fields
nsw_records = nsw.records()

# find suburbs in sydney boundary
tf_sydney = np.zeros((len(nsw_shapes)), dtype=bool)
for i in range(len(nsw_shapes)):
    a0 = Polygon(nsw_shapes[i].points)
    tf_sydney[i] = sydney_boundary.intersects(a0)

# make output shapefile
w = shapefile.Writer(shapefile.POLYGON)

# field
for item in nsw_fields:
    w.field(item[0], item[1], item[2], item[3])

idx = np.where(tf_sydney)[0]
for i in idx:
    w.poly(shapeType=3, parts=[nsw_shapes[i].points])
    tmp = nsw_records[i]
    w.record(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6],
        tmp[7], tmp[8], tmp[9], tmp[10], tmp[11])

w.save(os.path.join(data_path,'sydney_suburbs_polygon.shp'))

# example
# w = shapefile.Writer(shapefile.POLYGON)
# w.poly(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
# w.field('FIRST_FLD','C','40')
# w.field('SECOND_FLD','C','40')
# w.record('First','Polygon')
# w.save('polygon.shp')

# create point shapefile for epicentre
w = shapefile.Writer(shapefile.POINT)
w.point(151.153, -33.914, 0, 0)
w.field('FIRST_FLD')
w.field('SECOND_FLD', 'C', '40')
w.record('First', 'Point')
w.save(os.path.join(data_path, 'sydney_epicentre.shp'))

# create polygon shapefile for zone source
# create point shapefile for epicentre
w = shapefile.Writer(shapefile.POLYGON)
w.poly(parts=[[[151.000000, -32.000000],
               [152.000000, -32.500000],
               [152.200000, -32.900000],
               [150.900000, -35.300000],
               [150.600000, -35.300000],
               [150.000000, -34.700000],
               [149.500000, -33.500000],
               [148.300000, -32.800000],
               [148.000000, -32.000000],
               [149.500000, -31.500000],
               [151.000000, -32.000000]]])

w.poly(parts=[[[153.097000, -30.866000],
               [153.500000, -31.000000],
               [152.368000, -33.943000],
               [151.000000, -37.500000],
               [150.201000, -36.861000],
               [149.750000, -36.500000],
               [150.600000, -35.300000],
               [150.900000, -35.300000],
               [152.200000, -32.900000],
               [152.000000, -32.500000],
               [152.750000, -30.750000],
               [153.097000, -30.866000]]])

w.field('FIRST_FLD', 'C', '40')
w.field('SECOND_FLD', 'C', '40')
w.record('First', 'Polygon')
w.record('Second', 'Polygon')

w.save(os.path.join(data_path, 'sydney_zone_source.shp'))
# I need to define the projection in the ArcGIS using
# Arc Toolbox - Data Management - Projections.


