import pandas as pd
import os
import shapefile
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
from mpl_toolkits.basemap import Basemap

working_path = os.path.join(os.path.expanduser("~"),'Projects/scenario_Sydney')
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


# find suburbs in sydney boundary
tf_sydney = np.zeros((len(nsw_shapes)), dtype=bool)
for i in range(len(nsw_shapes)):
	a0 = Polygon(nsw_shapes[i].points)
	tf_sydney[i] = sydney_boundary.contains(a0) or sydney_boundary.intersects(a0)

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
