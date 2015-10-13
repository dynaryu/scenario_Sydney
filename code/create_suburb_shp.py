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
records = sf.records()

sydney_boundary = Polygon(shapes[2].points)

# read nsw suburb
nsw = shapefile.Reader(os.path.join(data_path, 'nswlocalitypolygon/NSW Locality Polygon.shp'))
nsw_shapes = nsw.shapes()
nsw_fields = nsw.fields[1:]

# find suburbs in sydney boundary
tf_sydney = np.zeros((len(nsw_shapes)), dtype=bool)
for i in range(len(nsw_shapes)):
	a0 = Polygon(nsw_shapes[i].points)
	tf_sydney[i] = sydney_boundary.contains(a0) or sydney_boundary.intersects(a0)

del sf
del nsw

# make output shapefile
e = shapefile.Editor(shapefile=os.path.join(data_path, 'nswlocalitypolygon/NSW Locality Polygon.shp'))

for i, okay in enumerate(tf_sydney):
    if not okay:
    	del e._shapes[i]
    	del e.records[i]
e.save(os.path.join(data_path, 'output.shp'))

w = shapefile.Writer(shapefile.POLYGON)

# field
for item in nsw_fields:
	w.field(item[0], item[1], item[2], item[3])

idx = np.where(tf_sydney)[0]
for i in idx:
	w.poly(parts=[nsw_shapes[i].points])
	tmp = nsw_records[i]
	w.record(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], 
		tmp[7], tmp[8], tmp[9], tmp[10], tmp[11])

w.save('polygon.shp')


w.record('14800',
 [2011, 5, 17],
 [0, 0, 0],
 'NSW825',
 [2012, 2, 4],
 [0, 0, 0],
 'CARDIFF HEIGHTS',
 '    ',
 '    ',
 'G',
 [0, 0, 0],
 '1')


for j,k in enumerate(x):
    w.point(k,y[j]) #write the geometry
    w.record(k,y[j],date[j], target[j], id_no[j]) #

w.poly(parts=[[[1,5],[5,5],[5,1],[3,3],[1,1]]])
w.field('FIRST_FLD','C','40')
w.field('SECOND_FLD','C','40')
w.record('First','Polygon')
w.save('polygon.shp')


w = shapefile.Writer(shapefile.POLYGON)
w.poly(shapeType=3, parts=[[[122,37,4,9], [117,36,3,4]], [[115,32,8,8],
... [118,20,6,4], [113,24]]])

w.field('FIRST_FLD','C','40')
w.field('SECOND_FLD','C','40')
w.record('First','Polygon')
w.save('shapefiles/test/polygon')



w = shapefile.Writer(shapefile.POLYGON)

for i in range(len(nsw_fields)):
	tmp_field = []
	for k in range(len(nsw_fields[i])):
		tmp_field.append(str(nsw_fields[i][k]))
	w.field(tmp_field)



for item in nsw_fields:
    w.field(item)

# only selected one
for j, okay in enumerate(tf_sydney):
	if okay:
		w.poly(parts=[nsw_shapes[j].points])
		w.record(nsw_records[j])

nsw_point = nsw.apply(lambda row: Point([row['LONGITUDE'], row['LATITUDE']]), axis=1)
tf_sydney = nsw_point.apply(sydney_boundary.contains)

# plot 
llcrnrlon = 

m = Basemap(llcrnrlon=llcrnlon,llcrnrlat=llcrnlat,urcrnrlon=urcrnlon,urcrnrlat=urcrnlat,resolution='f',projection='merc',lon_0=(urcrnlon+llcrnlon)/2,lat_0=(urcrnlat+llcrnlat)/2)

# plot contour map
fig = plt.figure()
ax = plt.gca()

m.drawcoastlines()
#m.drawmapboundary()
m.drawstates(linewidth=3)
#m.drawlsmask()
#m.fillcontinents(color='lightgrey',lake_color='white')
m.drawcountries(linewidth=3)
m.drawparallels(np.arange(-34,-33.2,0.25),labels=[1,0,0,0])
m.drawmeridians(np.arange(150.0,151.4,0.25),labels=[0,0,0,1])
