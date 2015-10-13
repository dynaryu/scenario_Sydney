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
sf = shapefile.Reader(os.path.join(data_path, 'GCCSA_2011_AUST.shp'))

shapes = sf.shapes()
records = sf.records()

sydney_boundary = Polygon(shapes[2].points)

# read nsw suburb
nsw = shapefile.Reader(os.path.join(data_path, 'nswlocalitypolygon/NSW Locality Polygon.shp'))
nsw_shapes = nsw.shapes()
nsw_fields = nsw.fields[1:]
nsw_fields
nsw_records = nsw.records()

# find suburbs in sydney boundary
tf_sydney = np.zeros((len(nsw_shapes)), dtype=bool)
for i in range(len(nsw_shapes)):
	a0 = Polygon(nsw_shapes[i].points)
	tf_sydney[i] = sydney_boundary.contains(a0) or sydney_boundary.intersects(a0)

# make output shapefile

e = shapefile.Editor(shapefile=os.path.join(data_path, 'nswlocalitypolygon/NSW Locality Polygon.shp'))

for i, okay in enumerate(tf_sydney):
    if not okay:
    	del e._shapes[i]
    	del e.records[i]
e.save(os.path.join(data_path, 'output.shp'))



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
