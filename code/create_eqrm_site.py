import pandas as pd
import os
import shapefile
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np

working_path = os.path.join(os.path.expanduser("~"),'Projects/scenario_Sydney')
data_path = os.path.join(working_path, 'data')

# Read the admin boundary of Greater Sydney
sf = shapefile.Reader(os.path.join(data_path, 'aust_polygon/GCCSA_2011_AUST.shp'))

shapes = sf.shapes()
#records = sf.records()

#sydney_boundary = Polygon(shapes[2].points)

[ll_lon, ll_lat, rl_lon, lu_lat] = shapes[2].bbox

#LATITUDE,LONGITUDE,SITE_CLASS,VS30
# ll_lat, ll_lon = -34.16, 150.58
# lu_lat, lu_lon = -33.46, 150.58
# ru_lat, ru_lon = -33.46, 152.00
# rl_lat, rl_lon = -34.16, 152.00

# create array with 0.005 degree
dlat = 0.005
dlon = 0.005
lat_array = np.arange(ll_lat, lu_lat+dlat, dlat)
lon_array = np.arange(ll_lon, rl_lon+dlon, dlon)

xv, yv = np.meshgrid(lat_array, lon_array)

xv = xv.flatten()
yv = yv.flatten()

"""
# it does not work as expected
tf_sydney = np.zeros(len(xv), dtype=bool)
for i in range(len(xv)):
    point_ = Point(yv[i], xv[i])
    tf_sydney[i] = sydney_boundary.contains(point_)
"""

#SITE_CLASS  VS30
dd = {'LATITUDE': xv, 'LONGITUDE': yv}
new = pd.DataFrame(dd)
new['SITE_CLASS'] = ['C']*len(xv)
#new['VS30'] = np.ones_like(xv)*560.0

new.to_csv('./input/sydney_par_site.csv', index=False)

# and then remove points outside of the boundary using ArcGIS