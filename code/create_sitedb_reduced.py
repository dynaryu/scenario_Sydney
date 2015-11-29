# create building exposure data for EQRM simulation

import pandas as pd
import os
import shapefile
#from shapely.geometry import Polygon
#from shapely.geometry import Point

# def checking_inside(idx):

#   tf_sydney = np.zeros(len(idx), dtype=bool)
#   for i, ix in enumerate(idx):
#       point_ = Point(nsw.ix[ix]['LONGITUDE'], nsw.ix[ix]['LATITUDE'])
#       tf_sydney[i] = sydney_boundary.contains(point_)

#   return np.any(tf_sydney)

working_path = os.path.join(os.path.expanduser("~"),
                            'Projects/scenario_Sydney')
data_path = os.path.join(working_path, 'data')

# sydney shapefile
sydney = shapefile.Reader(os.path.join(data_path, 'GIS',
                          'sydney_sa1/Export_Output.shp'))
records = sydney.records()

chosen_SA4_NAME11 = ['Sydney - City and Inner South',
                     'Sydney - Inner South West',
                     'Sydney - Eastern Suburbs',
                     #'Sydney - North Sydney and Hornsby',
                     'Sydney - Inner West',
                     #'Sydney - Ryde',
                     #'Sydney - Northern Beaches',
                     #'Sydney - Parramatta',
                     ]

# extract 'SA1_MAIN11' whithin chosen SA4
sydney_sa1 = [x[0] for x in records if x[8] in chosen_SA4_NAME11]

# Read sydney building data
sitedb = pd.read_csv(os.path.join(working_path, 'input',
                     'sitedb_sydney_soil.csv'), dtype={'SA1_CODE': str})

tf = sitedb['SA1_CODE'].isin(sydney_sa1)
# tf.sum() = 410592

sydney_reduced = sitedb[tf].copy()
del sitedb

file_ = os.path.join(working_path, 'input/sitedb_sydney_soil_reduced.csv')
sydney_reduced.to_csv(file_, index=False)
print("%s is created successfully." % file_)


##############################################################################
# Read sydney building data
sitedb = pd.read_csv(os.path.join(working_path, 'input',
                     'sitedb_sydney_soil.org.csv'), dtype={'SA1_CODE': str})

sitedb = pd.read_csv(os.path.join(working_path, 'input',
                     'sitedb_sydney_soil.csv'), dtype={'SA1_CODE': str})

for item, groups in sitedb.groupby('STRUCTURE_CLASSIFICATION'):
    file_ = os.path.join(working_path, 'input/sitedb_sydney_soil_{}.csv'.format(item))
    groups.to_csv(file_, index=False)
    print("%s is created successfully." % file_)

