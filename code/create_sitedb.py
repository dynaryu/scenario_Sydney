# create building exposure data for EQRM simulation

import pandas as pd
import os
import shapefile
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np

def checking_inside(idx):

	tf_sydney = np.zeros(len(idx), dtype=bool)
	for i, ix in enumerate(idx):
	    point_ = Point(nsw.ix[ix]['LONGITUDE'], nsw.ix[ix]['LATITUDE'])
	    tf_sydney[i] = sydney_boundary.contains(point_)

	return np.any(tf_sydney)    

def convert_to_float(string):
	val = string.split('-')[-1]
	try:
		float(val)
		return float(val)
	except ValueError:
		print "Not a Number"

working_path = os.path.join(os.path.expanduser("~"),'Projects/scenario_Sydney')
data_path = os.path.join(working_path, 'data')

# Read NSW building data
try:   
    nsw = pd.read_csv('/Volumes/NTFS/NSW_EQRMrevised_NEXIS2015.csv')
except IOError:
    nsw = pd.read_csv('/media/hyeuk/NTFS/NSW_EQRMrevised_NEXIS2015.csv')
except IOError:
    print "No USB"

# Read the admin boundary of Greater Sydney
sf = shapefile.Reader(os.path.join(working_path, 'data/GCCSA_2011_AUST.shp'))
shapes = sf.shapes()
records = sf.records()
#records[2] corresponds to the Greater Sydney

sydney_boundary = Polygon(shapes[2].points)

#table = pd.pivot_table(nsw, values="BID", index='SUBURB',\
#	columns='POSTCODE', aggfunc=len)

# randomly sample 20 buildings of each postcode
list_postcode = nsw['POSTCODE'].unique()
tf_sydney_postcode = np.zeros(len(list_postcode), dtype=bool)
for i, postcode in enumerate(list_postcode):
	idx = np.where(nsw['POSTCODE']==postcode)[0]
	no_blds = len(idx)
	if no_blds > 20:
		idx = np.random.choice(idx, 20, replace=False)
	tf_sydney_postcode[i] = checking_inside(idx)
	print "%s out of 607: %s, %s" %(i, postcode, tf_sydney_postcode[i])

sydney_postcode = list_postcode[tf_sydney_postcode]
sydney = nsw.loc[nsw['POSTCODE'].isin(sydney_postcode)].copy()

# assign new class given GA_class and year_built
# sydney['YEAR_BUILT'].value_counts()
idx_pre1945 = sydney['YEAR_BUILT'].apply(
	lambda x: True if convert_to_float(x) <= 1946 else False)
Timber_list = ['W1TIMBERTILE', 'W1TIMBERMETAL', 'W1BVTILE', 'W1BVMETAL']
idx_timber = sydney['GA_STRUCTURE_CLASSIFICATION'].isin(Timber_list)

idx_timber_pre1945 = idx_timber & idx_pre1945 
idx_timber_post1945 = idx_timber & (~idx_pre1945) 
idx_URM_pre1945 = (~idx_timber) & (idx_pre1945) 
idx_URM_post1945 = (~idx_timber) & (~idx_pre1945) 

# assign value
sydney.loc[idx_timber_pre1945, 'BLDG_CLASS'] = 'Timber_Pre1945'
sydney.loc[idx_timber_post1945, 'BLDG_CLASS'] = 'Timber_Post1945'
sydney.loc[idx_URM_pre1945, 'BLDG_CLASS'] = 'URM_Pre1945'
sydney.loc[idx_URM_post1945, 'BLDG_CLASS'] = 'URM_Post1945'

# Read the list of old suburbs
suburb_vintage = pd.read_excel(os.path.join(
       data_path, 'List_of_Sydney_postcodes.xlsx'), sheetname="Postcodes-Suburbs", skiprows=1)
idx_old_suburbs = pd.notnull(suburb_vintage['Alexandra Canal?'])
old_suburbs = suburb_vintage.ix[idx_old_suburbs]['Suburbs'].str.upper()
old_suburbs = old_suburbs.tolist()

# minor changes 
[old_suburbs.append(x) for x in ['ROSEBERY', 'BIRCHGROVE', 'ENFIELD']]

# Make sure that all of the old suburbs are included in the database. Otherwise there musht be some error.
idx_old_suburbs = sydney['SUBURB'].isin(old_suburbs)
no_bldgs_in_old_suburbs = sum(idx_old_suburbs)

# apply building mixture signature extracted from Alexandria canal survey to the old suburbs

# Final value suggested by Mark based on the Alexandra canal survey
#        Pre-1945, Post-1945
# Timber 0.002, 0.080
# URM    0.804, 0.114                     
bldg_mixture = {'Timber_Pre1945': 0.002, 'Timber_Post1945': 0.080,
                'URM_Pre1945': 0.804, 'URM_Post1945': 0.114}
value = np.random.choice(bldg_mixture.keys(), size=no_bldgs_in_old_suburbs, 
	p=bldg_mixture.values())

# assign value
sydney.loc[idx_old_suburbs, 'BLDG_CLASS'] = value

# before
# In [37]: sydney['BLDG_CLASS'].value_counts()
# Out[37]: 
# Timber_Post1945    921444
# URM_Post1945       144298
# Timber_Pre1945         37
# URM_Pre1945            30
# dtype: int64

# after
# sydney['BLDG_CLASS'].value_counts()
# Out[57]: 
# Timber_Post1945    841110
# URM_Post1945       142872
# URM_Pre1945         81578
# Timber_Pre1945        249

sydney.to_csv('../data/sydney_EQRMrevised_NEXIS2015.csv', index=False)

# one for ground motion
sydney_soil = sydney[['LATITUDE', 'LONGITUDE', 'SITE_CLASS']]
sydney_soil.to_csv('../input/sydney_soil_par_site.csv', index=False)