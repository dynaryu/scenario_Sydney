# read Alexandria dbf
import pandas as pd
import numpy as np

###############################################################################
# original xls file
alex = pd.read_excel('../data/Alexandria.xls')

# captialize column name 
alex.columns = [x.upper() for x in alex.columns]
# In [17]: alex.columns
# Out[17]: 
# Index([  u'ASSESSOR',    u'ADDRESS',     u'SUBURB',      u'STATE',
#          u'POSTCODE',       u'AREA',   u'COMMENTS',  u'DATA_USED',
#           u'BLD_AGE',   u'LIV_UNIT',  u'USE_MAJOR',  u'USE_MINOR',
#           u'STOREYS',  u'FOUND_TYP',  u'BASEMENTS',  u'FLOOR_HGT',
#          u'ROOF_MAT',    u'EAVES_W',     u'AIRCON',   u'AIRCON_P',
#           u'VERANDA',  u'GARAGE_NO',   u'GARAGE_D',    u'BLD_MAT',
#         u'BLD_STRUC',     u'RICS_0',     u'RICS_1',     u'RICS_2',
#               u'UFI',        u'LID',  u'ADDRESS_1', u'NEXIS_CAD_',
#        u'STRUCTURE_', u'CONTENTS_V',   u'DISTANCE',      u'DEPTH',
#            u'HEIGHT',      u'FLOOD',     u'UNI_FL',        u'VRM',
#          u'L_NSW_FI',  u'L_NSW_UNI',    u'FLOOR_H',     u'JOIN_1'],
#       dtype='object')

# clean up and save
for key_ in list(alex.columns):
	try:
		alex.set_value(alex.index, key_, alex[key_].str.strip())
	except AttributeError:
		print key_

alex.to_csv('./Alexandria.csv', index=False)

###############################################################################
# read clean data
alex = pd.read_csv('../data/Alexandria.csv')

# only choose residential
#idx = alex['USE_MAJOR'].str.contains("Residential", na=False)
use_include = alex['USE_MAJOR']=='Residential'

age_remove = (alex['BLD_AGE'] == u'Unassessable') | (
	alex['BLD_AGE'] == u'Unknown') | (
	alex['BLD_AGE'] == u'Not assessed')

bldg_remove = (alex['BLD_STRUC'] == u'Unassessable') | (
	alex['BLD_STRUC'] == u'Unknown') | (
	alex['BLD_STRUC'] == u'Other') | (
	alex['BLD_STRUC'] == u'Not assessed')

ndata = alex.ix[~age_remove & ~bldg_remove & use_include].copy()

# only choose BLD_STRUC, BLD_AGE, suburb 
#ndata = alex.ix[idx, ['BLD_STRUC', 'BLD_AGE', 'suburb']]

#ndata['BLD_AGE1'] = ndata['BLD_AGE'].str.strip()
#ndata['BLD_STRUC1'] = ndata['BLD_STRUC'].str.strip()

#age_remove = (ndata['BLD_AGE1'] == u'Unassessable') | (
#	ndata['BLD_AGE1'] == u'Unknown') | (
#	ndata['BLD_AGE1'] == u'Not assessed')

# age_post = (ndata['BLD_AGE1'] == '1990 to PRESENT') | (
# 	ndata['BLD_AGE1'] == '1980 to 1989') | (
# 	ndata['BLD_AGE1'] == '1960 to 1979') | (
# 	ndata['BLD_AGE1'] == u'1946 to 1959')

def convert_to_float(string):
	val = string.split(' ')[-1]
	try:
		float(val)
		return (True, float(val))
	except ValueError:
		return (False, 2015)

ndata['PRE1945'] = ndata['BLD_AGE'].apply(
	lambda x: True if convert_to_float(x)[1] <= 1945 else False)

table = pd.pivot_table(ndata, values="ASSESSOR", index='BLD_STRUC',\
	columns='PRE1945', aggfunc=len)

# PRE1945                 False  True 
# BLD_STRUC                           
# Masonry veneer            205      4
# Precast concrete frame      3    NaN
# RC frame                   61    NaN
# Reinforced masonry         12     22
# Unrienforced masonry      295   2074

pd.pivot_table(ndata, values="ASSESSOR", index='SUBURB',\
	columns='PRE1945', aggfunc=len)

# PRE1945       False  True 
# SUBURB                    
# ALEXANDRIA      113    356
# BEACONSFIELD     47     38
# ERSKINEVILLE     84    480
# NEWTOWN          13    187
# REDFERN          40    532
# ROSEBERY        141    NaN
# ST PETERS         2      1
# SURRY HILLS      14    467
# WATERLOO        101     39
# ZETLAND          21    NaN

# age_post = (alex['BLD_AGE'] == '1990 to PRESENT') | (
# 	ndata['BLD_AGE1'] == '1980 to 1989') | (
# 	ndata['BLD_AGE1'] == '1960 to 1979') | (
# 	ndata['BLD_AGE1'] == u'1946 to 1959')

# age_pre = ( ndata['BLD_AGE1'] == '1914 to 1945') | (
# 	ndata['BLD_AGE1'] == 'BEFORE 1891') | (
# 	ndata['BLD_AGE1'] == '1891 to 1913')

#assert age_remove.sum() + age_post.sum() + age_pre.sum() == len(ndata)

#ndata['PRE1945'] = pd.Series(np.zeros_like(ndata['BLD_AGE1']), index=ndata.index)
#ndata['PRE1945'].ix[age_pre] = 1

#ndata1 = ndata.ix[~age_remove & ~bldg_remove].copy()

# Final value suggested by Mark
#        Pre-1945, Post-1945
# Timber 0.002, 0.080
# URM    0.804, 0.114