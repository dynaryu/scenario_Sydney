# create sitedb 
import pandas as pd

#dat = pd.read_csv('./input/NSW_Residential_Earthquake_Exposure_201407_for_EQRM.csv')
dat = pd.read_csv('/media/hyeuk/NTFS/NSW_EQRMrevised_NEXIS2015.csv')

# suburbs in the Alexandria canal survey
# ERSKINEVILLE    591
# REDFERN         586
# ALEXANDRIA      515
# SURRY HILLS     483
# NEWTOWN         211
# WATERLOO        151
# ROSEBERY        144
# BEACONSFIELD     89
# ZETLAND          21
# ST PETERS         4

dat.ix[dat['SUBURB']=='ERSKINEVILLE']['YEAR_BUILT'].value_counts()
# Out[25]: 
# 1987 - 1991    1056
# 1962 - 1971     139
# 1992 - 1996      62
# 1997 - 2001      57
# 1972 - 1976      55
# 1982 - 1986      53
# 1977 - 1981      43
# 2002 - 2006      36
# 2007 - 2011      30
# 1952 - 1961       9

# from the survey data
# ndata1.ix[ndata1['suburb']=='ERSKINEVILLE']['PRE1945'].value_counts()
# Out[29]: 
# 1    480
# 0     84



# classify vintage
data['YEAR_BUILT'].value_counts()

# 1992 - 1996    529073
# 1987 - 1991    488241
# 1982 - 1986    210332
# 1962 - 1971    202718
# 1997 - 2001    140360
# 1977 - 1981    137135
# 1972 - 1976    117967
# 2002 - 2006     96181
# 2007 - 2011     46330
# 1952 - 1961     10474
# 1947 - 1951      3738

# Unknown           264
# 1891 - 1913        36
# 1914 - 1946        75

zip_code = pd.read_csv('./input/sydney_postcode.csv')
selected = zip_code['Sydney postcodes'].unique()

ndat = dat.loc[dat['POSTCODE'].isin(selected)]
ndat = ndat.drop('SA1_CODE',1)
ndat.to_csv('./.csv', index=False)