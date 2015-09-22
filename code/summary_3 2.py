#!/usr/bin/env python

'''
compute economic loss for each building for residential buildings in Sydney

1. read ground motion - DONE
2. read sitedb - DONE
	2.1 Modify bldg type and vintage
		- Suburb vintage
	2.2 Extract bldg mixture signature from Alexandria Canal survey data - DONE
3. read mmi-based vulnerability - DONE 
	3.1 Requires a corresponding set of fragility
4. compute economic loss - DONE
5. read damage dependent HAZUS fatality model 
6. compute fatality
'''

import scipy
import numpy as np
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_sitedb_data(eqrm_input_path, eqrm_output_path, site_tag=None):
    ''' read sitedb file '''

    if site_tag == None:
	    site_tag = [x for x in os.listdir(eqrm_output_path) \
        if 'sites' in x][0].replace('_sites', '')

    data = pd.read_csv(os.path.join(eqrm_input_path, 'sitedb_' + site_tag +\
        '.csv'))
    data['BLDG_COST'] = data['BUILDING_COST_DENSITY'] * data['FLOOR_AREA']\
       * data['SURVEY_FACTOR']
    data['CONTENTS_COST'] = data['CONTENTS_COST_DENSITY'] * data['FLOOR_AREA']\
       * data['SURVEY_FACTOR']
    data['TOTAL_COST'] = data['BLDG_COST'] + data['CONTENTS_COST']

    return(data, site_tag)

def read_gm(eqrm_output_path, site_tag):
    ''' read ground motion, an output of EQRM run'''

    from eqrm_code.RSA2MMI import rsa2mmi_array

    atten_periods = np.load(os.path.join(eqrm_output_path, site_tag + \
        '_motion/atten_periods.npy'))

    selected_periods = [0.0, 0.3, 1.0]

    idx_period = [(np.abs(atten_periods - period)).argmin() \
        for period in selected_periods]

    try:
    	print 'GM at soil is loaded'
        gmotion = np.load(os.path.join(eqrm_output_path, site_tag +\
            '_motion/soil_SA.npy')) # (1,1,1,sites,nsims,nperiods)
    except IOError:
    	print 'GM at bedrock is loaded'
        gmotion = np.load(os.path.join(eqrm_output_path, site_tag +\
            '_motion/bedrock_SA.npy'))

    pga = gmotion[0, 0, 0, :, :, idx_period[0]]
    sa03 = gmotion[0, 0, 0, :, :, idx_period[1]]
    sa10 = gmotion[0, 0, 0, :, :, idx_period[2]]
    mmi = rsa2mmi_array(sa10, period=1.0)

    return(pga, sa03, sa10, mmi)


def compute_vulnerability(mmi, **kwargs):

	def inv_logit(x):
		return  np.exp(x)/(1.0 + np.exp(x)) 

	# assign loss ratio
	coef = {
		"t0":-8.56, 
		"t1": 0.92, 
		"t2": -4.82, 
		"t3": 2.74, 
		"t4": 0.49, 
		"t5": -0.31}

	flag_timber = kwargs['flag_timber']
	flag_pre = kwargs['flag_pre']

	mu = coef["t0"] +\
		coef["t1"]*mmi +\
		coef["t2"]*flag_timber +\
		coef["t3"]*flag_pre +\
		coef["t4"]*flag_timber*mmi +\
		coef["t5"]*flag_pre*mmi

	return inv_logit(mu)

def plot_vulnerabilty(mmi_range, vul):
	''' plot vulnerability'''
	for bldg in vul.keys():
		for pre in vul[bldg].keys():
			label_str = '%s:%s' %(bldg, pre)
			plt.plot(mmi_range, vul[bldg][pre], label=label_str)

	plt.legend()
	plt.grid(1)
	plt.xlabel('MMI')
	plt.ylabel('Loss ratio')


###############################################################################
# main 

working_path = os.path.join(os.path.expanduser("~"),'Projects/scenario_Sydney')
eqrm_input_path = os.path.join(working_path, 'input')
eqrm_output_path = os.path.join(working_path, 'scen_gmMw5.0')

# read inventory
(data, _) = \
    read_sitedb_data(eqrm_input_path, eqrm_output_path, site_tag='sydney')

# read gmotion
(_, _, _, mmi) = read_gm(eqrm_output_path, site_tag='sydney_soil')


# combine mmi
data['MMI'] = pd.Series(mmi[:,0], index=data.index)

# remove nan value (FIXME)
# data.isnull().sum()
data = data[pd.notnull(data['STRUCTURE_CLASSIFICATION'])]

# pivot table
table = pd.pivot_table(data, values="BID", index='STRUCTURE_CLASSIFICATION',\
	columns='PRE1989', aggfunc=np.size)

# all URML will be assigned with URML and TIMBER with TIMBER
data['FLAG_TIMBER'] = data['STRUCTURE_CLASSIFICATION'].str.contains(
	"W1", na=False)
data['FLAG_PRE'] = (data['PRE1989'] == 1)

# MMI based vulnerability
mmi_range = np.arange()

data['LOSS_RATIO'] = compute_vulnerability(data['MMI'], 
	flag_timber=data['FLAG_TIMBER'], 
	flag_pre=data['FLAG_PRE'])

# LOSS_RATIO = 0 if MMI <=4.0 


#data[data['STRUCTURE_CLASSIFICATION'].str.contains("W1", na=False)]

#pd.Series('W1', index=data.index)
#data.ix[data.STRUCTURE_CLASSIFICATION=='URMLMETAL' & data.PRE1989==1,\
'BLDG_TYPE'] ='URML_PRE'
#data.ix[data.STRUCTURE_CLASSIFICATION=='URMLTILE', 'BLDG_TYPE'] ='URML'
#data.ix[data.STRUCTURE_CLASSIFICATION=='URMMMETAL', 'BLDG_TYPE'] ='URML'
#data.ix[data.STRUCTURE_CLASSIFICATION=='URMMTILE', 'BLDG_TYPE'] ='URML'

#