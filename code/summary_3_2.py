#!/usr/bin/env python

'''
compute economic loss for each building for residential buildings in Sydney

1. read ground motion - DONE
2. read sitedb - DONE
    2.1 Modify bldg type and vintage - DONE
        - Suburb vintage (provided by Martin)
    2.2 Extract bldg mixture signature from Alexandria Canal survey data - DONE
3. read mmi-based vulnerability - DONE
    3.1 Assess damage state by loss ratio
4. compute economic loss 
5. read damage dependent HAZUS casualty model - DONE
6. compute casualty
'''

import scipy
import numpy as np
import sys
import os
import copy
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_hazus_casualty_data(hazus_data_path, selected_bldg_class=None):

    # read indoor casualty (table13.3 through 13.7)
    casualty_rate={}
    colname = ['Bldg type', 'Severity1', 'Severity2', 'Severity3',\
               'Severity4']
    list_ds = ['slight', 'moderate', 'extensive', 'complete', 'collapse']
    for ds in list_ds:
        fname = os.path.join(hazus_data_path, 'hazus_indoor_casualty_' + ds + '.csv')
        tmp = pd.read_csv(fname, header=0, 
            names=colname, usecols=[1, 2, 3, 4, 5], index_col=0)
        if selected_bldg_class is not None:
            okay = tmp.index.isin(selected_bldg_class)
            casualty_rate[ds] = tmp.ix[okay].to_dict()
        else:
            casualty_rate[ds] = tmp.to_dict()

    # no damage
    casualty_rate['no'] = copy.deepcopy(casualty_rate['slight'])
    for str_ in ['Severity1', 'Severity2', 'Severity3', 'Severity4']:
        casualty_rate['no'][str_] = {key: 0 for key, val in \
        casualty_rate['no']['Severity4'].items()}

    return casualty_rate

def read_hazus_collapse_rate(hazus_data_path):

    # read collapse rate (table 13.8)
    fname = os.path.join(hazus_data_path, 'hazus_collapse_rate.csv')
    collapse_rate = pd.read_csv(fname, skiprows=1, names=['Bldg type','rate'], 
                index_col=0, usecols=[1, 2])
    return collapse_rate.to_dict()['rate']

def read_sitedb_data(sitedb_file):
    ''' read sitedb file '''

    data = pd.read_csv(sitedb_file)
    data['BLDG_COST'] = data['BUILDING_COST_DENSITY'] * data['FLOOR_AREA']\
       * data['SURVEY_FACTOR']
    data['CONTENTS_COST'] = data['CONTENTS_COST_DENSITY'] * data['FLOOR_AREA']\
       * data['SURVEY_FACTOR']
    data['TOTAL_COST'] = data['BLDG_COST'] + data['CONTENTS_COST']

    return data

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


def compute_vulnerability(mmi, bldg_class):

    def inv_logit(x):
        return  np.exp(x)/(1.0 + np.exp(x)) 

    def compute_mu(mmi, **kwargs):

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
        return mu

    flag_timber = 'Timber' in bldg_class
    flag_pre = 'Pre' in bldg_class

    # correction of vulnerability suggested by Mark
    if mmi < 5.5:
        prob55 = inv_logit(compute_mu(5.5, 
            flag_timber=flag_timber, 
            flag_pre=flag_pre))
        return(np.interp(mmi, [4.0, 5.5], [0.0, prob55], left=0.0))
    else:
        mu = compute_mu(mmi, 
            flag_timber=flag_timber, 
            flag_pre=flag_pre)
        return(inv_logit(mu))

def plot_vulnerabilty(mmi_range, vul):
    ''' plot vulnerability'''

    mmi_range = np.arange(4.0, 10.0, 0.05)

    vul = {}
    for bldg in ['Timber_Pre1945', 'Timber_Post1945', 'URM_Pre1945', 'URM_Post1945']:
        tmp = []
        for val in mmi_range:
            tmp.append(compute_vulnerability(val, bldg))
        vul[bldg] = np.array(tmp)

    for bldg in vul.keys():
        label_str = bldg.replace('_',':')
        plt.plot(mmi_range, vul[bldg], label=label_str)

    plt.legend()
    plt.grid(1)
    plt.xlabel('MMI')
    plt.ylabel('Loss ratio')

def compute_casualty(casualty_rate, damage_state, bldg_class, population):

    # casualty = np.zeros((4))
    # for i in range(4):
    #     severity_str = 'Severity' + str(i+1)
    #     rate_ = casualty_rate[damage_state][severity_str][bldg_class]*0.01
    #     casualty[i] = np.round(population*rate_)
    casualty = [ population*casualty_rate[damage_state][
        'Severity'+str(i)][bldg_class]*0.01 for i in range(1, 5) ]
    return pd.Series({'Severity1': casualty[0], 'Severity2': casualty[1], 
        'Severity3': casualty[2], 'Severity4': casualty[3]})

###############################################################################
# main 

working_path = os.path.join(os.path.expanduser("~"),'Projects/scenario_Sydney')
eqrm_input_path = os.path.join(working_path, 'input')
eqrm_output_path = os.path.join(working_path, 'scen_gmMw5.0')
data_path = os.path.join(working_path, 'data')

# read inventory
data = \
    read_sitedb_data(os.path.join(data_path,
        'sydney_EQRMrevised_NEXIS2015.csv'))

# read gmotion
(_, _, _, mmi) = read_gm(eqrm_output_path, site_tag='sydney_soil')

# read hazus indoor casualty data
casualty_rate = read_hazus_casualty_data(data_path,\
    selected_bldg_class = data['BLDG_CLASS'].unique())

# read hazus collapse rate data
#collapse_rate = read_hazus_collapse_rate(data_path)

# combine mmi
data['MMI'] = pd.Series(mmi[:,0], index=data.index)

# remove nan value (FIXME)
# data.isnull().sum()
#data = data[pd.notnull(data['STRUCTURE_CLASSIFICATION'])]

# pivot table
#table = pd.pivot_table(data, values="BID", index='STRUCTURE_CLASSIFICATION',\
#   columns='PRE1989', aggfunc=np.size)
#data['BLDG_CLASS'].value_counts()
# Out[43]: 
# Timber_Post1945    766179
# URM_Post1945       132624
# URM_Pre1945         72898
# Timber_Pre1945        212

#data['LOSS_RATIO'] = compute_vulnerability(data['MMI'], 
#   flag_timber=data['FLAG_TIMBER'], 
#   flag_pre=data['FLAG_PRE'])

# for name, group in data.groupby('BLDG_CLASS'):

#     tmp = group.apply(lambda row: compute_vulnerability(row['MMI'], name), axis=1)
#     data['new'] = pd.Series(data=tmp, index=tmp.index)

okay = data['MMI'] > 4.0
data.loc[~okay, 'LOSS_RATIO'] = 0.0
data.loc[okay, 'LOSS_RATIO'] = data.ix[okay].apply(lambda row: compute_vulnerability(
    row['MMI'], row['BLDG_CLASS']), axis=1)

data['LOSS'] = data['LOSS_RATIO'] * data['TOTAL_COST']

# mean loss ratio by suburb
data_by_suburb = pd.DataFrame(columns=['MEAN_LOSS_RATIO'])
for name, group in data.groupby('SUBURB'):
    data_by_suburb.loc[name, 'MEAN_LOSS_RATIO'] = group['LOSS'].sum()/group['TOTAL_COST'].sum()

print("Loss computed")

data['DAMAGE'] = pd.cut(data['LOSS_RATIO'], [-1.0, 0.02, 0.1, 0.5, 1.0], 
    labels=['no','slight','moderate','extensive'])

okay = data['DAMAGE'] != 'no'
casualty = pd.DataFrame(index=data.index, columns=['Severity'+str(i) for i in range(1, 5)])
tmp = data.ix[okay].apply(lambda row: compute_casualty(
    casualty_rate, row['DAMAGE'], row['BLDG_CLASS'], row['POPULATION']), axis=1)
casualty.loc[okay] = tmp

print("Casualty computed")

# casualty total
casualty.sum()
# Severity1    277.163146
# Severity2     65.053009
# Severity3      0.162633
# Severity4      0.162633
# dtype: float64

# casualty by suburbs
tmp = pd.DataFrame(columns=['Severity'+str(i) for i in range(1, 5)])
for name, group in data.groupby('SUBURB'):
    tmp = tmp.append(pd.DataFrame({name: casualty.ix[group.index].sum()}).transpose())

data_by_suburb = data_by_suburb.join(tmp)

data_by_suburb_zero = data_by_suburb.fillna(0)

