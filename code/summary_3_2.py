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
4. compute mean loss ratio
5. save it for further calculation

What to present
-. rock vs. soil motion
-. soil class
-. mean loss ratio by SA1
-. exposure table (Age vs. type)
-. vulnerability
-. fatality table
-. fatality by SA1
'''

import numpy as np
import os
import pandas as pd
from collections import OrderedDict
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_sitedb_data(sitedb_file):
    ''' read sitedb file '''

    data = pd.read_csv(sitedb_file, dtype={'SA1_CODE': str})
    data['BLDG_COST'] = data['BUILDING_COST_DENSITY'] * data['FLOOR_AREA']\
        * data['SURVEY_FACTOR']
    data['CONTENTS_COST'] = data['CONTENTS_COST_DENSITY'] * data['FLOOR_AREA']\
        * data['SURVEY_FACTOR']
    data['TOTAL_COST'] = data['BLDG_COST'] + data['CONTENTS_COST']

    return data


def read_gm(eqrm_output_path, site_tag):
    ''' read ground motion, an output of EQRM run'''

    from eqrm_code.RSA2MMI import rsa2mmi_array

    atten_periods = np.load(os.path.join(eqrm_output_path, site_tag +
                            '_motion/atten_periods.npy'))

    selected_periods = [0.0, 0.3, 1.0]

    idx_period = [(np.abs(atten_periods - period)).argmin()
                  for period in selected_periods]

    try:
        print 'GM at soil is loaded'
        gmotion = np.load(os.path.join(eqrm_output_path, site_tag +
                          '_motion/soil_SA.npy'))  # (1,1,1,sites,nsims,nperiods)
    except IOError:
        print 'GM at bedrock is loaded'
        gmotion = np.load(os.path.join(eqrm_output_path, site_tag +
                          '_motion/bedrock_SA.npy'))

    pga = gmotion[0, 0, 0, :, :, idx_period[0]]
    sa03 = gmotion[0, 0, 0, :, :, idx_period[1]]
    sa10 = gmotion[0, 0, 0, :, :, idx_period[2]]
    mmi = rsa2mmi_array(sa10, period=1.0)

    return(pga, sa03, sa10, mmi)


def compute_vulnerability(mmi, bldg_class):

    def inv_logit(x):
        return np.exp(x)/(1.0 + np.exp(x))

    def compute_mu(mmi, **kwargs):

        coef = {
            "t0": -8.56,
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


def compute_vulnerability_retrofit(mmi, bldg_class):

    def inv_logit(x):
        return np.exp(x)/(1.0 + np.exp(x))

    def compute_mu(mmi, **kwargs):

        coef = {
            "t0": -8.56,
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
    flag_pre = False  # for retrofitted building

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


def plot_vulnerabilty_retrofit():
    ''' plot vulnerability'''

    mmi_range = np.arange(4.0, 10.0, 0.05)

    vul = OrderedDict()
    for bldg in ['Timber_Pre1945', 'Timber_Post1945', 'URM_Pre1945',
                 'URM_Post1945']:
        tmp = []
        for val in mmi_range:
            tmp.append(compute_vulnerability_retrofit(val, bldg))
        vul[bldg] = np.array(tmp)

    line_type = ['b--', 'b-', 'r--', 'r-']
    plt.figure()
    for bldg, line_ in zip(vul.keys(), line_type):
        label_str = bldg.replace('_', ':')
        plt.plot(mmi_range, vul[bldg],  line_, label=label_str)

    plt.legend(loc=2)
    plt.grid(1)
    plt.xlabel('MMI')
    plt.ylabel('Loss ratio')


def plot_vulnerabilty():
    ''' plot vulnerability'''

    mmi_range = np.arange(4.0, 10.0, 0.05)

    vul = OrderedDict()
    for bldg in ['Timber_Pre1945', 'Timber_Post1945', 'URM_Pre1945',
                 'URM_Post1945']:
        tmp = []
        for val in mmi_range:
            tmp.append(compute_vulnerability(val, bldg))
        vul[bldg] = np.array(tmp)

    line_type = ['b--', 'b-', 'r--', 'r-']
    plt.figure()
    for bldg, line_ in zip(vul.keys(), line_type):
        label_str = bldg.replace('_', ':')
        plt.plot(mmi_range, vul[bldg],  line_, label=label_str)

    plt.legend(loc=2)
    plt.grid(1)
    plt.xlabel('MMI')
    plt.ylabel('Loss ratio')
    plt.xlim([4, 7])
    plt.ylim([0, 0.2])

###############################################################################
# main

def main(flag_retrofit=0):

    working_path = os.path.join(os.path.expanduser("~"), 'Projects/scenario_Sydney')
    eqrm_input_path = os.path.join(working_path, 'input')
    eqrm_output_path = os.path.join(working_path, 'gm_portfolio_Mw5.0')
    data_path = os.path.join(working_path, 'data')
    hazus_data_path = os.path.join(data_path, 'hazus')

    # read inventory
    data = read_sitedb_data(os.path.join(data_path,
                            'sydney_EQRMrevised_NEXIS2015.csv'))

    # read gmotion
    (_, _, _, mmi) = read_gm(eqrm_output_path, site_tag='sydney_soil')

    # combine mmi
    data['MMI'] = pd.Series(mmi[:, 0], index=data.index)

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

    if flag_retrofit:

        print "Computing using the models for retrofitted buildings"

        okay = data[data['MMI'] > 4.0].index
        data.loc[okay, 'LOSS_RATIO'] = \
            data.ix[okay].apply(lambda row: compute_vulnerability_retrofit(
                row['MMI'], row['BLDG_CLASS']), axis=1)

        # MEAN LOSS RATIO by SA1
        grouped = data.groupby('SA1_CODE')
        mean_loss_ratio_by_SA1 = grouped['LOSS_RATIO'].mean()
        mean_loss_ratio_by_SA1.fillna(0, inplace=True)
        mean_loss_ratio_by_SA1.columns = ['SA1_CODE', 'LOSS_RATIO']
        file_ = os.path.join(data_path,'mean_loss_ratio_by_SA1_retrofit.csv')
        mean_loss_ratio_by_SA1.to_csv(file_)
        print "%s is created" %file_

        # clean up data and save
        selected_columns = ['BID', 'SA1_CODE', 'POPULATION', 'BLDG_CLASS',\
            'TOTAL_COST', 'LOSS_RATIO']
        ndata = data.loc[okay, selected_columns]
        file_ = os.path.join(data_path,'loss_ratio_by_bldg_retrofit.csv')
        ndata.to_csv(file_, index=False)
        print("%s is created" %file_)

    else:

        okay = data[data['MMI'] > 4.0].index
        data.loc[okay, 'LOSS_RATIO'] = \
            data.ix[okay].apply(lambda row: compute_vulnerability(
                row['MMI'], row['BLDG_CLASS']), axis=1)

        # MEAN LOSS RATIO by SA1
        grouped = data.groupby('SA1_CODE')
        mean_loss_ratio_by_SA1 = grouped['LOSS_RATIO'].mean()
        mean_loss_ratio_by_SA1.fillna(0, inplace=True)
        mean_loss_ratio_by_SA1.columns = ['SA1_CODE', 'LOSS_RATIO']
        file_ = os.path.join(data_path,'mean_loss_ratio_by_SA1.csv')
        mean_loss_ratio_by_SA1.to_csv(file_)
        print "%s is created" %file_

        # clean up data and save
        selected_columns = ['BID', 'SA1_CODE', 'POPULATION', 'BLDG_CLASS',\
            'TOTAL_COST', 'LOSS_RATIO']
        ndata = data.loc[okay, selected_columns]
        file_ = os.path.join(data_path,'loss_ratio_by_bldg.csv')
        ndata.to_csv(file_, index=False)
        print("%s is created" %file_)

if __name__ == '__main__':
    main(flag_retrofit=0)
