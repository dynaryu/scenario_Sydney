import scipy
from scipy import stats
import numpy as np
import sys
import os
import copy
import pandas as pd

def sample_vulnerability(mean_lratio, nsample=1000, cov=1.0):

    """
    The probability density function for `gamma` is::

        gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)

    for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.

    The scale parameter is equal to ``scale = 1.0 / lambda``.

    `gamma` has a shape parameter `a` which needs to be set explicitly. For
    instance:

        >>> from scipy.stats import gamma
        >>> rv = gamma(3., loc = 0., scale = 2.)

    shape: a 
    scale: b
    mean = a*b
    var = a*b*b
    cov = 1/sqrt(a) = 1/sqrt(shape)
    shape = (1/cov)^2
    """

    shape_ = np.power(1/cov, 2)
    scale_ = mean_lratio/shape_
    sample = stats.gamma.rvs(shape_, loc=0, scale=scale_,\
        size=(nsample, len(mean_lratio)))

    sample[sample > 1] = 1.0
    #sample = pd.DataFrame(sample, columns=['LOSS_RATIO'])
    #sample = pd.Series(sample)

    # sample['DAMAGE'] = pd.cut(sample['LOSS_RATIO'], [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1], 
    # labels=['no','slight','moderate','extensive', 'complete'])
    # sample['DAMAGE'] = sample['DAMAGE'].astype(str)

    # # split complete either complete w/ collapse and complete w/o collapse
    # idx_complete = sample[sample['DAMAGE'] == 'complete'].index
    # ncomplete = len(idx_complete)

    # if ncomplete > 2:
    #     sample.loc[idx_complete, 'DAMAGE']  = \
    #         np.random.choice(['complete','collapse'], ncomplete,\
    #             p=[1-prob_collapse, prob_collapse])

    return sample

def assign_damage_state(data, okay, sample, damage_thresholds, damage_labels):

    nsample = sample.shape[0]
    df_damage = pd.DataFrame(index=okay, columns=range(nsample)) #, dtype=str)
    
    # assign damage by loss ratio
    for i in range(nsample):
        df_damage[i] = pd.cut(sample[i, :], damage_thresholds, 
        labels=damage_labels)
        df_damage[i] = df_damage[i].cat.add_categories(['collapse'])

    df_damage['BLDG_CLASS'] = data.loc[okay, 'BLDG_CLASS']

    # assign collapse
    for name, group in df_damage.groupby('BLDG_CLASS'):
        idx_group = group.index
        group_array = group.values[:, :-1]

        prob_collapse = collapse_rate[name]*1.0e-2
        (idx_complete, idy_complete) = np.where(group_array == 'complete')
        ncomplete = len(idx_complete)
        temp = np.random.choice(['complete', 'collapse'], size=ncomplete,\
            p=[1-prob_collapse, prob_collapse])

        idx_collapse = np.where(temp=='collapse')[0]
        for i in idx_collapse:
            df_damage.loc[idx_group[idx_complete[i]], idy_complete[i]] =\
                'collapse'

    return df_damage 

def assign_casualty(df_damage, casualty_rate, collapse_rate):

    casualty = {}
    for severity in casualty_rate.keys():

        df_casualty = pd.DataFrame(df_damage.values)

        for bldg, group in df_casualty.groupby(df_casualty.shape[1]-1):
            print "%s: %s" %(severity, bldg)
            replace_dic = casualty_rate[severity][bldg]
            for ds in replace_dic.keys():
                df_casualty[group==ds] = replace_dic[ds]

                #df_casualty.loc[group==ds] = replace_dic[ds]
                #df_casualty.where(group==ds, replace_dic[ds], inplace=True)

        casualty[severity] = df_casualty.copy()

    return casualty

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
            rate_ = tmp.ix[okay].to_dict()
        else:
            rate_ = tmp.to_dict()

        # create dictionary (severity, bldg, damage)
        for level in rate_.keys():
            for bldg in rate_[level].keys():
                casualty_rate.setdefault(level, {}).setdefault(bldg, {})[ds] =\
                    rate_[level][bldg]

    # no damage
    for level in casualty_rate.keys():
        for bldg in casualty_rate[level].keys():
            casualty_rate[level][bldg]['no'] = 0.0

    return casualty_rate

def read_hazus_collapse_rate(hazus_data_path, selected_bldg_class=None):

    # read collapse rate (table 13.8)
    fname = os.path.join(hazus_data_path, 'hazus_collapse_rate.csv')
    collapse_rate = pd.read_csv(fname, skiprows=1, names=['Bldg type','rate'], 
                index_col=0, usecols=[1, 2])
    collapse_rate = collapse_rate.to_dict()['rate']

    if selected_bldg_class is  not None:
        removed_bldg_class = (set(collapse_rate.keys())).difference(set(
            selected_bldg_class))
        [collapse_rate.pop(item) for item in removed_bldg_class]

    return collapse_rate 

###############################################################################
# main 

working_path = os.path.join(os.path.expanduser("~"),'Projects/scenario_Sydney')
data_path = os.path.join(working_path, 'data')
hazus_data_path = os.path.join(data_path, 'hazus')

# read data
data = pd.read_csv(os.path.join(data_path, 'loss_ratio_by_bldg.csv'), dtype={'SA1_CODE': str})

# MEAN LOSS RATIO by SA1
grouped = data.groupby('SA1_CODE')
mean_loss_ratio_by_SA1 = grouped['LOSS_RATIO'].mean()
mean_loss_ratio_by_SA1.to_csv(os.path.join(data_path,'mean_loss_ratio_by_SA1.csv'), index=False)

# sample loss ratio assuming gamma distribution with constant cov
nsample = 100
cov = 1.0
okay = data[~data['LOSS_RATIO'].isnull()].index
mean_lratio = data.loc[okay, 'LOSS_RATIO'].values
pop = data.loc[okay, 'POPULATION'].values

np.random.seed(99)
# np.array(nsample, nbldgs)
sample = sample_vulnerability(mean_lratio, nsample=nsample, cov=cov)

# assign damage state
damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]

df_damage = assign_damage_state(data, okay, sample, damage_thresholds,\
    damage_labels):

np.save(sample, os.path.join(data_path, 'sample_loss_ratio.npy'))
del sample

# read hazus indoor casualty data
casualty_rate = read_hazus_casualty_data(hazus_data_path,\
    selected_bldg_class = data['BLDG_CLASS'].unique())

# read hazus collapse rate data
collapse_rate = read_hazus_collapse_rate(hazus_data_path,\
    selected_bldg_class = data['BLDG_CLASS'].unique())

# assign casualty rate by damage state
# casualty{'Severity'}.DataFrame
casualty = assign_casualty(df_damage, casualty_rate, collapse_rate)
       
# save casualty
for severity in casualty.keys():
    casualty[severity].to_csv(os.path.join(data_path,'%s_%s.csv' %('casualty_rate',\
     severity)), index=False)

# multiply casualty with population
casualty_number = pd.DataFrame(index=range(nsample), columns=casualty_rate.keys())
for severity in casualty_rate.keys():
    value_ = 0.01*casualty[severity].\
        iloc[:, range(nsample)].multiply(pop, axis=0)
    casualty_number.loc[:, severity] = value_.sum(axis=0)

casualty_number.to_csv(os.path.join(data_path,'casualty_number.csv'), index=False)

#casualty_number[severity]['BLDG_CLASS'] = data.loc[okay, 'BLDG_CLASS']
#grouped = casualty_all[severity].groupby('BLDG_CLASS')
#grouped.sum()

# aggregate over portfolio

# using pandas
#casualty_number = {}
#for severity in casualty_rate.keys():
#    casualty_ = casualty[severity].values[:, :-1]
#    casualty_number[severity] = 0.01*casualty_*pop[:, np.newaxis]

# mean casualty

"""
for name, group in df_damage.groupby('BLDG_CLASS'):
    idx_group = group.index
    group_array = group.values[:,:-1]


casualty = pd.DataFrame(index=data.index, columns=['Severity'+str(i) for i in range(1, 5)])
tmp = data.ix[okay].apply(lambda row: compute_casualty(
    casualty_rate, row['DAMAGE'], row['BLDG_CLASS'], row['POPULATION']), axis=1)
casualty.loc[okay] = tmp


data_by_SA1.loc[name, 'MEAN_LOSS_RATIO'] = group['LOSS'].sum()/group['TOTAL_COST'].sum()


ncomplete = len(idx_x)

np.random.choice(['complete', 'collapse'], size=ncomplete, replace=True, p=)

# wonder if there is an euqivalent pd command for np.where

for name, group in data.groupby(''):
    data_by_SA1.loc[name, 'MEAN_LOSS_RATIO'] = group['LOSS'].sum()/group['TOTAL_COST'].sum()




data['LOSS'] = data['LOSS_RATIO'] * data['TOTAL_COST']

# mean loss ratio by SA1
data_by_SA1 = pd.DataFrame(columns=['MEAN_LOSS_RATIO'])
for name, group in data.groupby('SA1_CODE_STR'):
    data_by_SA1.loc[name, 'MEAN_LOSS_RATIO'] = group['LOSS'].sum()/group['TOTAL_COST'].sum()

print("Loss computed")

data['DAMAGE'] = pd.cut(data['LOSS_RATIO'], [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1], 
    labels=['no','slight','moderate','extensive, complte'])


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
for name, group in data.groupby('SA1_CODE_STR'):
    tmp = tmp.append(pd.DataFrame({name: casualty.ix[group.index].sum()}).transpose())

data_by_SA1 = data_by_SA1.join(tmp)

for name, group in data.groupby('SA1_CODE_STR'):
    count_by_bldg = group['BLDG_CLASS'].value_counts()
    data_by_SA1.loc[name, 'MAJOR_BLDG'] = count_by_bldg.argmax()

data_by_SA1_zero = data_by_SA1.fillna(0)
data_by_SA1_zero.to_csv(os.path.join(data_path,'result_by_SA1.csv'), index_label='SA1_CODE')

"""
