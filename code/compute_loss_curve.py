""" read loss loop through and compute the loss ratio
"""

import os
import matplotlib.pyplot as plt
#import glob
#import pandas

from summary_3_2 import read_sitedb_data, read_gm

# read building value
working_path = os.path.join(os.path.expanduser("~"),
                            'Projects/scenario_Sydney')

output_path = os.path.join(working_path,
                           'scen_risk')

input_path = os.path.join(working_path,
                          'input')

site_tag = 'sydney_soil'

file_bldg_value = os.path.join(output_path, site_tag + '_building_value.txt')
# not sure with the first element

#dd = pd.read_csv(file_bldg_value, header=None)

data = read_sitedb_data(os.path.join(input_path,
                        'sitedb_sydney_soil.csv'))
# better with data


# list building loss txt files

#files_loss_txt = glob.glob(os.path.join(output_path,
#                           site_tag + '_building_loss.txt*'))

file_bldg_loss = os.path.join(output_path, site_tag + '_building_loss.txt')

with open(file_bldg_loss, 'r') as fid:
    tmp = fid.read().strip().split('\n')

bldg_loss = np.array(tmp[2].split(), dtype=float)

# read gmotion
(_, _, _, mmi) = read_gm(eqrm_output_path, site_tag='sydney_soil')

data['MMI'] = pd.Series(mmi[:, 0], index=data.index)

data['LOSS_RATIO'] = bldg_loss/data['BLDG_COST']

grouped = data.groupby('STRUCTURE_CLASSIFICATION')

for item, groups in grouped:
    plt.figure()
    plt.plot(groups['MMI'], groups['LOSS_RATIO'])
    plt.title(item)

# WHY 1?????

# loss = pd.read_csv(file_bldg_loss, skiprows=1)



# nevents = 453209

# loss_array = np.zeros((nevents, len(files_loss_txt)))
# for i, file_ in enumerate(files_loss_txt):

#     with open(file_) as fid:
#         tmp = fid.read().strip().split('\n')

#     loss_array[:, i] = tmp[2:]

# loss_arry = np.sum(loss_array, axis=1)