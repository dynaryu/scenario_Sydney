""" read loss loop through and compute the loss ratio
"""

import os
#import matplotlib.pyplot as plt
import glob
import sys
#import subprocess
import numpy as np

from summary_3_2 import read_sitedb_data


# def file_len(fname):
#     p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE)
#     result, err = p.communicate()
#     if p.returncode != 0:
#         raise IOError(err)
#     return int(result.strip().split()[0])


def main(args):

    output_path, bldg_class = args[0], args[1]

    site_tag = [x for x in os.listdir(output_path) if 'event' in x]
    site_tag = site_tag[0].replace('_event_set', '')

    # read building value
    if sys.platform == 'win32':
        working_path = u'R:\scenario_Sydney'
    else:
        working_path = os.path.join(os.path.expanduser("~"),
                                    'Projects/scenario_Sydney')

    file_event = os.path.join(output_path, site_tag + '_event_set',
                              'event_activity', 'event_activity.npy')

    activity = np.load(file_event)[0, 0, 0, :]  # (1, 1, 1, nevents)
    nevents = len(activity)

    input_path = os.path.join(working_path, 'input')

    data = read_sitedb_data(os.path.join(input_path,
                            'sitedb_sydney_soil.csv'))

    df_bldg_cost = data.groupby('STRUCTURE_CLASSIFICATION')['BLDG_COST'].sum()

    # denominator
    if bldg_class.lower() == 'all':
        bldg_cost = df_bldg_cost.sum()
    else:
        try:
            bldg_cost = df_bldg_cost[bldg_class]
        except KeyError:
            print "invlid bldg class: {}".format(bldg_class)

    # better with data

    # read gmotion
    #(_, _, _, mmi) = read_gm(output_path, site_tag='sydney_soil')

    #data['MMI'] = pd.Series(mmi[:, 0], index=data.index)

    # list building loss txt files

    files_loss_txt = glob.glob(os.path.join(output_path,
                               site_tag + '_building_loss.txt*'))

    loss_array = np.zeros((nevents, len(files_loss_txt)))

    for i, file_ in enumerate(files_loss_txt):

        with open(file_) as fid:
            tmp = fid.read().strip().split('\n')
            assert len(tmp[2:]) == nevents
            loss_array[:, i] = tmp[2:]

    loss_array = np.sum(loss_array, axis=1)

    result = np.vstack((activity, loss_array)).T

    np.save(os.path.join(output_path, 'activity_loss.npy'), result)

    return result

if __name__ == '__main__':
    summary = main(sys.argv[1:])
