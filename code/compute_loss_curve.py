""" read loss loop through and compute the loss ratio
"""

import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import sys
import numpy as np

from summary_3_2 import read_sitedb_data, read_gm

def main(output_path):

    # read building value
    if sys.platform == 'win32':
        working_path = u'R:\scenario_Sydney'
    else:
        working_path = os.path.join(os.path.expanduser("~"),
                                'Projects/scenario_Sydney')

    #output_path = os.path.join(working_path,
    #                           'scen_risk_parallel')

    input_path = os.path.join(working_path,
                              'input')



    #file_bldg_value = os.path.join(output_path, site_tag + '_building_value.txt')
    # not sure with the first element

    #dd = pd.read_csv(file_bldg_value, header=None)

    data = read_sitedb_data(os.path.join(input_path,
                            'sitedb_sydney_soil.csv'))
    # better with data

    # read gmotion
    #(_, _, _, mmi) = read_gm(output_path, site_tag='sydney_soil')

    #data['MMI'] = pd.Series(mmi[:, 0], index=data.index)

    # list building loss txt files

    files_loss_txt = glob.glob(os.path.join(output_path,
                               site_tag + '_building_loss.txt*'))
    bldg_loss = dict()
    for file_ in files_loss_txt:
        with open(file_, 'r') as fid:
            tmp = fid.read().strip().split('\n')
            bid = np.array(tmp[1].split(), dtype=np.int64)
            loss = np.array(tmp[2].split(), dtype=float)
            assert len(bid) == len(loss)
            for bid_, loss_ in zip(bid, loss):
                if bid_ in bldg_loss:
                    raise('Something is wrong')
                else:
                    bldg_loss[bid_] = loss_

    # merge to data by bid
    df_bldg_loss = pd.DataFrame.from_dict(bldg_loss, orient='index')
    df_bldg_loss.reset_index(level=0, inplace=True)
    df_bldg_loss = df_bldg_loss.rename(columns = {'index':'BID', 0: 'LOSS'})

    ndata = pd.merge(data, df_bldg_loss, on='BID', how='outer')

    # file_bldg_loss = os.path.join(output_path, site_tag + '_building_loss.txt')

    # with open(file_bldg_loss, 'r') as fid:
    #     tmp = fid.read().strip().split('\n')

    # bldg_loss = np.array(tmp[2].split(), dtype=float)

    ndata['LOSS_RATIO'] = ndata['LOSS']/data['BLDG_COST']

    #grouped = data.groupby('STRUCTURE_CLASSIFICATION')
    #for item, groups in grouped:
    #    plt.figure()
    #    plt.plot(groups['MMI'], groups['LOSS_RATIO'])
    #    plt.title(item)

    #plt.show()

    # WHY 1?????

    # loss = pd.read_csv(file_bldg_loss, skiprows=1)



    # nevents = 453209

    # loss_array = np.zeros((nevents, len(files_loss_txt)))
    # for i, file_ in enumerate(files_loss_txt):

    #     with open(file_) as fid:
    #         tmp = fid.read().strip().split('\n')

    #     loss_array[:, i] = tmp[2:]

    # loss_arry = np.sum(loss_array, axis=1)

if __name__ == '__main__':
    main(argv[1])


