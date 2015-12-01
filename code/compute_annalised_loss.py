"""
plot loss curve and compute anualised loss
"""

import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


def integrate_backwards(x_array, y_array):
    """
    Given x_array and y_array values, calculate the backwards cumulative
    area under the y axis of the curve.

    preconditions;
    x is sorted, ascending.
    y is sorted, ascending.
    x and y values are positive
    x and y are vectors.

    return:
      Vector[0] is the area under the y axis of the curve.
      if n is the last index of the vector
      vector[n-i] is the area under the y-axis from y_array[0]
      to y_array[i], so
      vector[n] is 0.
    """

    n = len(y_array)
    tmp_ann_loss = np.zeros(n)
    #print "range(n-2, -1, -1)", range(n-2, -1, -1)
    for s in range(n-2, -1, -1):
        height = abs(y_array[s+1] - y_array[s])
        tri_width = abs(x_array[s+1] - x_array[s])
        rect_width = min(x_array[s+1], x_array[s])
        tri_area = 0.5 * tri_width * height
        rect_area = height * rect_width
        tmp_ann_loss[s] = tmp_ann_loss[s+1] + tri_area + rect_area
    return tmp_ann_loss


def compute_exceedance_rate(loss_by_event, event_activity):

    idx_loss = loss_by_event.argsort()[::-1]  # sort in descending order
    exceedance_rate = np.cumsum(event_activity[idx_loss])
    sorted_loss_by_event = loss_by_event[idx_loss]
    prob_exceed = 1.0-np.exp(-1.0*exceedance_rate)

    return sorted_loss_by_event, exceedance_rate, prob_exceed


working_path = os.path.join(os.path.expanduser("~"),
                            'Projects/scenario_Sydney')

bldg_classes = ['Timber_Pre1945', 'Timber_Post1945', 'URM_Pre1945',
                'URM_Post1945']

# all
df_loss = pd.DataFrame(columns=bldg_classes)
for i, bldg_class in enumerate(bldg_classes):
    file_ = os.path.join(working_path, 'prob_risk_' + bldg_class,
                         'activity_loss_all.npy')
    loss = np.load(file_)

    if i == 0:
        activity = loss[:, 0]
    else:
        assert np.allclose(loss[:, 0], activity)

    df_loss[bldg_class] = loss[:, 1]

loss_event = df_loss.sum(axis=1).values

(sorted_loss_by_event, exceed_rate, prob_exceed) = \
    compute_exceedance_rate(loss_event, activity)

ann_loss = integrate_backwards(sorted_loss_by_event, prob_exceed)

# each bldg class
loss_curve = dict()
for i, bldg_class in enumerate(bldg_classes):
    file_ = os.path.join(working_path, 'prob_risk_' + bldg_class,
                         'activity_loss_{}.npy'.format(bldg_class))
    loss = np.load(file_)

    if i == 0:
        activity = loss[:, 0]
    else:
        assert np.allclose(loss[:, 0], activity)

    (sorted_loss_by_event_,
     exceed_rate_,
     prob_exceed_) = compute_exceedance_rate(loss[:, 1], activity)

    loss_curve.setdefault(bldg_class, {})['loss'] = sorted_loss_by_event_
    loss_curve[bldg_class]['rate'] = exceed_rate_
    loss_curve[bldg_class]['prob'] = prob_exceed_

# plt.figure()
# plt.semilogy(sorted_loss_by_event, prob_exceed, 'k-', label='All')

# color_dic = {'Timber_Pre1945': 'b--', 'Timber_Post1945': 'b-',
#              'URM_Pre1945': 'r--', 'URM_Post1945': 'r-'}

# for bldg_class in bldg_classes:
#     plt.semilogy(loss_curve[bldg_class]['loss'],
#                  loss_curve[bldg_class]['prob'], color_dic[bldg_class],
#                  label=bldg_class)
# plt.legend(loc=0)

# plt.show()


# all_retrofit
bldg_classes_retro = ['Timber_Pre1945_retrofit', 'Timber_Post1945',
                      'URM_Pre1945_retrofit', 'URM_Post1945']

df_loss_retro = pd.DataFrame(columns=bldg_classes)
for i, bldg_class in enumerate(bldg_classes_retro):
    file_ = os.path.join(working_path, 'prob_risk_' + bldg_class,
                         'activity_loss_all.npy')
    loss = np.load(file_)

    if i == 0:
        activity = loss[:, 0]
    else:
        assert np.allclose(loss[:, 0], activity)

    df_loss_retro[bldg_class] = loss[:, 1]

loss_event_retro = df_loss_retro.sum(axis=1).values

(sorted_loss_by_event_retro, exceed_rate_retro, prob_exceed_retro) = \
    compute_exceedance_rate(loss_event_retro, activity)

ann_loss_retro = integrate_backwards(sorted_loss_by_event_retro,
                                     prob_exceed_retro)

# each bldg class retrofit
loss_curve_retro = dict()
for bldg_class in ['Timber_Pre1945_retrofit', 'URM_Pre1945_retrofit']:

    bldg_class_name = bldg_class[:-9]

    file_ = os.path.join(working_path, 'prob_risk_' + bldg_class,
                         'activity_loss_{}.npy'.format(bldg_class_name))
    loss = np.load(file_)

    assert np.allclose(loss[:, 0], activity)

    (sorted_loss_by_event_,
     exceed_rate_,
     prob_exceed_) = compute_exceedance_rate(loss[:, 1], activity)

    loss_curve_retro.setdefault(bldg_class_name, {})['loss'] = sorted_loss_by_event_
    loss_curve_retro[bldg_class_name]['rate'] = exceed_rate_
    loss_curve_retro[bldg_class_name]['prob'] = prob_exceed_

# comparison


