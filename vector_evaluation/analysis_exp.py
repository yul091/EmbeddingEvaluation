import pickle
import matplotlib.pyplot as plt
import numpy as np
from vector_evaluation.experiment import Metric


res_file = './res/500_exp_m.res'
metric_key = [
    'acc',
    'map',
    'ndcg',
    'mrr',
    'recall_3',
    'prec_3'
]

with open(res_file, 'rb') as f:
    ori_result = pickle.load(f)

if type(ori_result) == dict:
    ori_result = list(ori_result.values())


result = sorted(ori_result, key=lambda student: student[2])


res_mat = np.zeros([len(result), len(metric_key)])
rnd_mat = np.zeros([len(result), len(metric_key)])

for i, k in enumerate(metric_key):
    y_1 = []
    y_2 = []
    x = [5, 6, 7, 8, 9, 10]
    for j, r in enumerate(result):
        y_1.append(r[0].res[k])
        y_2.append(r[0].random[k])
        res_mat[j, i] = r[0].res[k]
        rnd_mat[j, i] = r[0].random[k]
    plt.plot(x, y_1, 'r', label='app')
    plt.plot(x, y_2, 'b', label='random')
    plt.title(k)
    plt.show()
np.savetxt('./res/res.csv', res_mat, delimiter=',')
np.savetxt('./res/rnd.csv', rnd_mat, delimiter=',')
