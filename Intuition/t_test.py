import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


def dict2list(word2index):
    res = {}
    for k in word2index:
        res[word2index[k]] = k
    return res


def main():
    VEC_DIR = '/glusterfs/dataset/sxc180080/EmbeddingEvaluation/vec/100_2'
    dict_list, vec_list = [], []
    file_list = []
    for vec_file in os.listdir(VEC_DIR):
        file_list.append(vec_file)
        vec_file = os.path.join(VEC_DIR, vec_file)
        d, v = torch.load(vec_file)
        dict_list.append(d)
        vec_list.append(v)
    word2index = dict_list[0]
    index2word = dict2list(dict_list[0])
    sample_num = (1000, 3)
    rand_index = np.random.randint(0, len(index2word)-1, sample_num)

    res = np.zeros([1000, len(dict_list)])
    for i, (tk_1, tk_2, tk_3) in enumerate(rand_index):
        for j, vec in enumerate(vec_list):
            vec_1, vec_2, vec_3 = \
                vec[tk_1].reshape([1, -1]), vec[tk_2].reshape([1, -1]), vec[tk_3].reshape([1, -1])
            res[i, j] = cosine_similarity(vec_1 - vec_3, vec_2 - vec_3)
    diff = np.random.uniform(-1, 1, res.shape)

    t_res = np.zeros([len(dict_list), len(dict_list)])
    for i in range(len(dict_list)):
        for j in range(len(dict_list)):
            tmp = stats.levene(res[:, i], res[:, j])
            if tmp.pvalue > 0.05:
                res_val = stats.ttest_ind(res[:, i], res[:, j], equal_var=True)
            else:
                res_val = stats.ttest_ind(res[:, i], res[:, j], equal_var=False)
            t_res[i, j] = res_val.pvalue
    print(np.around(t_res, decimals=-2))


main()
