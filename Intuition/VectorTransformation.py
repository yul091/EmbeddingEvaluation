# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
import os
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from utils import collect_common_tk


def collect_vec(dict_list, vec_list, common_tk, vec_dim):
    res = [np.zeros([len(common_tk), vec_dim]) for _ in dict_list]
    for i, word2index in enumerate(dict_list):
        for j, tk in enumerate(common_tk):
            res[i][j] = vec_list[i][word2index[tk]]
    return res


def build_corrcoef(vec_mat):
    corr_mat = np.ones([len(vec_mat), len(vec_mat)])
    for i in range(len(vec_mat)):
        for j in range(i + 1, len(vec_mat)):
            linear_model = LinearRegression()
            linear_model.fit(vec_mat[i], vec_mat[j])
            pred = linear_model.predict(vec_mat[i])
            tmp = np.concatenate([pred, vec_mat[j]], 0)
            model = SparsePCA(n_components=1)
            tmp = model.fit_transform(tmp).reshape([2, -1])
            src, tar = tmp[0], tmp[1]
            corr_val = abs(np.corrcoef(src.reshape([-1]), tar.reshape([-1]))[0][1])
            corr_mat[i, j] = corr_val
            corr_mat[j, i] = corr_val
    return np.around(corr_mat, decimals=2, out=None)


def main():
    VEC_DIR = '/glusterfs/dataset/sxc180080/EmbeddingEvaluation/vec/100_2'
    vec_dim = 100
    sample_num = 1000
    dict_list, vec_list = [], []
    for vec_file in os.listdir(VEC_DIR):
        vec_file = os.path.join(VEC_DIR, vec_file)
        d, v = torch.load(vec_file)
        dict_list.append(d)
        vec_list.append(v)
        print(vec_file)

    common_tk = collect_common_tk(dict_list)
    index = np.random.randint(0, len(common_tk), sample_num, dtype=np.long)
    common_tk = [common_tk[i] for i in index]
    res = collect_vec(dict_list, vec_list, common_tk, vec_dim)
    corr_mat = build_corrcoef(res)
    print(corr_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(corr_mat)))
    ax.set_xticks(range(len(corr_mat)))
    im = ax.imshow(corr_mat, cmap=plt.cm.autumn_r)
    plt.colorbar(im)
    plt.title("This is a title")
    plt.show()


if __name__ == '__main__':
    main()