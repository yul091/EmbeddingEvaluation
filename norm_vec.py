# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import collect_common_tk
from utils import BASEDICT


def generate_common_tk(code2vec_dict, code2seq_dict, dict_list):
    def subtk_code2seq(tk_list):
        for k in tk_list:
            if k not in code2seq_dict:
                return False
        return True

    def tk_dictlist(token):
        for word2index in dict_list:
            if token not in word2index:
                return False
        return True

    common = []
    for ori_tk in code2vec_dict:
        tk = ori_tk.split('|')
        if subtk_code2seq(tk):
            tk = ''.join(tk)
            if tk_dictlist(tk):
                common.append(ori_tk)
    return common


def generate_new_vec(
        common_dict, vec_list, dict_list, code2vec_dict,
        code2vec_vec,  code2seq_dict, code2seq_vec, dim):
    common_vec = [np.zeros([len(common_dict), dim]) for _ in range(len(dict_list) + 2)]

    for ori_tk in common_dict:
        tk_list = ori_tk.split('|')
        tk = ''.join(tk_list)
        for i, vec in enumerate(vec_list):
            now_dict = dict_list[i]
            tk_index = now_dict[tk]
            common_vec[i][common_dict[ori_tk]] = vec[tk_index]
        common_vec[-2][common_dict[ori_tk]] = code2vec_vec[code2vec_dict[ori_tk]]
        common_vec[-1][common_dict[ori_tk]] = sum([code2seq_vec[code2seq_dict[kkk]] for kkk in tk_list])

    return common_vec


def main():
    VEC_DIR = '/glusterfs/dataset/sxc180080/EmbeddingEvaluation/embedding_vec/'
    OUT_DIR = '/glusterfs/dataset/sxc180080/EmbeddingEvaluation/vec/'
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    vec_type = '100_2/'
    dim = 100
    VEC_DIR = VEC_DIR + vec_type
    OUT_DIR = OUT_DIR + vec_type
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    dict_list, vec_list, file_list = [], [], []
    for vec_file in os.listdir(VEC_DIR):
        if 'code' in vec_file:
            continue
        file_list.append(vec_file)
        vec_file = os.path.join(VEC_DIR, vec_file)
        d, v = torch.load(vec_file)
        dict_list.append(d)
        vec_list.append(v)

    code2vec_dict, code2vec_vec = torch.load(VEC_DIR + 'code2vec.vec')
    code2seq_dict, code2seq_vec = torch.load(VEC_DIR + 'ori_code2seq.vec')
    file_list.append('code2vec.vec')
    file_list.append('ori_code2seq.vec')

    common_tk = generate_common_tk(code2vec_dict, code2seq_dict, dict_list)
    common_dict = BASEDICT.copy()
    for tk in common_tk:
        if tk not in common_dict:
            common_dict[tk] = len(common_dict)
    norm_vec = generate_new_vec(
        common_dict, vec_list, dict_list, code2vec_dict,
        code2vec_vec,  code2seq_dict, code2seq_vec, dim=dim
    )
    for i, file_name in enumerate(file_list):
        file_name = OUT_DIR + file_name
        torch.save([common_dict, norm_vec[i]], file_name)
        print(file_name)


if __name__ == '__main__':
    main()