# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from sklearn.linear_model import LinearRegression
import os


def get_vec(word2index_list, vecs_list, key):
    res = []
    for i, word2index in enumerate(word2index_list):
        vecs = vecs_list[i]
        res.append([vecs[word2index[key]]])
    return res

key_list = ['short', 'return', 'List', 'static', 'new', 'for', 'if', 'except', 'try']

def select_common_tk(dict_1: dict, dict_2: dict):
    res = []
    for tk in dict_1.keys():
        if tk == '____UNKNOW____' or tk == '____PAD____':
            continue
        if '}' in tk or '{' in tk or ';' in tk or '(' in tk or ')' in tk or '.' in tk or '*' in tk:
            continue
        if tk in dict_2:
            res.append(tk)
        if len(res) == 1000:
            break
    res = res + key_list
    return res


def get_data():
    dir_name = './embedding_vec100_1/'
    dict_1, vec_1 = torch.load(dir_name + 'word2vec.vec')
    dict_2, vec_2 = torch.load(dir_name + 'doc2vec.vec')
    dict_3, vec_3 = torch.load(dir_name + 'fasttext.vec')
    common_tk = select_common_tk(dict_1, dict_2)
    linear_vec_1, linear_vec_2, linear_vec_3 = [], [], []
    for tk in common_tk:
        linear_vec_1.append([vec_1[dict_1[tk]]])
        linear_vec_2.append([vec_2[dict_2[tk]]])
        linear_vec_3.append([vec_3[dict_3[tk]]])
    linear_vec_1 = np.concatenate(linear_vec_1, axis=0)
    linear_vec_2 = np.concatenate(linear_vec_2, axis=0)
    linear_vec_3 = np.concatenate(linear_vec_3, axis=0)
    linear_model_1 = fit_data(linear_vec_1, linear_vec_2)
    linear_model_2 = fit_data(linear_vec_3, linear_vec_2)

    return linear_model_1, linear_model_2, linear_vec_1, linear_vec_2, linear_vec_3, common_tk


def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    #fig = plt.figure()
    color_list = ['green', 'red', 'blue']
    for i in range(data.shape[0]):
        space_id = int(label[i].split('_')[0])
        plt.text(
            data[i, 0], data[i, 1], str(label[i]),
            color=color_list[space_id],
            fontdict={'weight': 'bold', 'size': 9}
        )
    plt.show()


def fit_data(X, Y):
    linreg = LinearRegression()
    linreg.fit(X, Y)
    err = linreg.predict(X) - Y
    return linreg


def main():
    linear_model_1, linear_model_2, space_1, space_2, space_3, tk_list = get_data()
    print('Computing t-SNE embedding')
    model = TSNE(n_components=2, init='pca', random_state=0, n_jobs= -1)
    #model = PCA(n_components=2)
    new_space_1 = linear_model_1.predict(space_1)
    new_space_3 = linear_model_2.predict(space_3)

    orig_res = np.concatenate([space_1, space_2, space_3], axis=0)
    orig_res = model.fit_transform(orig_res)
    tk_len = len(tk_list)
    ky_len = len(key_list)
    orig_res = np.concatenate(
        [
            orig_res[:tk_len][-ky_len:],
            orig_res[:(2*tk_len)][-ky_len:],
            orig_res[-ky_len:]
        ], axis=0
    )

    new_res = np.concatenate([new_space_1, space_2, new_space_3], axis=0)
    new_res = model.fit_transform(new_res)
    new_res = np.concatenate(
        [
            new_res[:tk_len][-ky_len:],
            new_res[:(2*tk_len)][-ky_len:],
            new_res[-ky_len:]
        ], axis=0
    )

    label = ['0_' + k for k in key_list] + ['1_' + k for k in key_list] + ['2_' + k for k in key_list]
    plot_embedding(orig_res, label)
    plot_embedding(new_res, label)

    if not os.path.isdir('./intuition_res'):
        os.mkdir('./intuition_res')
    np.savetxt('./intuition_res/' + model.__class__.__name__ + 'orig.csv', orig_res, delimiter=',')
    np.savetxt('./intuition_res/' + model.__class__.__name__ + 'new.csv', new_res, delimiter=',')


def split_data(data, label):
    res_1, res_2 = [], []
    label_1, label_2 = [], []
    for i, val in enumerate(data):
        if i % 2 == 0:
            res_1.append(val.reshape([1, -1]))
            label_1.append(label[i])
        else:
            res_2.append(val.reshape([1, -1]))
            label_2.append(label[i])
    return np.concatenate(res_1, axis=0), np.concatenate(res_2, axis=0), label_1, label_2


if __name__ == '__main__':
    main()