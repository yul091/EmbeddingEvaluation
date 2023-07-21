import os
import torch
import numpy as np


def find_common_token(src_index, dict_list):
    src_dict = dict_list[src_index]
    count = np.zeros([len(dict_list)])
    for tk in src_dict:
        is_common = 0
        for tar_dict in dict_list:
            if tk in tar_dict:
                is_common += 1
        count[:is_common] += 1
    print('index', src_index)
    print('count', count)


#vec_dir = '/glusterfs/dataset/sxc180080/EmbeddingEvaluation/embedding_vec/100_2/'
# vec_dir = '/glusterfs/dataset/wei/EmbeddingData/embedding_vec/100_2/'
vec_dir = '/home/yxl190090/dataset/EmbeddingEvaluation/embedding_vec/100_2/'

dict_list, file_list = [], []
for file_name in os.listdir(vec_dir):
    dict_val, _ = torch.load(vec_dir + file_name)
    dict_list.append(dict_val)
    file_list.append(file_name)

for i, tar_dict in enumerate(dict_list):
    print(file_list[i], len(tar_dict))
find_common_token(0, dict_list)
find_common_token(1, dict_list)
find_common_token(2, dict_list)
find_common_token(3, dict_list)
find_common_token(4, dict_list)
find_common_token(5, dict_list)
