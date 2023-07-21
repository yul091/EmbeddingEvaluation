import os
import torch
from vector_evaluation.SemanticMetric import SemanticCosine
import joblib

# vec_dir = '/glusterfs/data/wei/EmbeddingData/embedding_vec/100_2/'
vec_dir = '/home/yxl190090/dataset/EmbeddingEvaluation/embedding_vec/100_2/'
dict_list, vec_list = {}, {}


def trainsfer_dict(word2index):
    res = {}
    for k in word2index:
        res[word2index[k]] = k
    return word2index


file_list = []
for file_name in os.listdir(vec_dir):
    dict_val, vec_val = torch.load(vec_dir + file_name)
    key_val = file_name
    dict_list[key_val] = dict_val
    vec_list[key_val] = vec_val
    file_list.append(file_name)

vec = []
word2index = []
index2word = []
for k in vec_list:
    vec.append(vec_list[k])
    word2index.append(dict_list[k])
    index2word.append(trainsfer_dict(dict_list[k]))


metric = SemanticCosine(
    vec, index2word=index2word, word2index=word2index, sampling_num=1000
)
metric.calculate_score()
# metric = SemanticCosine(vec, sampling_num=5000)
# metric.calculate_score()
# metric = SemanticCosine(vec, sampling_num=10000)
# metric.calculate_score()