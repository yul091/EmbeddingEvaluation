import os
import torch
import numpy as np

vec_dir = '/glusterfs/dataset/wei/EmbeddingData/embedding_vec/100_1/'

word2vec_dict, _ = torch.load('/glusterfs/dataset/sxc180080/EmbeddingEvaluation/embedding_vec/100_1/' + 'word2vec.vec')

code2seq_dict, _ = torch.load(vec_dir + 'ori_code2seq.vec')

code2vec_dict, _ = torch.load(vec_dir + 'code2vec.vec')

glove_dict, _ = torch.load(vec_dir + 'glove.vec')

print()

# todo reason 1: no lower case



