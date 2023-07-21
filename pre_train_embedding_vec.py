import os
import torch
import argparse


from embedding_algorithms import *
from utils import set_random_seed


EMBED_DIR = 'embedding_vec/'
DATA_DIR = 'dataset/code_embedding_java_small'

if not os.path.isdir(EMBED_DIR):
    os.mkdir(EMBED_DIR)


def produce_vec(model, model_type, dir_name):
    vocab, vec = model.generate_embedding(model_type=model_type)
    if type(model_type) is str:
        save_name = dir_name + model.__class__.__name__ + model_type + '.vec'
    else:
        save_name = dir_name + model.__class__.__name__ + str(model_type) + '.vec'
    torch.save([vocab, vec], save_name)
    print(save_name)


def train_vec(vec_dim, epoch, data_dir, dir_name):
    if args.version == 0:           #torch_env
        from embedding_algorithms.fasttext2vec import FastEmbedding
        model = FastEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
        produce_vec(model, model_type='skipgram', dir_name=dir_name)
        produce_vec(model, model_type='cbow', dir_name=dir_name)

        model = Word2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
        produce_vec(model, model_type=0, dir_name=dir_name)
        produce_vec(model, model_type=1, dir_name=dir_name)

    elif args.version == 1:         #py3.6
        from embedding_algorithms.glove import GloVeEmbedding
        model = GloVeEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
        produce_vec(model, model_type='None', dir_name=dir_name)

        model = Doc2VecEmbedding(data_dir, None, None, vec_dim=vec_dim, epoch=epoch)
        produce_vec(model, model_type=0, dir_name=dir_name)
        produce_vec(model, model_type=1, dir_name=dir_name)

    else:
        raise TypeError


def main():
    for dim in [100,  200,  300]:
        for epoch in [2, 4, 6]:
            dir_name = EMBED_DIR + str(dim) + '_' + str(epoch) + '/'
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            train_vec(dim, epoch, DATA_DIR, dir_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', default=0, type=int, choices=[0, 1])
    args = parser.parse_args()
    set_random_seed(100)
    main()
