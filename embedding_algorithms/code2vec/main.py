from torch.utils.data import DataLoader
import pickle
import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import datetime
import argparse
from utils import set_random_seed
import numpy as np
import os
from tqdm import tqdm

from se_tasks.code_summary.scripts.main import train_model


def main(args_set):
    tk_path = args_set.tk_dict
    train_path = args_set.train_data
    test_path = args_set.test_data
    embed_dim = args_set.embed_dim
    train_batch = args.batch
    out_dir = '../../embedding_vec/' + str(args_set.embed_dim) + '_' + str(args.epochs) + '/'
    model, token2index = train_model(
        tk_path, train_path, test_path,
        embed_dim, embed_type=1, vec_path=None,
        experiment_name='code2vec',
        train_batch=train_batch, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, max_size=None,
        out_dir=out_dir, device_id=args.device,

    )
    weight = model.node_embedding.weight.detach().cpu().numpy()
    assert len(token2index) == len(weight)
    torch.save([token2index, weight], out_dir + 'code2vec.vec')
    print('finish code2vec embedding training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--device', default=0, help='gpu id')

    #todo: need change

    parser.add_argument('--train_data', type=str, default='dataset/java-small-preprocess/train.pkl')
    parser.add_argument('--test_data', type=str, default='dataset/java-small-preprocess/val.pkl')
    parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--tk_dict', type=str, default='dataset/java-small-preprocess/tk.pkl')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--train_data', type=str, default='../../dataset/java-small-preprocess/train.pkl')
    parser.add_argument('--test_data', type=str, default='../../dataset/java-small-preprocess/val.pkl')
    parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--tk_dict', type=str, default='../../dataset/java-small-preprocess/tk.pkl')
    args = parser.parse_args()
    # set_random_seed(10)
    main(args)
