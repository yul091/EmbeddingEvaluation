from __future__ import print_function

import gc
import os
import argparse
import datetime
import numpy as np
import joblib
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from gensim.models import word2vec

from se_tasks.code_completion.scripts.vocab import VocabBuilder
from se_tasks.code_completion.scripts.dataloader import Word2vecLoader
from se_tasks.code_completion.scripts.util import AverageMeter, accuracy
from se_tasks.code_completion.scripts.util import adjust_learning_rate
from se_tasks.code_completion.scripts.model import Word2vecPredict


def preprocess_data():
    print("===> creating vocabs ...")
    train_path = args.train_data
    test_path = args.test_data
    pre_embedding_path = args.embedding_path
    if args.embedding_type == 0:
        d_word_index, embed = torch.load(pre_embedding_path)
        print('load existing embedding vectors, name is ', pre_embedding_path)
    elif args.embedding_type == 1:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        print('create new embedding vectors, training from scratch')
    elif args.embedding_type == 2:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        embed = torch.randn([len(d_word_index), args.embedding_dim]).cuda()
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')

    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float).cuda()
        assert embed.size()[1] == args.embedding_dim

    if not os.path.exists('se_tasks/code_completion/result'):
        os.mkdir('se_tasks/code_completion/result')

    train_loader = Word2vecLoader(train_path, d_word_index, batch_size=args.batch_size)
    val_loader = Word2vecLoader(test_path, d_word_index, batch_size=args.batch_size)

    return d_word_index, embed, train_loader, val_loader


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def test(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target, _) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))
    return top1.avg


def main():
    d_word_index, embed, train_loader, val_loader = preprocess_data()
    vocab_size = len(d_word_index)
    print('vocab_size is', vocab_size)
    model = Word2vecPredict(d_word_index, embed)
    model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print('training dataset size is ', train_loader.n_samples)
    t1 = datetime.datetime.now()
    time_cost = None
    for epoch in range(1, args.epochs + 1):
        st = datetime.datetime.now()
        train(train_loader, model, criterion, optimizer)
        ed = datetime.datetime.now()
        if time_cost is None:
            time_cost = ed - st
        else:
            time_cost += (ed - st)
        res = test(val_loader, model, criterion)
        print(epoch, 'cost time', ed - st, 'accuracy is', res.item())
    print('time cost', time_cost / args.epochs)
    t2 = datetime.datetime.now()

    weight_save_model = os.path.join('se_tasks/code_completion', args.weight_name)
    torch.save(model.encoder.weight, weight_save_model)
    print('result is ', res)
    print('result is ', res, 'cost time', t2 - t1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--embedding_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--min_samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--weight_name', type=str, default='1', help='model name')
    parser.add_argument('--embedding_path', type=str, default='embedding_vec100_1/fasttext.vec')
    parser.add_argument('--train_data', type=str, default='se_tasks/code_completion/dataset/train.tsv',)
    parser.add_argument('--test_data', type=str, default='se_tasks/code_completion/dataset/test.tsv', help='model name')
    parser.add_argument('--embedding_type', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code2vec')

    args = parser.parse_args()
    main()
