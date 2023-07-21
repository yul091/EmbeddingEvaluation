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
import matplotlib.pyplot as plt

from se_tasks.code_authorship import VocabBuilder
from se_tasks.code_authorship import TextClassDataLoader
from se_tasks.code_authorship import RNN
from se_tasks.code_authorship import AverageMeter, accuracy
from se_tasks.code_authorship import adjust_learning_rate


def train_model(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (x, target, seq_lengths) in enumerate(train_loader):
        if args.cuda:
            x = x.cuda()
            target = target.cuda()
        output = model(x, seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, x.size(0))
        top1.update(prec1[0][0], x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time


def test_model(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (x, target, seq_lengths) in enumerate(val_loader):
        if args.cuda:
            x = x.cuda()
            target = target.cuda()
        # compute output
        output = model(x, seq_lengths)
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, x.size(0))
        top1.update(prec1[0][0], x.size(0))
    #print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def main():
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
        embed = torch.randn([len(d_word_index), args.embedding_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        assert embed.size()[1] == args.embedding_dim
    if not os.path.exists('./result'):
        os.mkdir('./result')

    train_loader = TextClassDataLoader(train_path, d_word_index, batch_size=args.batch)
    val_loader = TextClassDataLoader(test_path, d_word_index, batch_size=args.batch)
    vocab_size = len(d_word_index)
    model = RNN(
        vocab_size=vocab_size,
        embed_size=args.embedding_dim,
        num_output=args.classes,
        rnn_model=args.rnn,
        use_last=(not args.mean_seq),
        hidden_size=args.hidden_size,
        embedding_tensor=embed,
        num_layers=args.layers,
        batch_first=True
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        model.cuda()
        criterion = criterion.cuda()

    # training and testing
    t1 = datetime.datetime.now()
    acc_curve = []
    for epoch in range(1, args.epochs + 1):
        st = datetime.datetime.now()
        adjust_learning_rate(args.lr, optimizer, epoch)
        train_model(train_loader, model, criterion, optimizer)
        res = test_model(val_loader, model, criterion)
        ed = datetime.datetime.now()
        print(epoch, ' epoch cost time', ed - st, 'accuracy is', res.item())
        acc_curve.append(res.item())
    plt.plot(acc_curve)
    plt.show()
    t2 = datetime.datetime.now()
    save_name = './result/' + args.experiment_name + '.h5'
    res = {
        'word2index': d_word_index,
        'model': model.state_dict(),
        'acc_curve': acc_curve
    }
    torch.save(res, save_name)
    # save the dictionary and the model
    print('final result is ', res.item(), 'cost time', t2 - t1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch', default=16, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
    parser.add_argument('--embedding_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--classes', default=250, type=int, metavar='N', help='number of output classes')
    parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--embedding_path', type=str, default='d1_author_1', help='model name')
    parser.add_argument('--train_data', type=str, default='./dataset/train.tsv', help='model name')
    parser.add_argument('--test_data', type=str, default='./dataset/test.tsv', help='model name')
    parser.add_argument('--embedding_type', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='worst_case')

    args = parser.parse_args()
    main()
