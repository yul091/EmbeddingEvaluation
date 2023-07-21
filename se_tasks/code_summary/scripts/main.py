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
from torch.utils.data import DataLoader


from se_tasks.code_summary.scripts.CodeLoader import CodeLoader
from se_tasks.code_summary.scripts.Code2VecModule import Code2Vec


def my_collate(batch):
    x, y = zip(*batch)
    sts, paths, eds = [], [], []
    for data in x:
        st, path, ed = zip(*data)
        sts.append(torch.tensor(st, dtype=torch.int))
        paths.append(torch.tensor(path, dtype=torch.int))
        eds.append(torch.tensor(ed, dtype=torch.int))

    length = [len(i) for i in sts]
    sts = rnn_utils.pad_sequence(sts, batch_first=True, padding_value=1).long()
    eds = rnn_utils.pad_sequence(eds, batch_first=True, padding_value=1).long()
    paths = rnn_utils.pad_sequence(paths, batch_first=True, padding_value=1).long()
    return (sts, paths, eds), y, length


def dict2list(tk2index):
    res = {}
    for tk in tk2index:
        res[tk2index[tk]] = tk
    return res


def new_acc(pred, y, index2func):
    pred = pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    tp, fp, fn = 0, 0, 0
    acc = np.sum(pred == y)
    for i, pred_i in enumerate(pred):
        pred_i = set(index2func[pred_i].split('|'))
        y_i = set(index2func[y[i]].split('|'))
        tp += len(pred_i & y_i)
        fp += len(pred_i - y_i)
        fn = len(y_i - pred_i)
    return acc, tp, fp, fn


def perpare_train(tk_path, embed_type, vec_path, embed_dim, out_dir):
    tk2num = None
    with open(tk_path, 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
        embed = None
    if embed_type == 0:
        tk2num, embed = torch.load(vec_path)
        print('load existing embedding vectors, name is ', vec_path)
    elif embed_type == 1:
        tk2num = token2index
        print('create new embedding vectors, training from scratch')
    elif embed_type == 2:
        tk2num = token2index
        embed = torch.randn([len(token2index), embed_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float)
        assert embed.size()[1] == embed_dim
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return token2index, path2index, func2index, embed, tk2num


def train_model(
    tk_path, train_path, test_path, embed_dim, embed_type,
    vec_path, experiment_name, train_batch, epochs, lr,
    weight_decay, max_size, out_dir, device_id):
    token2index, path2index, func2index, embed, tk2num =\
        perpare_train(tk_path, embed_type, vec_path, embed_dim, out_dir)
    nodes_dim, paths_dim, output_dim = len(tk2num), len(path2index), len(func2index)

    model = Code2Vec(nodes_dim, paths_dim, embed_dim, output_dim, embed)
    criterian = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    st_time = datetime.datetime.now()
    train_dataset = CodeLoader(train_path, max_size, token2index, tk2num)
    train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=my_collate)
    ed_time = datetime.datetime.now()
    print('train dataset size is ', len(train_dataset), 'cost time', ed_time - st_time)

    # device = torch.device(int(device_id) if torch.cuda.is_available() else "cpu")
    # print(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    index2func = dict2list(func2index)
    time_cost = None
    for epoch in range(1, epochs + 1):
        acc, tp, fp, fn = 0, 0, 0, 0
        st_time = datetime.datetime.now()
        for i, ((sts, paths, eds), y, length) in tqdm(enumerate(train_loader)):
            sts = sts.to(device)
            paths = paths.to(device)
            eds = eds.to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            pred_y = model(sts, paths, eds, length, device)
            loss = criterian(pred_y, y)
            loss.backward()
            optimizer.step()
            pos, pred_y = torch.max(pred_y, 1)
            acc_add, tp_add, fp_add, fn_add = new_acc(pred_y, y, index2func)
            tp += tp_add
            fp += fp_add
            fn += fn_add
            acc += acc_add
        acc = acc / len(train_dataset)
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = prec * recall * 2 / (prec + recall + 1e-8) 
        ed_time = datetime.datetime.now()
        print(
            'epoch', epoch, 
            'acc:', acc,
            'cost time', ed_time - st_time,
            'p', prec,
            'r', recall,
            'new_f1', f1
        )
        if time_cost is None:
            time_cost = ed_time - st_time
        else:
            time_cost += (ed_time - st_time)
    print('cost time', time_cost / args.epochs)
    return model, token2index


def main(args_set):
    tk_path = args_set.tk_path
    train_path = args_set.train_data
    test_path = args_set.test_data
    embed_dim = args_set.embed_dim
    embed_type = args_set.embed_type
    vec_path = args_set.embed_path
    experiment_name = args_set.experiment_name
    train_batch = args_set.batch
    epochs = args_set.epochs
    lr = args_set.lr
    weight_decay=args_set.weight_decay
    train_model(
        tk_path, train_path, test_path, embed_dim,
        embed_type, vec_path, experiment_name, train_batch,
        epochs, lr, weight_decay, max_size=args_set.max_size,
        out_dir=args_set.res_dir, device_id=args_set.device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--classes', default=250, type=int, metavar='N', help='number of output classes')
    parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')

    parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--device', default=6, help='gpu id')
    parser.add_argument('--embed_path', type=str, default='../../../vec/100_2/Doc2VecEmbedding0.vec')
    parser.add_argument('--train_data', type=str, default='../../../dataset/java-small-preprocess/train.pkl')
    parser.add_argument('--test_data', type=str, default='../../../dataset/java-small-preprocess/val.pkl')
    parser.add_argument('--tk_path', type=str, default='../../../dataset/java-small-preprocess/tk.pkl')
    parser.add_argument('--embed_type', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code2vec')
    parser.add_argument('--res_dir', type=str, default='../result')
    parser.add_argument('--max_size', type=int, default=30000)

    args = parser.parse_args()
    set_random_seed(10)
    main(args)
