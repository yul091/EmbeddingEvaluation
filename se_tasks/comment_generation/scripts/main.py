import os
import argparse
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchtext.data.metrics import bleu_score
from torch.nn.utils.rnn import pad_sequence
import datetime

from se_tasks.comment_generation.scripts.options import train_opts, model_opts
from se_tasks.comment_generation.scripts.dataloader import CommentData
from se_tasks.comment_generation.scripts.model import Seq2seqAttn

MAX_SRC = 200
MAX_TGT = 20


def adjust_learning_rate(optimizer, epoch, ori_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = ori_lr * (0.5 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def my_collate(batch):
    def add_eos(corpus, sor_len):
        for i, data in enumerate(corpus):
            corpus[i][sor_len[i] - 1] = 3
        return corpus

    src, tgt = list(zip(*batch))
    del_index = [i for i in range(len(src)) if len(src[i]) > MAX_SRC or len(tgt[i]) > MAX_TGT]
    src = [torch.tensor(t) for i, t in enumerate(src) if i not in del_index]
    tgt = [torch.tensor(t) for i, t in enumerate(tgt) if i not in del_index]
    src_len = [len(i) for i in src]
    tgt_len = [len(i) for i in tgt]
    src = pad_sequence(src, batch_first=True, padding_value=1)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=1)
    src = add_eos(src, src_len)
    tgt = add_eos(tgt, tgt_len)
    src = src.transpose(1, 0)
    tgt = tgt.transpose(1, 0)
    return src, tgt, src_len, tgt_len


def perpare_embed(embed_type, tk_path, embed_path, embed_dim, res_dir):
    d_word_index, embed_vec = None, None
    with open(tk_path, 'rb') as f:
        tk2index, word2index = pickle.load(f)

    if embed_type == 0:
        d_word_index, embed = torch.load(embed_path)
        if type(embed) is np.ndarray:
            embed_vec = torch.tensor(embed, dtype=torch.float)
        print('load existing embedding vectors, name is ', embed_path)
    elif args.embed_type == 1:
        print('create new embedding vectors, training from scratch')
    elif args.embed_type == 2:
        embed_vec = torch.randn([len(tk2index), embed_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')

    if embed_vec is not None:
        if type(embed_vec) is np.ndarray:
            embed_vec = torch.tensor(embed_vec, dtype=torch.float)
        assert embed_vec.size(1) == embed_dim
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return embed_vec, tk2index, word2index, d_word_index

def id2word(pred_tensor):
    def find_index(str_list):
        for i, v in enumerate(str_list):
            if v == '3':
                return i
        return -1
    res = []
    ori = pred_tensor.detach().cpu().numpy()
    for data in ori:
        translate = [str(v) for v in data]
        eos = find_index(translate)
        res.append(translate[:eos])
    return res


def test_model(test_loader, model, device):
    reference = []
    pred_list = []
    model.eval()
    for data in test_loader:
        src, tgt, src_len, tgt_len = data
        preds = model(src, None, device)
        preds = preds.max(2)[1].transpose(1, 0)
        ref = tgt.transpose(1, 0)
        ref = id2word(ref)
        reference += [[r] for r in ref]
        pred_list += id2word(preds)
    score = bleu_score(pred_list, reference)
    return score


def train_model(train_loader, optimizer, model, device, criterion, scheduler):
    loss_sum = 0
    model.train()
    for data in train_loader:
        src, tgt, src_len, tgt_len = data
        optimizer.zero_grad()
        outs = model(src, tgt, device, tf_ratio=0.5)
        loss = criterion(
            outs.view(-1, outs.size(2)),
            tgt.reshape(-1).to(device)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        loss_sum += loss
    return loss_sum / len(train_loader)


def main(arg_set):
    train_path, test_path = args.train_data, args.test_data
    device = torch.device(args.device)
    embed_type = arg_set.embed_type
    tk_path = arg_set.tk_path
    embed_path = arg_set.embed_path
    embed_dim = arg_set.embed_dim
    res_dir = arg_set.res_dir
    max_size = arg_set.max_size
    layer_num = arg_set.layer_num
    batch = arg_set.batch
    lr = arg_set.lr
    epoch = arg_set.epochs

    tmp_res = perpare_embed(embed_type, tk_path, embed_path, embed_dim, res_dir)
    embed_vec, tk2index, word2index, d_word2index = tmp_res

    train_data = CommentData(train_path, tk2index, word2index, d_word2index, embed_vec, max_size)
    train_loader = DataLoader(train_data, batch_size=batch, collate_fn=my_collate)
    test_data = CommentData(test_path, tk2index, word2index, d_word2index, embed_vec, max_size)
    test_loader = DataLoader(test_data, batch_size=batch, collate_fn=my_collate)
    print('load data: training size', len(train_data), 'testing size', len(test_data))

    tk_size, word_size = len(tk2index), len(word2index)
    model = Seq2seqAttn(tk_size, word_size, device, embed_dim,
                        embed_dim, embed_vec, 'general',
                        layer_num, True, dropout=0.5, tied=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    time_cost = None
    for i in range(1, epoch + 1):
        adjust_learning_rate(optimizer, i, ori_lr=lr)
        st_time = datetime.datetime.now()
        loss = train_model(train_loader, optimizer, model, device, criterion, scheduler)
        ed_time = datetime.datetime.now()
        if time_cost is None:
            time_cost = ed_time - st_time
        else:
            time_cost += (ed_time - st_time)
        score = test_model(test_loader, model, device)
        print(i, 'loss:', loss.item(), 'score', score, 'cost time', ed_time - st_time)
    print('average cost time', time_cost / epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    train_opts(parser)
    model_opts(parser)
    args = parser.parse_args()
    main(args)
