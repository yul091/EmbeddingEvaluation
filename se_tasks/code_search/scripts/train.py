import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
import argparse

from utils import set_random_seed

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter  # install tensorboardX (pip install tensorboardX) before importing this package

import torch

from se_tasks.code_search.scripts import models, configs, data_loader
from se_tasks.code_search.scripts.modules import get_cosine_schedule_with_warmup
from se_tasks.code_search.scripts.util import similarity, normalize
from se_tasks.code_search.scripts.data_loader import *

try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML, SESSION_NAME
except:
    IS_ON_NSML = False

def dic2list(d):
    res = {}
    for k in d:
        res[d[k]] = k
    return res


def transfer(val, d_1, d_2):
    for i, v in enumerate(val):
        v = d_1[v].lower()
        val[i] = d_2[v] if v in d_2 else d_2['____UNKNOW____']
    return val


def transfer_dataset(dataset, meth_name_list, api_list, tokens_list, d_word_index):
    #dataset.apis = transfer(dataset.apis, api_list, d_word_index)
    dataset.names = transfer(dataset.names, meth_name_list, d_word_index)
    dataset.tokens = transfer(dataset.tokens, tokens_list, d_word_index)
    return dataset


def preprocess_data(train_set, valid_set, d_word_index, data_dir):
    with open(data_dir + 'vocab.methname.pkl', 'rb') as f:
        meth_name_dict = pickle.load(f)
        meth_name_list = dic2list(meth_name_dict)
    with open(data_dir + 'vocab.desc.pkl', 'rb') as f:
        desc_dict = pickle.load(f)
        desc_list = dic2list(desc_dict)
    with open(data_dir + 'vocab.apiseq.pkl', 'rb') as f:
        api_dict = pickle.load(f)
        api_list = dic2list(api_dict)
    with open(data_dir + 'vocab.tokens.pkl', 'rb') as f:
        tokens_dict = pickle.load(f)
        tokens_list = dic2list(tokens_dict)
    train_set = transfer_dataset(train_set, meth_name_list, api_list, tokens_list, d_word_index)
    valid_set = transfer_dataset(valid_set, meth_name_list, api_list, tokens_list, d_word_index)
    return train_set, valid_set


def set_embedding(train_set, valid_set):
    embed = [None, None, None]
    vocab_size = 10001
    if args.embed_type == 0:
        d_word_index, embed = torch.load(args.embed_path)
        embed = [embed, embed, embed]
        train_set, valid_set = preprocess_data(train_set, valid_set, d_word_index, args.data_path)
        vocab_size = len(d_word_index)
        print('load existing embedding vectors, name is ', args.embed_path)
    elif args.embed_type == 1:
        print('create new embedding vectors, training from scratch')
    elif args.embed_type == 2:
        embed = [
            torch.randn([vocab_size, args.embed_dim]).cuda(),
            torch.randn([vocab_size, args.embed_dim]).cuda(),
            torch.randn([vocab_size, args.embed_dim]).cuda(),
        ]
    else:
        raise ValueError('unsupported type')
    if embed[0] is not None:
        if type(embed[0]) is np.ndarray:
            embed = [torch.tensor(e, dtype=torch.float).cuda() for e in embed]
        assert embed[0].size(1) == args.embed_dim
    print('vacab size is ', vocab_size)
    return train_set, valid_set, vocab_size, embed


def bind_nsml(model, **kwargs):
    if type(model) == torch.nn.DataParallel: model = model.module

    def infer(raw_data, **kwargs):
        pass

    def load(path, *args):
        weights = torch.load(path)
        model.load_state_dict(weights)
        logger.info(f'Load checkpoints...!{path}')

    def save(path, *args):
        torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))
        logger.info(f'Save checkpoints...!{path}')

    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)


def train_model(args):
    fh = logging.FileHandler(f"se_tasks/code_search/output/{args.model}/{args.dataset}/logs.txt")
    # create file handler which logs even debug messages
    logger.addHandler(fh)  # add the handlers to the logger
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    tb_writer = SummaryWriter(f"se_tasks/code_search/output/{args.model}/{args.dataset}/logs/{timestamp}") if args.visual else None
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    config = getattr(configs, 'config_' + args.model)()
    if args.automl:
        config.update(vars(args))


    ###############################################################################
    # Load dataset
    ###############################################################################
    data_path = DATASET_PATH + "/train/" if IS_ON_NSML else args.data_path
    train_set = eval(config['dataset_name'])(data_path, config['train_name'], config['name_len'],
                                             config['train_api'], config['api_len'],
                                             config['train_tokens'], config['tokens_len'],
                                             config['train_desc'], config['desc_len'])
    valid_set = eval(config['dataset_name'])(data_path,
                                             config['valid_name'], config['name_len'],
                                             config['valid_api'], config['api_len'],
                                             config['valid_tokens'], config['tokens_len'],
                                             config['valid_desc'], config['desc_len'])

    train_set, valid_set, vocab_size, embed = set_embedding(train_set, valid_set)

    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                              shuffle=True, drop_last=True, num_workers=1)

    ###############################################################################
    # Define Model
    ###############################################################################
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config, vocab_size, args.embed_dim, embed, args.embed_type)  # initialize the model

    def save_model(model, ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))

    if args.reload_from > 0:
        ckpt = f'se_tasks/code_search/output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
        load_model(model, ckpt, device)

    if IS_ON_NSML:
        bind_nsml(model)
    model.to(device)

    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['learning_rate'], eps=config['adam_epsilon']
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'],
        num_training_steps=len(data_loader) * config[
            'nb_epoch'])  # do not foget to modify the number when dataset is changed
    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])

    ###############################################################################
    # Training Process
    ###############################################################################    
    n_iters = len(data_loader)

    time_cost = None
    for epoch in range(int(args.reload_from / n_iters) + 1, config['nb_epoch'] + 1):
        st_time = datetime.now()
        losses = []
        for batch_data in data_loader:
            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch_data]
            loss = model(*batch_gpu)

            if config['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            losses.append(loss.item())
        ed_time = datetime.now()
        if time_cost is None:
            time_cost = (ed_time - st_time)
        else:
            time_cost += (ed_time - st_time)
        if epoch % args.valid_every == 0:
            # logger.info("validating..")
            valid_result = validate(valid_set, model, 10000, 1, config['sim_measure'])
            print(epoch, valid_result)
            # logger.info(valid_result)
            # if tb_writer is not None:
            #     for key, value in valid_result.items():
            #         tb_writer.add_scalar(key, value, itr_global)
            # if IS_ON_NSML:
            #     summary = {"summary": True, "scope": locals(), "step": itr_global}
            #     summary.update(valid_result)
            #     nsml.report(**summary)
    print('cost time', time_cost / (config['nb_epoch'] - int(args.reload_from / n_iters)))


def validate(valid_set, model, pool_size, K, sim_measure):
    """
    simple validation in a code pool. 
    @param: poolsize - size of the code pool, if -1, load the whole test set
    """

    def ACC(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + 1
        return sum / float(len(real))

    def MAP(real, predict):
        sum = 0.0
        for id, val in enumerate(real):
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + (id + 1) / float(index + 1)
        return sum / float(len(real))

    def MRR(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + 1.0 / float(index + 1)
        return sum / float(len(real))

    def NDCG(real, predict):
        dcg = 0.0
        idcg = IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i + 1
                dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcg)

    def IDCG(n):
        idcg = 0
        itemRelevance = 1
        for i in range(n): idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg

    model.eval()
    device = next(model.parameters()).device

    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=10000,
                                              shuffle=True, drop_last=True, num_workers=1)
    accs, mrrs, maps, ndcgs = [], [], [], []
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in data_loader:
        if len(batch) == 10:  # names, name_len, apis, api_len, toks, tok_len, descs, desc_len, bad_descs, bad_desc_len
            code_batch = [tensor.to(device) for tensor in batch[:6]]
            desc_batch = [tensor.to(device) for tensor in batch[6:8]]
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
            if sim_measure == 'cos':
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    for k in range(0, n_processed, pool_size):
        code_pool, desc_pool = code_reprs[k:k + pool_size], desc_reprs[k:k + pool_size]
        for i in range(min(10000, pool_size)):  # for i in range(pool_size):
            desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]
            n_results = K
            if sim_measure == 'cos':
                sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]
            else:
                sims = similarity(code_pool, desc_vec, sim_measure)  # [pool_size]

            negsims = np.negative(sims)
            predict = np.argpartition(negsims, kth=n_results - 1)  # predict=np.argsort(negsims)#
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
    return {'acc': np.mean(accs), 'mrr': np.mean(mrrs), 'map': np.mean(maps), 'ndcg': np.mean(ndcgs)}


def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='example', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=5, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # Model Hyperparameters for automl tuning
    parser.add_argument('--n_hidden', type=int, default=-1,
                        help='number of hidden dimension of code/desc representation')
    parser.add_argument('--lstm_dims', type=int, default=-1)
    parser.add_argument('--margin', type=float, default=-1)
    parser.add_argument('--sim_measure', type=str, default='cos', help='similarity measure for training')

    parser.add_argument('--learning_rate', type=float, help='learning rate')

    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)

    ### Todo
    parser.add_argument('--embed_path', type=str, default='../../../vec/100_2/code2vec.vec')
    parser.add_argument('--embed_type', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--experiment_name', type=str, default='code2vec')
    parser.add_argument('--data_path', type=str, default='../dataset/example/', help='location of the dataset corpus')
    #### Todo
    return parser.parse_args()


if __name__ == '__main__':
    # set_random_seed(42)
    args = parse_args()

    # make output directory if it doesn't already exist
    os.makedirs(f'se_tasks/code_search/output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'se_tasks/code_search/output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)

    torch.backends.cudnn.benchmark = True  # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True  # fix the random seed in cudnn

    train_model(args)
