import numpy as np
import torch
from gensim.models import word2vec
import datetime
import random
import warnings
import argparse
import pickle
import os
from scipy import stats

import multiprocessing
from multiprocessing import Manager
warnings.filterwarnings('ignore')

from utils import set_random_seed, BASEDICT
from Intuition.investigate_generability import constructModel, dict2list
from embedding_algorithms.word2vec import Word2VecEmbedding


def vocab_frequency(data_dir):
    sentence = word2vec.PathLineSentences(data_dir)
    model = word2vec.Word2Vec()
    model.build_vocab(sentences=sentence)
    torch.save(model.wv, 'wv_fre.fcy')


class Metric:
    def __init__(self):
        self.res = {
            'acc': 0.0,
            'map': 0.0,
            'ndcg': 0.0,
            'mrr': 0.0,
            'recall_3': 0.0,
            'prec_3': 0.0
        }
        self.random = {
            'acc': 0.0,
            'map': 0.0,
            'ndcg': 0.0,
            'mrr': 0.0,
            'recall_3': 0.0,
            'prec_3': 0.0
        }

    @staticmethod
    def MAP(ranked_list, ground_truth):
        hits = 0
        sum_precs = 0
        for n in range(len(ranked_list)):
            if ranked_list[n] in ground_truth:
                hits += 1
                sum_precs += hits / (n + 1.0)
        if hits > 0:
            return sum_precs / len(ground_truth)
        else:
            return 0

    @staticmethod
    def MRR(ranked_list: list, ground_truth):
        return 1 / (ranked_list.index(ground_truth[0]) + 1)

    @staticmethod
    def NDCG(rank_list, ground_truth):
        def getDCG(scores):
            return np.sum(
                np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                dtype=np.float32)
        relevance = np.ones_like(ground_truth)
        it2rel = {it: r for it, r in zip(ground_truth, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
        idcg = getDCG(np.sort(relevance)[::-1])
        dcg = getDCG(rank_scores)
        if dcg == 0.0:
            return 0.0
        ndcg = dcg / idcg
        return ndcg

    def update(self, preds, truth):
        self.res['acc'] += (preds[0] == truth[0])
        self.res['recall_3'] += (truth[0] in preds[:3])
        self.res['prec_3'] += (preds[0] in truth[:3])
        self.res['map'] += self.MAP(preds[:3], truth[:3])
        self.res['mrr'] += self.MRR(preds, truth)
        self.res['ndcg'] += self.NDCG(preds[:3], truth[:3])

        tmp = truth.copy()
        random.shuffle(tmp)
        self.random['acc'] += (tmp[0] == truth[0])
        self.random['recall_3'] += (truth[0] in tmp[:3])
        self.random['prec_3'] += (tmp[0] in truth[:3])
        self.random['map'] += self.MAP(tmp[:3], truth[:3])
        self.random['mrr'] += self.MRR(tmp, truth)
        self.random['ndcg'] += self.NDCG(tmp[:3], truth[:3])

    def produce_final(self, exp_num):
        for k in self.res:
            self.res[k] = self.res[k] / exp_num
        for k in self.random:
            self.random[k] = self.random[k] / exp_num
        return self

    def print(self):
        print('Approach:')
        print(self.res)
        print('Random:')
        print(self.random)


class GroundTruth:
    res = [
        [69.48, 59.04, 59.04, 62.65, 59.44, 45.78, 47.38, 59.84, 18.48, 49.40],
        [25, 38, 32, 38, 21, 25, 20, 28, 37, 21],
        [53.68, 50.16, 61.9, 35.27, 62.03, 58.18, 5.07, 58.51, 56.32, 0.84],
        [27.46, 34.15, 29.8, 32.22, 29.3, 18.99, 30.65, 23.51, 19.07, 29.61],
        [63.85, 67.64, 70.03, 67.88, 69.27, 63.85, 66.3, 65.45, 63.85, 51.36]
    ]
    task = ['authorship', 'summary', 'search', 'completion', 'clone']
    embed = [
        'random',
        'Word2VecEmbedding0.vec',
        'Word2VecEmbedding1.vec',
        'FastEmbeddingcbow.vec',
        'FastEmbeddingskipgram.vec',
        'Doc2VecEmbedding0.vec',
        'Doc2VecEmbedding1.vec',
        'GloVeEmbeddingNone.vec',
        'ori_code2seq.vec',
        'code2vec.vec'
    ]
    task_num = 5
    embed_num = 10

    def get_tk_frec(self):
        mv = torch.load('./wv_fre.fcy')
        vocab = mv.vocab
        frec = {}
        for k in self.word2index:
            new_k = ''.join(k.split('|')).lower()
            if new_k in vocab:
                frec[self.word2index[k]] = vocab[new_k].count
        sort_frec = sorted(frec.items(), key=lambda item: item[1])
        return frec, sort_frec

    def __init__(self, vec_dir, dim, thresh, top_fre):
        self.vec_dir = vec_dir
        self.res = [np.array(i).reshape([1, -1]) for i in self.res]
        self.truth = np.concatenate(self.res, axis=0)
        self.norm_score = self._normalize_truth(norm_type=0)
        self.thresh = thresh
        self.word2index, _ = torch.load(vec_dir + 'ori_code2seq.vec')
        self.index2word = dict2list(self.word2index)
        self.top_fre = top_fre
        if top_fre is not None:
            self.frequency, self.sorted_fre = self.get_tk_frec()
            self.sorted_fre = self.sorted_fre[-top_fre:]

        self.w2v_list = []
        self.vec_list = []
        self.src_index, self.src_tk = None, None
        self.token_num, self.vec_dim = len(self.word2index), dim
        for file_name in self.embed:
            if file_name == 'random':
                vec = np.random.randn(self.token_num, self.vec_dim)
            else:
                _, vec = torch.load(vec_dir + file_name)
            self.vec_list.append(vec)
            m = constructModel(vec, self.word2index, self.index2word)
            self.w2v_list.append(m)

    def get_simalarity_index(self, sample_num):
        #src_i = np.random.randint(0, len(self.sorted_fre), sample_num)
        #src_i = [self.sorted_fre[i][0] for i in src_i]

        src_tk = [self.index2word[i[0]] for i in self.sorted_fre]
        sample_index = np.random.choice(
            range(0, len(src_tk) - 1), size=sample_num, replace=False)
        src_tk = list(np.array(src_tk)[sample_index])

        res = []
        for m_id, w2v_model in enumerate(self.w2v_list):
            model_sim = []
            st = datetime.datetime.now()
            for _, s_tk in enumerate(src_tk):
                (t, s) = w2v_model.wv.most_similar(s_tk, topn=1)[0]
                # if self.word2index[t] not in possible_index:
                #     possible_index.append(self.word2index[t])
                model_sim.append((t, s))
            ed = datetime.datetime.now()
            res.append(model_sim)
            print(ed - st)
        return src_tk, res

    def update(self, sample_num):
        src_tk, offline_res = self.get_simalarity_index(sample_num)
        offline_res = offline_res
        sim_mat = np.zeros([sample_num, len(self.w2v_list)])
        sim_token = []
        for i in range(sample_num):
            tmp = []
            for j, _ in enumerate(offline_res):
                sim_mat[i, j] = offline_res[j][i][1]
                tmp.append(offline_res[j][i][0])
            sim_token.append(tmp)
        kde_list = []
        for i in range(len(self.w2v_list)):
            gkde = stats.gaussian_kde(sim_mat[:, i])
            kde_list.append(gkde)

        mean_val = np.mean(sim_mat, axis=0)
        mean_val = np.tile(mean_val, (sample_num, 1))
        std_val = np.std(sim_mat, axis=0)
        std_val = np.tile(std_val, (sample_num, 1))
        self.src_tk = src_tk
        self.sim_mat = sim_mat
        self.sim_token = sim_token
        self.mean_val = mean_val
        self.std_val = std_val
        self.kde_list = kde_list

    def _normalize_truth(self, norm_type):
        if norm_type == 0:
            max_acc, min_acc = np.max(self.truth, axis=1), np.min(self.truth, axis= 1)
            max_acc, min_acc = max_acc.reshape([-1, 1]), min_acc.reshape([-1, 1])
            norm_truth = (self.truth - min_acc) / (max_acc - min_acc + 1e-8)
        elif norm_type == 1:
            mean, std = np.mean(self.truth, axis=1), np.std(self.truth, axis=1)
            norm_truth = (self.truth - mean) / std
        else:
            raise ValueError
        score = np.average(norm_truth, axis=0)
        return score

    def loacte_correct(self, base, candidate, sample_num):
        sel_index = np.random.choice(
            range(0, len(self.src_tk)), size=sample_num, replace=False)
        src_tk = [self.src_tk[i] for i in sel_index]

        src_index = [self.word2index[tk] for tk in src_tk]
        tgt_tk = [self.sim_token[i][base] for i in sel_index]
        score = np.zeros([sample_num, 2 + len(candidate)])
        score[:, 0] = self.sim_mat[sel_index, base]
        score[:, 1] = self.sim_mat[sel_index, 0]

        for j, c in enumerate(candidate):
            tgt_index = [self.word2index[tk] for tk in tgt_tk]
            src_vec = self.vec_list[j][src_index]
            tgt_vec = self.vec_list[j][tgt_index]
            for iii, v in enumerate(src_vec):
                u = tgt_vec[iii:iii + 1]
                score[iii, j + 2] = self.w2v_list[j].wv.cosine_similarities(v, u)
        res = self.detector(score, base, candidate)
        return res, src_tk, tgt_tk

    def calculate_score(self, base, candidate, sample_num):
        sel_index = np.random.choice(
             range(0, len(self.src_tk)), size=sample_num, replace=False)
        src_tk = [self.src_tk[i] for i in sel_index]

        src_index = [self.word2index[tk] for tk in src_tk]
        tgt_tk = [self.sim_token[i][base] for i in sel_index]
        score = np.zeros([sample_num, 2 + len(candidate)])
        score[:, 0] = self.sim_mat[sel_index, base]
        score[:, 1] = self.sim_mat[sel_index, 0]

        for j, c in enumerate(candidate):
            tgt_index = [self.word2index[tk] for tk in tgt_tk]
            src_vec = self.vec_list[j][src_index]
            tgt_vec = self.vec_list[j][tgt_index]
            for iii, v in enumerate(src_vec):
                u = tgt_vec[iii:iii + 1]
                score[iii, j + 2] = self.w2v_list[j].wv.cosine_similarities(v, u)
        res = self.detector(score, base, candidate)
        return np.sum(res) / sample_num

    def detector(self, sim_mat, base, candidate):
        base_score = sim_mat[:, 0]
        rnd_score = sim_mat[:, 1]
        cand_score = sim_mat[:, 2:]

        like_hood = np.zeros_like(cand_score)
        rand_hood = np.zeros_like(cand_score)
        sampel_num, candiate_num = cand_score.shape

        for i, c in enumerate(candidate):
            like_hood[:, i] = self.kde_list[c].evaluate(cand_score[:, i])
            rand_hood[:, i] = self.kde_list[0].evaluate(cand_score[:, i])
        pred = (like_hood / (like_hood + rand_hood + 1e-9) > 0.5)
        pred = np.sum(pred, axis=1)

        return pred > (candiate_num/2)


def experiment(instance, candidate_num, exp_num, sample_num, return_dict):
    metric = Metric()
    st_time = datetime.datetime.now()
    for _ in range(exp_num):
        candidate = np.random.choice(
            range(0, instance.embed_num), size=candidate_num, replace=False)
        #candidate = np.arange(1, instance.embed_num)
        candidate_score = np.zeros([candidate_num])
        truth_score = instance.norm_score[candidate]
        for j, base in enumerate(candidate):
            index = [i for i in range(candidate_num) if i != j]
            baseline = candidate[index]
            s = instance.calculate_score(base, baseline, sample_num)
            candidate_score[j] = s
        truth_sort = list(np.argsort(truth_score * -1))
        pred_sort = list(np.argsort(candidate_score * -1))
        metric.update(pred_sort, truth_sort)
    ed_time = datetime.datetime.now()
    print(candidate_num, 'cost time:', ed_time - st_time)
    time_cost = (ed_time - st_time) / (exp_num * candidate_num)
    metric.produce_final(exp_num)
    return_dict[candidate_num] = [metric, time_cost, candidate_num]


def construct_instance(dim, thresh, top_fre):
    vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
    m = GroundTruth(vec_dir, dim=dim, thresh=thresh, top_fre=top_fre)
    return m


def multpile_precess(m, exp_num, sample_num):
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for candidate_num in range(5, 11):
        p = multiprocessing.Process(
            target=experiment,
            args=(m, candidate_num, exp_num, sample_num, return_dict)
        )
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    with open('./res/' + str(exp_num) + '_' + str(sample_num) + '_exp.res', 'wb') as f:
        pickle.dump(return_dict.values(), f)


def single_precess(m, exp_num, sample_num):
    return_dict = dict()
    for candidate_num in range(5, 11):
        experiment(m, candidate_num, exp_num, sample_num, return_dict)
        with open('./res/' + str(exp_num) + '_' + str(sample_num) + '_exp.res', 'wb') as f:
            pickle.dump(return_dict, f)


def main():
    vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
    m = GroundTruth(vec_dir, dim=100, thresh=1, top_fre=30000)
    m.update(20000)
    exp_num = args.exp_num
    sample_num = args.sample_num

    use_multiple = True
    if use_multiple:
        multpile_precess(m, exp_num, sample_num)
    else:
        single_precess(m, exp_num, sample_num)
    print('successful')


if __name__ == '__main__':
    if not os.path.isdir('./res'):
        os.mkdir('./res')

    parser = argparse.ArgumentParser('')
    parser.add_argument('--exp_num', default=500, type=int)
    parser.add_argument('--sample_num', default=1500, type=int)
    args = parser.parse_args()
    set_random_seed(100)
    st_time = datetime.datetime.now()
    main()
    ed_time = datetime.datetime.now()
    print(ed_time - st_time)
