from vector_evaluation.experiment import GroundTruth
from sklearn.linear_model import LinearRegression
import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity



class DebugVector:
    def __init__(self, debug, base_list, top_num):
        vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
        m = GroundTruth(vec_dir, dim=100, thresh=1, top_fre=50000)
        m.update(top_num)
        self.m = m
        self.debug = debug
        self.base_list = base_list
        self.top_num = top_num
        self.debug_vec = self.m.vec_list[debug]
        self.base_vec = [self.m.vec_list[i] for i in base_list]

        self.debug_index = []

    def find_triangle(self):
        for i in range(self.top_num):
            src_index = self.m.sorted_fre[i][0]
            src_tk = self.m.index2word[src_index]
            top2 = self.m.w2v_list[self.debug].wv.most_similar(src_tk, topn=2)
            tgt_1, tgt_2 = self.m.word2index[top2[0][0]], self.m.word2index[top2[1][0]]
            self.debug_index.append([src_index, tgt_1, tgt_2])

    def index_vec(self, i, embedNum):
        return self.m.vec_list[embedNum][i].reshape([1, -1])

    def locate_fault(self):
        def calculate_hood(s, t1, t2, e,):
            vec_1 = self.index_vec(s, e)
            vec_2 = self.index_vec(t1, e)
            vec_3 = self.index_vec(t2, e)
            vec = np.concatenate([vec_1, vec_2, vec_3], axis=0)
            cosine = cosine_similarity(vec)
            s1, s2, s3 = cosine[0, 1], cosine[0, 2], cosine[1, 2]
            cosine = [s1, s2, s3]
            return self.m.kde_list[e].evaluate(cosine)


        err_list = []
        for tri in self.debug_index:
            src, tgt_1, tgt_2 = tri
            random_hood = calculate_hood(src, tgt_1, tgt_2, 0)
            pred_list = []
            for embed in self.base_list:
                like_hood = calculate_hood(src, tgt_1, tgt_2, embed)
                pred = (like_hood > random_hood)
                pred_list.append(pred)
            pred = np.mean(pred_list, axis=0)
            if pred[0] == 0:
                err_list.append()
            print()


if __name__ == '__main__':
    num = 50000
    a = DebugVector(1, [1, 2], top_num=num)
    src, tgt = 1, 9
    res_1, src_tk_1, tgt_tk_1 = a.m.loacte_correct(tgt, [1, 2, 3, 9], num)
    res_2, src_tk_2, tgt_tk_2 = a.m.loacte_correct(src, [0, 2, 3, 9], num)

    tgt_tk = set()
    for i, v in enumerate(res_1):
        if v == 1:
            tgt_tk.add(src_tk_1[i])
            tgt_tk.add(tgt_tk_1[i])
    src_tk = set()
    for i, v in enumerate(res_2):
        if v == 1:
            src_tk.add(src_tk_2[i])
            src_tk.add(tgt_tk_2[i])
    common_tk = list(tgt_tk & src_tk)
    print('token size', len(common_tk))
    common_index = [a.m.word2index[tk] for tk in common_tk]
    src_mat = a.m.vec_list[src][common_index]
    tgt_mat = a.m.vec_list[tgt][common_index]

    linear_model = LinearRegression()
    linear_model.fit(src_mat, tgt_mat)
    pred_vec = linear_model.predict(a.m.vec_list[src])
    torch.save([a.m.word2index, pred_vec], '../vec/100_2/debug_1.vec')
    print('finish debug-1')

    linear_model = LinearRegression()
    linear_model.fit(tgt_mat, src_mat)
    pred_vec = linear_model.predict(a.m.vec_list[tgt])
    torch.save([a.m.word2index, pred_vec], '../vec/100_2/debug_2.vec')
    print('finish debug-2')