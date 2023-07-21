import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


class OutlierDetection:
    def __init__(self, p_table):
        self.p_table = p_table

    def pauta_criterion(self):
        def get_zscore(p_val):
            z_score = np.zeros_like(p_val)
            for i, val in enumerate(p_val):
                tmp = np.copy(p_val)
                new_p = np.delete(tmp, i)
                mean_val, std_val = np.mean(new_p), np.std(new_p)
                z_score[i] = abs((val - mean_val) / std_val)
            return z_score
        res = []
        for p_val in self.p_table:
            z_score = get_zscore(p_val)
            res.append(abs(z_score) > 2)
        return res

    # def dbscan_criterion(self):
    #     dbscan = DBSCAN(
    #         eps=500, min_samples=4, metric=geodistance).fit(list(group['lon_lat']))
    #     # 对于DBSCAN来说，两个最重要的参数就是eps，和min_samples。当然这两个值不是随便定义的，这个在下文再说

    def isolation_criterion(self):
        res = []
        for i, val in enumerate(self.p_table):
            print(i)
            clf = IsolationForest(max_samples=6, random_state=1, contamination="auto")
            clf.fit(np.array(val).reshape([-1, 1]))
            preds = clf.predict(np.array(val).reshape([-1, 1]))
            res.append(preds == 1)
        return res


class SemanticCosine:
    def __init__(self, vec_list, index2word, word2index, sampling_num=5000):
        self.vec_list = vec_list
        self.index2word = index2word
        self.word2index = word2index
        self.vec_num = len(self.vec_list)
        self.sampling_num = sampling_num
        self.vocab_size = len(self.vec_list[0])

    def calculate_score(self):
        p_table = []
        common_tk = self.get_common_tk()
        for i in range(self.sampling_num):
            st, ed, orig = np.random.randint(0, len(common_tk), 3)
            st, ed, orig = common_tk[st], common_tk[ed], common_tk[orig]
            p_list = []
            for j in range(self.vec_num):
                st_index = self.word2index[j][st]
                ed_index = self.word2index[j][ed]
                orig_index = self.word2index[j][orig]
                off_1 = self.vec_list[j][st_index] - self.vec_list[j][orig_index]
                off_2 = self.vec_list[j][ed_index] - self.vec_list[j][orig_index]
                p1 = cosine_similarity([off_1, off_2])
                p1 = p1[0][1]
                p_list.append(p1)
            p_table.append(p_list)
        detector = OutlierDetection(p_table)
        res = detector.pauta_criterion()
        #res = detector.isolation_criterion()
        print(1 - np.sum(res, 0, dtype=np.float) / self.sampling_num)

    def is_contain(self, tk):
        for dict_map in self.word2index:
            if tk not in dict_map:
                return False
        return True

    def get_common_tk(self):
        res = []
        for tk in self.word2index[0]:
            if self.is_contain(tk):
                res.append(tk)
        return res

