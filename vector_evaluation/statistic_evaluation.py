import numpy as np
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from scipy import stats

from vector_evaluation.experiment import GroundTruth



def generate_token():
    vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
    m = GroundTruth(vec_dir, dim=100, thresh=1, top_fre=50000)
    sample_num = 10000
    m.update(sample_num)

    src_tk = m.src_tk
    sim_mat, sim_token = m.sim_mat, m.sim_token

    best_case = [i[1] for i in sim_token]
    print()

    sim_result = []
    rnd_result = []
    for i, src in enumerate(src_tk):
        tgt = best_case[i]
        r_1, r_2 = set(src.split('|')), set(tgt.split('|'))
        common = len(r_1 & r_2)
        p = common / len(r_1)
        r = common / len(r_2)
        f1 = p * r * 2 / (p + r + 1e-9)
        if f1 > 0.6:
            sim_result.append([src, tgt])
            print(src, tgt)
        if f1 == 0:
            rnd_result.append([src, tgt])

    with open('./sim_token_1.pkl', 'wb') as f:
        pickle.dump([sim_result, rnd_result], f)

    print(len(sim_result))
    return m


def get_stastic_data(m):
    with open('./sim_token_1.pkl', 'rb') as f:
        [sim_result, rnd_result] = pickle.load(f)
    sim_result = sim_result[:1000]
    rnd_result = rnd_result[:1000]
    if m is None:
        vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
        m = GroundTruth(vec_dir, dim=100, thresh=1, top_fre=50000)
        m.update(5000)
    correct_score = np.zeros([len(sim_result), len(m.w2v_list)])
    error_score = np.zeros([len(rnd_result), len(m.w2v_list)])

    def compute_sim_score(tk_1, tk_2, index):
        src_vec, tgt_vec = \
            m.vec_list[index][m.word2index[tk_1]], m.vec_list[j][m.word2index[tk_2]]
        s = m.w2v_list[index].wv.cosine_similarities(src_vec.reshape([-1]), tgt_vec.reshape([1, -1]))
        return s

    for i in range(len(sim_result)):
        for j in range(len(m.w2v_list)):
            src_tk, tgt_tk = sim_result[i]
            s = compute_sim_score(src_tk, tgt_tk, j)
            correct_score[i, j] = s

            src_tk, tgt_tk = rnd_result[i]
            s = compute_sim_score(src_tk, tgt_tk, j)
            error_score[i, j] = s

    like_hood = np.zeros_like(correct_score)
    rand_hood = np.zeros_like(error_score)
    for i in range(len(m.w2v_list)):
        like_hood[:, i] = m.kde_list[i].evaluate(correct_score[:, i])
        rand_hood[:, i] = m.kde_list[0].evaluate(correct_score[:, 0])
    correct_pred = (like_hood / (like_hood + rand_hood + 1e-9))

    for i in range(len(m.w2v_list)):
        like_hood[:, i] = m.kde_list[i].evaluate(error_score[:, i])
        rand_hood[:, i] = m.kde_list[0].evaluate(error_score[:, 0])
    error_pred = (like_hood / (like_hood + rand_hood + 1e-9) )

    with open('./res/stastic.res', 'wb') as f:
        pickle.dump([correct_pred, error_pred], f)
        print()


def analysis():
    with open('./sim_token.pkl', 'rb') as f:
        [sim_result, rnd_result] = pickle.load(f)
    # for i in rnd_result:
    #     print(i)
    with open('./res/stastic.res', 'rb') as f:
        [correct_pred, error_pred] = pickle.load(f)
        corr_pred = np.sum(correct_pred > 0.5, axis=1) / 10
        err_pred = np.sum(error_pred > 0.5, axis=1) / 10

    corr_mean, corr_std, corr_med = \
        np.mean(corr_pred), np.std(corr_pred), np.median(corr_pred)
    print('correct:', corr_mean, corr_std, corr_med )
    err_mean, err_std, err_med = \
        np.mean(err_pred), np.std(err_pred), np.median(err_pred)
    print('error:', err_mean, err_std, err_med)
    t, p = stats.levene(corr_pred, err_pred)
    if p > 0.05:
        t, p = stats.ttest_ind(corr_pred, err_pred)
    else:
        t, p = stats.ttest_ind(corr_pred, err_pred, equal_var=False)
    print(t, p)


if __name__ == '__main__':
    #m = generate_token()
    #get_stastic_data(None)
    analysis()