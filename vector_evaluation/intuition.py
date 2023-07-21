import numpy as np
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy import stats


from vector_evaluation.experiment import GroundTruth

vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
m = GroundTruth(vec_dir, dim=100, thresh=1, top_fre=50000)
sample_num = 10000
m.update(sample_num)

sim_mat, sim_token = m.sim_mat, m.sim_token


# mean_val = np.mean(sim_mat, axis=0)
# mean_val = np.tile(mean_val, (sample_num, 1))
# std_val = np.std(sim_mat, axis=0)
# std_val = np.tile(std_val, (sample_num, 1))

kde_list = []
for i in range(len(m.w2v_list)):
    gkde = stats.gaussian_kde(sim_mat[:, i])
    kde_list.append(gkde)


# result = []
# for i, w2v_model in enumerate(m.w2v_list):
#     score = np.zeros([sample_num, len(m.w2v_list)])
#     tgt_tk = [tk_list[i] for tk_list in sim_token]
#     for j, _ in enumerate(m.w2v_list):
#         src_index = [m.word2index[tk] for tk in m.src_tk]
#         tgt_index = [m.word2index[tk] for tk in tgt_tk]
#         src_vec = m.vec_list[j][src_index]
#         tgt_vec = m.vec_list[j][tgt_index]
#         for iii, v in enumerate(src_vec):
#             u = tgt_vec[iii:iii+1]
#             score[iii, j] = w2v_model.wv.cosine_similarities(v, u)
#     result.append(score)
#
# thresh = 5
#
# for i in range(1, len(m.w2v_list)):
#     score = result[i]
#     like_hood = np.zeros_like(score)
#     rand_hood = np.zeros_like(score)
#     rand_kde = kde_list[0]
#     for j, kde_model in enumerate(kde_list):
#         like_hood[:, j] = kde_model.evaluate(score[:, j])
#         rand_hood[:, j] = rand_kde.evaluate(score[:, j])
#     pred = np.sum(like_hood / (like_hood + rand_hood + 1e-9) > 0.5, axis=1)
#     print(np.sum(pred > thresh) / sample_num)


x = np.arange(0, 1.1, 0.01)

res = np.zeros([len(x), len(m.w2v_list) + 1])
res[:, 0] = x.reshape([-1])
for i in range(len(m.w2v_list)):
    y = sim_mat[:, i].reshape([-1])
    gkde = stats.gaussian_kde(y)
    kdepdf = gkde.evaluate(x)
    res[:, i + 1] = kdepdf
    plt.plot(x, kdepdf)
plt.show()
print()
np.savetxt('./res/gaussian.csv', res, delimiter=',')


