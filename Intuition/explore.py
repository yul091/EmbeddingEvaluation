import numpy as np


from vector_evaluation.experiment import construct_instance


def main():
    m = construct_instance(100, 1, None)
    m.update(2000)
    sim_mat = np.zeros([1000, m.embed_num])
    for i in range(1000):
        tk_i, tk_j, tk_k = np.random.randint(0, m.token_num-1, 3)
        for j in range(m.embed_num):
            vec_1 = m.vec_list[j][tk_j] #- m.vec_list[j][tk_i]
            vec_2 = m.vec_list[j][tk_k] #- m.vec_list[j][tk_i]
            vec_1 = vec_1.reshape([-1])
            vec_2 = vec_2.reshape([1, -1])
            sim_mat[i][j] = m.w2v_list[j].wv.cosine_similarities(vec_1, vec_2)
    print()
    print('success')


if __name__ == '__main__':
    main()