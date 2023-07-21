import torch
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

vec_dir = '/glusterfs/dataset/sxc180080/EmbeddingEvaluation/vec/100_2/'


def dict2list(w2i):
    res = {}
    for k in w2i:
        res[w2i[k]] = k
    return res

def list_dict(index2word):
    res = []
    for i in range(len(index2word)):
        res.append(index2word[i])
    return res


class Index:
    def __init__(self, v):
        self.index = v


def constructDict(word2index):
    res = {}
    for k in word2index:
        res[k] = Index(word2index[k])
    return res


def constructModel(vec, word2index, index2word):
    m = Word2Vec(workers=10)
    m.wv.vectors = vec
    m.wv.vocab = constructDict(word2index)
    m.wv.index2word = list_dict(index2word)
    return m




def main():
    token_list = [
        'int', 'long', 'short', 'char', 'send', 'get',
        'array', 'for', 'while', 'if',
        'public', 'void', 'throw'
    ]

    _, vec_1 = torch.load(vec_dir + 'Doc2VecEmbedding1.vec')
    word2index, vec_2 = torch.load(vec_dir + 'Word2VecEmbedding0.vec')
    index2word = dict2list(word2index)

    m_1 = Word2Vec()
    m_1.wv.vectors = vec_1
    m_1.wv.vocab = constructDict(word2index)
    m_1.wv.index2word = list_dict(index2word)

    m_2 = Word2Vec()
    m_2.wv.vectors = vec_2
    m_2.wv.vocab = constructDict(word2index)
    m_2.wv.index2word = list_dict(index2word)

    for key in token_list:
        print(key)
        print(m_1.wv.most_similar(key, topn=3))
        print(m_2.wv.most_similar(key, topn=3))
        print('===============================')


if __name__ == '__main__':

   main()