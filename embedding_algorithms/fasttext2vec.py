import fasttext
import numpy as np
from gensim.models import word2vec
import os
from utils import BASEDICT
from embedding_algorithms import BasicEmbedding


class FastEmbedding(BasicEmbedding):
    def __init__(self, file_name, dataset, vocab, vec_dim, epoch):
        super(FastEmbedding, self).__init__(file_name, dataset, vocab, vec_dim, epoch)
        file_list = word2vec.PathLineSentences(self.file_name).input_files
        res = []
        for file_name in file_list:
            with open(file_name, 'r') as f:
                res.append(f.read())
        if not os.path.isdir('../tmp_res'):
            os.mkdir('../tmp_res')
        with open('../tmp_res/tmp_file', 'w') as f:
            f.writelines(res)

    def generate_embedding(self, model_type):
        classifier = fasttext.train_unsupervised(
            input='../tmp_res/tmp_file',
            model=model_type,
            dim=self.vec_dim,
            epoch=self.epoch,
            minCount=1,
            thread=10
        )
        return self.get_res(classifier)

    def get_res(self, classifier):
        words = classifier.words
        vec = np.random.rand(len(words) + len(BASEDICT), self.vec_dim)
        word2index = BASEDICT.copy()
        for i, word in enumerate(words):
            word2index[word] = i + len(BASEDICT)
            vec[i + len(BASEDICT)] = classifier.get_word_vector(word)
        return word2index, vec
