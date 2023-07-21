from gensim.models import word2vec
from embedding_algorithms import BasicEmbedding
from glove import Corpus, Glove 
from utils import BASEDICT
import numpy as np


def trans_vocab(vocab, vectors):
    new_vocab = BASEDICT.copy()
    for tk in vocab:
        new_vocab[tk] = vocab[tk] + len(BASEDICT)
    dim = vectors.shape[1]
    tmp_vec = np.random.rand(len(BASEDICT), dim)
    vec = np.concatenate([tmp_vec, vectors])
    return new_vocab, vec


class GloVeEmbedding(BasicEmbedding):
    def __init__(
        self, file_name, dataset, vocab, vec_dim, learning_rate=0.05, 
        window=10, epoch=1, no_threads=10, verbose=True
    ):
        super(GloVeEmbedding, self).__init__(
            file_name, dataset, vocab, vec_dim, learning_rate
        )
        self.learning_rate = learning_rate
        self.no_threads = no_threads
        self.window = window
        self.epoch = epoch
        self.verbose = verbose

    def generate_embedding(self, model_type):
        sentences = word2vec.PathLineSentences(self.file_name)

        # Training the corpus to generate the co-occurance matrix which is used in GloVe
        corpus = Corpus()  # Creating a corpus object
        corpus.fit(sentences, window=self.window) 

        # Training GloVe model
        glove = Glove(
            no_components=self.vec_dim, 
            learning_rate=self.learning_rate
        ) 
        glove.fit(
            corpus.matrix, epochs=self.epoch, 
            no_threads=self.no_threads, verbose=self.verbose
        )
        glove.add_dictionary(corpus.dictionary)

        return trans_vocab(glove.dictionary, glove.word_vectors)







