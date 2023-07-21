import torch
import pandas as pd
import numpy as np


def _tokenize(text):
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [ x.lower() for x in text.split() ]


class CodeComment:
    def __init__(self, code, comment):
        self.body = code
        self.label = comment


class CodeDataset(object):
    def __init__(self, path_file):
        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        df['body'] = df['body'].apply(_tokenize)
        df['label'] = df['label'].apply(_tokenize)
        self.old_samples = df['body'].copy()
        df['body'] = df['body']
        df['label'] = df['label']
        samples = df.values.tolist()
        self.samples = [CodeComment(s[0], s[1]) for s in samples]
        self.samples = self.samples[:5000]
        self.body = df['label']
        self.label = df['body']

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

class TextClassDataLoader(object):
    def __init__(self, path_file, src_word2index, tar_word2index, batch_size=32):
        """

        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.src_word2index = src_word2index
        self.tar_word2index = tar_word2index
        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        df['body'] = df['body'].apply(_tokenize)
        df['label'] = df['label'].apply(_tokenize)
        self.old_samples = df['body'].copy()
        df['body'] = df['body'].apply(
            self.generate_indexifyer(word2index=self.src_word2index)
        )
        df['label'] = df['label'].apply(
            self.generate_indexifyer(word2index=self.tar_word2index)
        )
        self.samples = df.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        print('dataset number is ', self.n_samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self._shuffle_indices()

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[1]))
        return length

    def generate_indexifyer(self, word2index):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in word2index:
                    indices.append(word2index[word])
                else:
                    indices.append(word2index['____UNKNOW____'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1
        comment, code = tuple(zip(*batch))
        # get the length of each seq in your batch
        code_lengths = torch.LongTensor([len(s) for s in code])
        comment_lengths = torch.LongTensor([len(s) for s in comment])
        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        code_tensor = torch.zeros((len(code), code_lengths.max())).long()
        comment_tensor = torch.zeros((len(comment), comment_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(code, code_lengths)):
            code_tensor[idx, :seqlen] = torch.LongTensor(seq)

        for idx, (seq, seqlen) in enumerate(zip(comment, comment_lengths)):
            comment_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        code_lengths, perm_idx = code_lengths.sort(0, descending=True)
        code_tensor = code_tensor[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        comment_tensor = comment_tensor[perm_idx]
        comment_lengths = comment_lengths[perm_idx]
        return code_tensor, comment_tensor, code_lengths, comment_lengths

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))