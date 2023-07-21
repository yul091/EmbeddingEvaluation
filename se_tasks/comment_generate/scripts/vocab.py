from collections import Counter

import pandas as pd
#import torchwordemb
import torch


def _tokenize(text):
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [ x.lower() for x in text.split() ]


class VocabBuilder(object):
    '''
    Read file and create word_to_index dictionary.
    This can truncate low-frequency words with min_sample option.
    '''
    def __init__(self, path_file=None):
        # word count
        self.df = pd.read_csv(path_file, delimiter='\t')
        self.src_count = VocabBuilder.count_from_file(self.df['body'])
        self.tar_count = VocabBuilder.count_from_file(self.df['label'])
        self.src_word_to_index = {}
        self.tar_word_to_index = {}

    @staticmethod
    def count_from_file(data, tokenizer=_tokenize):
        """
        count word frequencies in a file.
        Args:
            path_file:
        Returns:
            dict: {word_n :count_n, ...}
        """

        data = data.apply(tokenizer)
        word_count = Counter([tkn for sample in data.values.tolist() for tkn in sample])
        print('Original Vocab size:{}'.format(len(word_count)))
        return word_count

    def get_word_index(self, min_sample=2, padding_marker='____PAD____', unknown_marker='____UNKNOW____',):
        """
        create word-to-index mapping. Padding and unknown are added to last 2 indices.

        Args:
            min_sample: for Truncation
            padding_marker: padding mark
            unknown_marker: unknown-word mark

        Returns:
            dict: {word_n: index_n, ... }

        """
        # truncate low fq word
        _src_word_count = filter(lambda x:  min_sample<=x[1], self.src_count.items())
        _tar_word_count = filter(lambda x:  min_sample<=x[1], self.tar_count.items())
        src_tokens, _ = zip(*_src_word_count)
        tar_tokens, _ = zip(*_tar_word_count)

        # inset padding and unknown
        self.src_word_to_index = {
            tkn: i for i, tkn in enumerate([unknown_marker, padding_marker] + sorted(src_tokens))
        }
        print('Turncated vocab size:{} (removed:{})'.format(
            len(self.src_word_to_index),
            len(self.src_count) - len(self.src_word_to_index))
        )

        self.tar_word_to_index = {
            tkn: i for i, tkn in enumerate([unknown_marker, padding_marker] + sorted(tar_tokens))
        }
        print('Turncated vocab size:{} (removed:{})'.format(
            len(self.tar_word_to_index),
            len(self.tar_count) - len(self.tar_word_to_index))
        )

        return self.src_word_to_index, None, self.tar_word_to_index, None





