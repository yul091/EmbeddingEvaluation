import torch
import torch.nn as nn


class Word2vecPredict(nn.Module):
    def __init__(self, d_word_index, token_vec):
        super(Word2vecPredict, self).__init__()
        vocab_size = len(d_word_index)
        if torch.is_tensor(token_vec):
            self.encoder = nn.Embedding(vocab_size, 100, padding_idx=0, _weight=token_vec)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, 100, padding_idx=0)

        self.linear = nn.Linear(100, len(d_word_index))

    def forward(self, x):
        vec = self.encoder(x)
        vec = torch.mean(vec, dim=1)
        pred = self.linear(vec)
        return pred

