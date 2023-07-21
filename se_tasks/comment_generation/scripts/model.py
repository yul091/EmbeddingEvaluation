# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncRNN(nn.Module):
    def __init__(self, vsz, embed_dim, pad_id, hidden_dim,
                 n_layers, use_birnn, dout, embed_vec, device):
        super(EncRNN, self).__init__()

        if embed_vec is None:
            self.embed = nn.Embedding(vsz, embed_dim, padding_idx=pad_id)
        else:
            self.embed = nn.Embedding(vsz, embed_dim, padding_idx=pad_id, _weight=embed_vec)
            self.embed.weight.requires_grad = False
        self.device = device
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           bidirectional=use_birnn)
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs):
        embs = self.dropout(self.embed(inputs).to(self.device))
        enc_outs, hidden = self.rnn(embs)
        return self.dropout(enc_outs), hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out*enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out*energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class DecRNN(nn.Module):
    def __init__(self, vsz, embed_dim, pad_id, hidden_dim,
                 n_layers, use_birnn, dout, attn, tied, device):
        super(DecRNN, self).__init__()

        hidden_dim = hidden_dim*2 if use_birnn else hidden_dim

        self.embed = nn.Embedding(vsz, embed_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers)
        self.device = device
        self.w = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn = Attention(hidden_dim, attn)

        self.out_projection = nn.Linear(hidden_dim, vsz)
        if tied:
            if embed_dim != hidden_dim:
                raise ValueError(
                    f"when using the tied flag, embed-dim:{embed_dim} \
                    must be equal to hidden-dim:{hidden_dim}")
            self.out_projection.weight = self.embed.weight
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs):
        inputs = inputs.unsqueeze(0)
        embs = self.dropout(self.embed(inputs))
        dec_out, hidden = self.rnn(embs, hidden)

        attn_weights = self.attn(dec_out, enc_outs).transpose(1, 0)
        enc_outs = enc_outs.transpose(1, 0)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs)
        cats = self.w(torch.cat((dec_out, context.transpose(1, 0)), dim=2))
        pred = self.out_projection(cats.tanh().squeeze(0))
        return pred, hidden


class Seq2seqAttn(nn.Module):
    def __init__(self, src_vsz, tgt_vsz, device, embed_dim, hidden_dim,
                 embed_vec, atten_method, n_layers, bidirectional,
                 dropout, tied, pad_id=1, sos_id=2, eos_id=3):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.src_vsz = src_vsz
        self.tgt_vsz = tgt_vsz
        self.embed_dim = embed_dim
        self.encoder = EncRNN(
            self.src_vsz, embed_dim, pad_id, hidden_dim,
            n_layers, bidirectional, dropout, embed_vec, device
        )
        self.decoder = DecRNN(
            self.tgt_vsz, embed_dim, pad_id, hidden_dim,
            n_layers, bidirectional, dropout,
            atten_method, tied, device
        )

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_birnn = bidirectional

    def forward(self, srcs, tgts, device, maxlen=20, tf_ratio=0.0):
        slen, bsz = srcs.size()
        tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        tf_ratio = tf_ratio if isinstance(tgts, torch.Tensor) else 0.0
        enc_outs, hidden = self.encoder(srcs.to(device))
        dec_inputs = torch.ones_like(srcs[0]) * self.sos_id
        outs = []

        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, bsz, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)

        for i in range(int(tlen)):
            preds, hidden = \
                self.decoder(dec_inputs.to(device), hidden, enc_outs)
            outs.append(preds)
            use_tf = random.random() < tf_ratio
            dec_inputs = tgts[i] if use_tf else preds.max(1)[1]
        return torch.stack(outs)


if __name__ == '__main__':
    my_device = torch.device(7)

    model = Seq2seqAttn(10, 10, my_device, 100, 100,
                 1, 2, 3, None, 'general',
                 2, True, dropout=0.5, tied=False)
    model.to(my_device)
    x = torch.tensor([[5, 6, 7], [6, 6, 6]]).to(my_device)
    y = torch.tensor([[6, 6, 3], [5, 4, 1]]).to(my_device)
    pred = model(x, y, my_device)
    print()