import torch
import torch.nn as nn
import torch.nn.functional as F


class Code2Vec(nn.Module):
    def __init__(
            self, nodes_dim, paths_dim, embed_dim,
            output_dim, embed_vec, dropout=0.5, padding_index=1
    ):
        super().__init__()
        self.embedding_dim = embed_dim

        if torch.is_tensor(embed_vec):
            self.node_embedding = nn.Embedding(nodes_dim, embed_dim, padding_idx=padding_index, _weight=embed_vec)
            self.node_embedding.weight.requires_grad = False
        else:
            self.node_embedding = nn.Embedding(nodes_dim, embed_dim, padding_idx=padding_index)

        self.path_embedding = nn.Embedding(paths_dim, embed_dim)
        self.W = nn.Parameter(torch.randn(1, embed_dim, 3 * embed_dim))
        self.a = nn.Parameter(torch.randn(1, embed_dim, 1))
        self.out = nn.Linear(embed_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, starts, paths, ends, length, device):
        W = self.W.repeat(len(starts), 1, 1)
        # W = [batch size, embedding dim, embedding dim * 3]
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        # embedded_* = [batch size, max length, embedding dim]

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # [batch, embedding_dim * 3, max_length]
        x = torch.tanh(torch.bmm(W, c))  # [batch, embedding dim, max length]
        x = x.permute(0, 2, 1)
        a = self.a.repeat(len(starts), 1, 1)  # [batch, embedding dim, 1]
        z = torch.bmm(x, a).squeeze(2)  # [batch size, max length]
        z = F.softmax(z, dim=1)   # [batch size, max length]
        z = z.unsqueeze(2)  # [batch size, max length, 1]
        x = x.permute(0, 2, 1)  # [batch, embedding dim, max length]

        v = torch.zeros(len(x), self.embedding_dim).to(device)
        for i in range(len(x)):
            v[i] = torch.bmm(
                x[i:i+1, :, :length[i]], z[i:i+1, :length[i], :]
            ).squeeze(2)
        #v = torch.bmm(x, z).squeeze(2)  # [batch size, embedding dim]
        out = self.out(v)  # [batch size, output_dim]
        return out
