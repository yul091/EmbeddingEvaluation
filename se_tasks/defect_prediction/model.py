import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_class, embed, max_len, ):
        super(textCNN, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.embeding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5,
                stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,64,64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.attention = SelfAttention(36)
        self.out = nn.Linear(36, n_class)


    def forward(self, x):
        x = self.embeding(x)
        x = x.view(x.size(0), 1, self.max_len, self.embed_dim)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), x.size(1), -1)
        #x = x.view(x.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        x, _ = self.attention(x)
        output = self.out(x)
        return output


if __name__ == '__main__':
    model = textCNN(1000, 100, 2, None, 100).cuda()
    x = torch.zeros([1, 100, 1], dtype=torch.long).cuda()
    y = model(x)
    print(y)