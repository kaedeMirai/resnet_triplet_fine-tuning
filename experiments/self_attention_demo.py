import copy
import math

import torch
import torch.nn as nn

Wq = torch.randn(512, 64)
Wk = torch.randn(512, 64)
Wv = torch.randn(512, 64)

X = torch.randn(1, 512)

query = X @ Wq
key = X @ Wk
value = X @ Wv


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(0)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = torch.softmax(scores, dim=-1)
    result = torch.matmul(p_attn, value)

    return result


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        print(f"Batch size: {nbatches}")
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x, in zip(self.linears, (query, key, value))
        ]
        print(f"Q/K/V shapes: {query.size()}, {key.size()}, {value.size()}")
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        print(f"Attention output shape: {x.size()}")
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        print(f"Merged output shape: {x.size()}")
        return self.linears[-1](x)


batch_size = 2
seq_len = 13
d_model = 512
h = 8

mha = MultiHeadedAttention(h=h, d_model=d_model, dropout=0.1)
X = torch.randn(batch_size, seq_len, d_model)

output = mha(query=X, key=X, value=X, mask=None)
