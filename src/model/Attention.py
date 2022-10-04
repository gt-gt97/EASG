import math

import torch
from torch import nn
from torch.nn.functional import softmax


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = 0

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        # queries和keys的最后一维都为d
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度 (b, q, d) * (b, d, k) = (b, q, k)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

# # d = softmax(torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=float))
# # print(d)
# queries = torch.tensor([[1, 2, 3]], dtype=float)
# keys = torch.tensor([[1, 2, 3]], dtype=float)
# values = torch.tensor([[1, 2, 3]], dtype=float)
# attention = DotProductAttention(dropout=0.5)
# attention.eval()
# # 部分参数沿用加性注意力中的参数
# b = attention(queries, keys, values)
# print(b)
