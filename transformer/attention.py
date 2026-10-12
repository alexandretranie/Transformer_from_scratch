import math, torch
from torch import Tensor
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    # q : (..., L_q, d_k)   k : (..., L_k, d_k)   v : (..., L_k, d_v)
    d = q.shape[-1]                                         # d_k
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)       # (..., L_q, L_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))   # (..., L_q, L_k)

    weights = torch.softmax(scores, dim=-1)                 # (..., L_q, L_k)
    output = weights @ v                                    # (..., L_q, d_v)

    return output, weights



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'{d_model} is not divided by {n_heads}')
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.req = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.val = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        



    
    def forward(self, query, key, value, mask=None):
        # query : (B, L_q, D)   key, value : (B, L_k, D)
        q = self.req(query)                                 # (B, L_q, D)
        k = self.key(key)                                   # (B, L_k, D)
        v = self.val(value)                                 # (B, L_k, D)

        B , L_q, D = q.shape
        B , L_k, D = k.shape

        q = q.reshape(B,L_q,self.n_heads,self.d_k)          # (B, L_q, h, d_k)
        k = k.reshape(B,L_k,self.n_heads,self.d_k)          # (B, L_k, h, d_k)
        v = v.reshape(B,L_k,self.n_heads,self.d_k)          # (B, L_k, h, d_k)

        q = q.transpose(1,2)                                # (B, h, L_q, d_k)
        k = k.transpose(1,2)                                # (B, h, L_k, d_k)
        v = v.transpose(1,2)                                # (B, h, L_k, d_k)

        output, weights = scaled_dot_product_attention(q, k, v, mask)  # (B, h, L_q, d_k), (B, h, L_q, L_k)
        output = output.transpose(1, 2)                     # (B, L_q, h, d_k)
        output = output.reshape(B, L_q, D)                  # (B, L_q, D)

        output = self.out(output)                           # (B, L_q, D)
        return output, weights