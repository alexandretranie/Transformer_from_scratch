import torch.nn as nn
from torch import Tensor
from transformer.attention import MultiHeadAttention
from transformer.feed_forward import FeedForward




class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff=2048,dropout= 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads) 
        self.ffn = FeedForward(d_model, d_ff) 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        attn_out, weights = self.self_attn(x,x,x,mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x) #(B,L,D)
        x = self.norm2(x + self.dropout(ffn_out))
        return x,weights

class Encoder(nn.Module):
    def __init__(self,n_layers,d_model,n_heads,d_ff = 2048,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])


    def forward(self, x: Tensor, mask: Tensor | None = None):
        all_weights = []
        for layer in self.layers:
            x,weights = layer(x,mask)
            all_weights.append(weights)
        return x,all_weights