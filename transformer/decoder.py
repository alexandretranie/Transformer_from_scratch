import torch.nn as nn
from torch import Tensor
from transformer.attention import MultiHeadAttention
from transformer.feed_forward import FeedForward



class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff=2048,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,memory,tgt_mask=None,src_mask=None):
        attn_out, self_weights = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x+self.dropout(attn_out))
        cross_out, cross_weights = self.cross_attn(x,memory,memory,src_mask)
        x = self.norm2(x+self.dropout(cross_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x+self.dropout(ffn_out))
        return x, self_weights, cross_weights


class Decoder(nn.Module):
    def __init__(self,n_layers,d_model,n_heads,d_ff = 2048,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    
    def forward(self, x: Tensor,memory,tgt_mask=None, src_mask = None):
        self_weights_all = []
        cross_weights_all = []  
        for layer in self.layers:
            x, self_weights, cross_weights = layer(x, memory, tgt_mask, src_mask)
            self_weights_all.append(self_weights)
            cross_weights_all.append(cross_weights)
        return x, self_weights_all, cross_weights_all
