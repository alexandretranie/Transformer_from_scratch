import torch
import torch.nn as nn 
from torch import Tensor
import math
from transformer.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int,max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        D = d_model 
        L = max_len
        self.emb = nn.Embedding(vocab_size, D)  
        self.pe = PositionalEncoding(D, L)
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model
      
    def forward(self, tokens: Tensor):
        x = self.emb(tokens)            # (B, L) => (B, L, D)
        x = math.sqrt(self.d_model)*x
        x = self.pe(x)
        x= self.drop(x)
        return x