import torch
import torch.nn as nn
from torch import Tensor
import math



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        D = d_model
        L = max_len
        tab = torch.zeros(L, D)    # (L, D)

        if D%2 == 1:
            raise ValueError(f'{D} should be even')
        
        for p in range(L):
            for i in range(0,D,2):
                omega_i = (1/(10000**(i/D)))
                tab[p,i] = math.sin(omega_i*p)
                tab[p,i+1] = math.cos(omega_i*p)

        self.register_buffer("tab", tab) 


    def forward(self, x: Tensor):
        L = x.shape[1]
        return x + self.tab[:L,:]