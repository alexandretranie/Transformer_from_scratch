import torch
import torch.nn as nn
from torch import Tensor



class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff = 2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x): #(B,L,D)
        x = self.linear1(x)     # (B, L, d_ff)
        x = torch.relu(x)       # (B, L, d_ff)
        x = self.linear2(x)     # (B, L, D)
        return x