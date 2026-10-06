import torch
from torch import Tensor



def make_padding_mask(tokens:Tensor, pad_idx: int = 0):     # tokens de shape (B,L)
    mask = (tokens != pad_idx)
    mask = mask[:,None,None,:] # mask de shape (B,1,1,L)
    return mask


  
def make_causal_mask(seq_len: int,device=None):
    L = seq_len
    mask_caus = torch.ones(L, L, dtype=torch.bool,device=device)
    mask_caus = torch.tril(mask_caus)
    mask_caus = mask_caus[None,None,:,:]
    return mask_caus 


if __name__ == "__main__":
    tokens = torch.tensor([[5, 7, 9, 0, 0],
                        [3, 0, 0, 0, 0]])
    m = make_padding_mask(tokens)
    print(m.shape)
    print(m)