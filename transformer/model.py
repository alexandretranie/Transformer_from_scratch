import torch.nn as nn
from torch import Tensor
from transformer.embeddings import TransformerEmbedding
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.masks import make_padding_mask, make_causal_mask


class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,n_heads=8,n_layers=6,d_ff=2048,dropout=0.1,max_len=5000,pad_idx=0):
        super().__init__()
        self.src_embed = TransformerEmbedding(src_vocab_size,d_model,max_len,dropout)
        self.tgt_embed = TransformerEmbedding(tgt_vocab_size,d_model,max_len,dropout)
        self.encoder = Encoder(n_layers,d_model,n_heads,d_ff,dropout)
        self.decoder = Decoder(n_layers,d_model,n_heads,d_ff,dropout)
        self.out_proj = nn.Linear(d_model,tgt_vocab_size)
        self.pad_idx = pad_idx


    def forward(self,src,tgt):
        # src : (B, L_s)   tgt : (B, L_t)   (des entiers : les indices des tokens)
        src_mask = make_padding_mask(src,self.pad_idx)                  # (B, 1, 1, L_s)
        tgt_mask = make_padding_mask(tgt,self.pad_idx) & make_causal_mask(tgt.shape[1], device=tgt.device)   # (B, 1, L_t, L_t)
        src_x = self.src_embed(src)                                     # (B, L_s, D)
        memory, enc_weights = self.encoder(src_x,src_mask)              # (B, L_s, D), N × (B, h, L_s, L_s)
        tgt_x = self.tgt_embed(tgt)                                     # (B, L_t, D)
        x, dec_self_weights, dec_cross_weights = self.decoder(tgt_x,memory,tgt_mask,src_mask)   # (B, L_t, D), N × (B, h, L_t, L_t), N × (B, h, L_t, L_s)
        logits = self.out_proj(x)                                       # (B, L_t, V_t)
        return logits,enc_weights,dec_self_weights,dec_cross_weights