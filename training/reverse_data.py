import torch


PAD, BOS, EOS = 0, 1, 2
FIRST_DIGIT = 3
VOCAB_SIZE = 13  



def make_reverse_batch(batch_size: int, max_len: int, device=None):
    # B = batch_size, L = max_len
    lengths = torch.randint(1, max_len + 1, (batch_size,), device=device)                  # (B,)        longueur de chaque phrase, entre 1 et L
    digits = torch.randint(FIRST_DIGIT, VOCAB_SIZE, (batch_size, max_len), device=device)  # (B, L)      chiffres au hasard (3 à 12)
    pos = torch.arange(max_len, device=device)                                             # (L,)        0, 1, ..., L-1
    real = pos[None, :] < lengths[:, None]                                                 # (B, L)      bool : True là où la phrase existe
    src = torch.where(real, digits, PAD)                                                   # (B, L)      chiffres, puis pads
    idx = (lengths[:, None] - 1 - pos[None, :]).clamp(min=0)                               # (B, L)      position à lire dans src pour inverser
    rev = torch.gather(src, 1, idx)                                                        # (B, L)      rev[b, p] = src[b, idx[b, p]]
    rev = torch.where(real, rev, PAD)                                                      # (B, L)      suite inversée, puis pads
    bos = torch.full((batch_size, 1), BOS, device=device)                                  # (B, 1)
    pad = torch.full((batch_size, 1), PAD, device=device)                                  # (B, 1)
    tgt_in = torch.cat([bos, rev], dim=1)                                                  # (B, L+1)    BOS + suite inversée
    tgt_out = torch.cat([rev, pad], dim=1)                                                 # (B, L+1)    suite inversée + une case
    tgt_out[torch.arange(batch_size, device=device), lengths] = EOS                        # EOS juste après le dernier chiffre de chaque ligne
    return src, tgt_in, tgt_out