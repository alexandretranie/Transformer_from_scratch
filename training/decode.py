import torch
from transformer.model import Transformer
from training.reverse_data import make_reverse_batch, PAD, BOS, EOS, FIRST_DIGIT
from training.train_reverse import MAX_LEN, CHECKPOINT


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)        # {"config": ..., "model": ...}
    model = Transformer(**ckpt["config"]).to(device)    # même architecture, poids aléatoires
    model.load_state_dict(ckpt["model"])                # poids entraînés
    model.eval()                                        # pas de dropout
    return model


@torch.no_grad()
def greedy_decode(model, src, max_len):
    # src : (B, L_s)
    B = src.shape[0]
    tgt = torch.full((B, 1), BOS, device=src.device)            # (B, 1)
    for _ in range(max_len):
        logits, _, _, _ = model(src, tgt)                       # (B, t, V)
        next_tok = logits[:, -1, :].argmax(dim=-1)              # (B,)   meilleur token à la dernière position
        tgt = torch.cat([tgt, next_tok[:, None]], dim=1)        # (B, t+1)
    return tgt[:, 1:]                                           # (B, max_len)   sans le BOS


def to_digits(tokens):
    # tokens : (L,)  →  "9 6 3"
    out = []
    for t in tokens.tolist():
        if t == EOS:
            break
        if t >= FIRST_DIGIT:
            out.append(str(t - FIRST_DIGIT))
    return " ".join(out)


def main():
    torch.manual_seed(1)                                 
    model = load_model(CHECKPOINT)

    src, _, tgt_out = make_reverse_batch(1000, MAX_LEN)         # (1000, L), (1000, L+1)
    pred = greedy_decode(model, src, MAX_LEN + 1)               # (1000, L+1)   +1 pour la place de l'EOS

    for b in range(5):
        print(f"{to_digits(src[b]):>20}  →  {to_digits(pred[b])}")

    mask = tgt_out != PAD                                       # (1000, L+1)   True là où la réponse compte
    seq_ok = ((pred == tgt_out) | ~mask).all(dim=1)             # (1000,)       True si toute la suite est juste
    print(f"suites entièrement justes : {seq_ok.float().mean().item():.1%}")


if __name__ == "__main__":
    main()
