import os
import torch
import torch.nn as nn
from transformer.model import Transformer
from training.reverse_data import make_reverse_batch, PAD, VOCAB_SIZE


CONFIG = dict(src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE, d_model=64, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, max_len=50, pad_idx=PAD)


MAX_LEN = 10
BATCH_SIZE = 64 
N_STEPS = 1500 
WARMUP = 200 
DEVICE = "cpu" 
CHECKPOINT = "checkpoints/reverse.pt" 



def lr_lambda(step):
    step = max(step, 1)
    return CONFIG["d_model"] ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)

def main():
    torch.manual_seed(0)
    model = Transformer(**CONFIG).to(DEVICE)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for step in range(1, N_STEPS + 1):
        model.train()
        src, tgt_in, tgt_out = make_reverse_batch(BATCH_SIZE, MAX_LEN, DEVICE)
        logits, _, _, _ = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        scheduler.step()

        if step % 100 == 0:
            pred = logits.argmax(dim=-1) 
            mask = tgt_out != PAD
            acc = (pred == tgt_out)[mask].float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.3f}  acc {acc:.3f}  lr {scheduler.get_last_lr()[0]:.2e}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"config": CONFIG, "model": model.state_dict()}, CHECKPOINT)
    print("modèle sauvegardé dans", CHECKPOINT)



if __name__ == "__main__":
    main()