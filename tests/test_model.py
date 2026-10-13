import pytest
import torch
from transformer.model import Transformer


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = Transformer(src_vocab_size=11, tgt_vocab_size=13, d_model=16, n_heads=4,
                    n_layers=2, d_ff=40, max_len=50)
    m.eval()
    return m


@pytest.fixture
def src():
    return torch.tensor([[4, 6, 2, 9, 5, 0, 0],
                         [7, 3, 0, 0, 0, 0, 0]])    # (B, L_s) = (2, 7)


@pytest.fixture
def tgt():
    return torch.tensor([[1, 8, 3, 2, 0],
                         [1, 5, 9, 0, 0]])          # (B, L_t) = (2, 5)


def test_logits_shape(model, src, tgt):
    logits, _, _, _ = model(src, tgt)
    assert logits.shape == (2, 5, 13)


@pytest.mark.parametrize("L_s, L_t", [(7, 5), (3, 9), (1, 1)])
def test_various_lengths(model, L_s, L_t):
    src = torch.randint(1, 11, (2, L_s))
    tgt = torch.randint(1, 13, (2, L_t))
    logits, _, _, _ = model(src, tgt)
    assert logits.shape == (2, L_t, 13)


def test_cross_attention_ignores_source_pads(model, src, tgt):
    _, _, _, cross_ws = model(src, tgt)
    for w in cross_ws:                           # w : (B, h, L_t, L_s)
        assert (w[0, :, :, 5:] == 0).all()       # phrase 0 : pads aux positions 5 et 6
        assert (w[1, :, :, 2:] == 0).all()       # phrase 1 : pads à partir de la position 2


def test_model_does_not_see_future(model, src, tgt):
    logits, _, _, _ = model(src, tgt)
    tgt_modif = tgt.clone()
    tgt_modif[:, 3:] = 7
    logits_modif, _, _, _ = model(src, tgt_modif)
    assert torch.allclose(logits[:, :3], logits_modif[:, :3], atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="pas de GPU Apple")
def test_model_on_mps(model, src, tgt):
    model.to("mps")
    logits, _, _, _ = model(src.to("mps"), tgt.to("mps"))
    assert logits.device.type == "mps"
