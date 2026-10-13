import pytest
import torch
from transformer.attention import scaled_dot_product_attention, MultiHeadAttention
from transformer.masks import make_causal_mask


def test_sdpa_shapes():
    q = torch.randn(2, 4, 5, 8)                  # (B, h, L_q, d_k)
    k = torch.randn(2, 4, 7, 8)                  # (B, h, L_k, d_k)
    v = torch.randn(2, 4, 7, 8)                  # (B, h, L_k, d_v)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5, 8)
    assert w.shape == (2, 4, 5, 7)


def test_sdpa_weights_sum_to_one():
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 5))


def test_sdpa_causal_mask():
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    mask = make_causal_mask(4)[0]                # (1, 4, 4)
    _, w = scaled_dot_product_attention(q, k, v, mask)
    futur = ~mask[0]                             # True au-dessus de la diagonale
    assert (w[0][futur] == 0).all()


def test_mha_cross_shapes():
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16)                    # requêtes : (B, L_q, D)
    memory = torch.randn(2, 7, 16)               # clés et valeurs : (B, L_k, D)
    out, w = mha(x, memory, memory)
    assert out.shape == (2, 5, 16)
    assert w.shape == (2, 4, 5, 7)


def test_mha_bad_n_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=16, n_heads=3)
