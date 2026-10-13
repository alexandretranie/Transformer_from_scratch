import pytest
import torch
from transformer.positional_encoding import PositionalEncoding
from transformer.embeddings import TransformerEmbedding


def test_pe_shape():
    pe = PositionalEncoding(d_model=16, max_len=50)
    x = torch.zeros(2, 10, 16)                   # (B, L, D)
    assert pe(x).shape == (2, 10, 16)


def test_pe_position_zero():
    pe = PositionalEncoding(d_model=16, max_len=50)
    out = pe(torch.zeros(1, 1, 16))              # x nul : la sortie est le vecteur de position 0
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(8))   # sin(0) = 0
    assert torch.allclose(out[0, 0, 1::2], torch.ones(8))    # cos(0) = 1


def test_pe_is_buffer():
    pe = PositionalEncoding(d_model=16, max_len=50)
    assert len(list(pe.parameters())) == 0
    assert len(list(pe.buffers())) == 1


def test_pe_odd_d_model():
    with pytest.raises(ValueError):
        PositionalEncoding(d_model=15, max_len=50)


def test_embedding_shape():
    emb = TransformerEmbedding(vocab_size=20, d_model=16, max_len=50)
    tokens = torch.tensor([[4, 7, 1],
                           [2, 0, 0]])           # (B, L) = (2, 3)
    assert emb(tokens).shape == (2, 3, 16)
