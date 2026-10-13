import torch
from transformer.feed_forward import FeedForward
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.masks import make_causal_mask


def test_feed_forward_shape():
    ffn = FeedForward(d_model=16, d_ff=40)
    assert ffn(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_encoder_shapes():
    enc = Encoder(n_layers=3, d_model=16, n_heads=4, d_ff=40)
    out, weights = enc(torch.randn(2, 7, 16))
    assert out.shape == (2, 7, 16)
    assert len(weights) == 3
    assert weights[0].shape == (2, 4, 7, 7)


def test_decoder_shapes():
    dec = Decoder(n_layers=3, d_model=16, n_heads=4, d_ff=40)
    out, self_w, cross_w = dec(torch.randn(2, 5, 16), torch.randn(2, 7, 16))
    assert out.shape == (2, 5, 16)
    assert self_w[0].shape == (2, 4, 5, 5)
    assert cross_w[0].shape == (2, 4, 5, 7)


def test_decoder_does_not_see_future():
    dec = Decoder(n_layers=2, d_model=16, n_heads=4, d_ff=40)
    dec.eval()                                   # coupe le dropout
    x = torch.randn(1, 5, 16)
    memory = torch.randn(1, 7, 16)
    mask = make_causal_mask(5)
    out, _, _ = dec(x, memory, mask)
    x_modif = x.clone()
    x_modif[:, 3:] = 100.                        # on change les positions 3 et 4
    out_modif, _, _ = dec(x_modif, memory, mask)
    assert torch.allclose(out[:, :3], out_modif[:, :3], atol=1e-5)
