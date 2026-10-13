import torch
from transformer.masks import make_padding_mask, make_causal_mask


def test_padding_mask_shape():
    tokens = torch.tensor([[5, 3, 0],
                           [7, 0, 0]])          # (B, L) = (2, 3)
    mask = make_padding_mask(tokens)
    assert mask.shape == (2, 1, 1, 3)
    assert mask.dtype == torch.bool


def test_padding_mask_values():
    tokens = torch.tensor([[5, 3, 0],
                           [7, 0, 0]])
    mask = make_padding_mask(tokens)
    attendu = torch.tensor([[True, True, False],
                            [True, False, False]])
    assert torch.equal(mask[:, 0, 0, :], attendu)


def test_padding_mask_pad_idx():
    tokens = torch.tensor([[5, 3, 1]])           # ici le pad vaut 1
    mask = make_padding_mask(tokens, pad_idx=1)
    assert torch.equal(mask[0, 0, 0], torch.tensor([True, True, False]))


def test_causal_mask():
    mask = make_causal_mask(3)
    attendu = torch.tensor([[True, False, False],
                            [True, True, False],
                            [True, True, True]])
    assert mask.shape == (1, 1, 3, 3)
    assert torch.equal(mask[0, 0], attendu)


def test_combined_mask_shape():
    tokens = torch.tensor([[5, 3, 0, 0],
                           [7, 2, 4, 0]])        # (B, L) = (2, 4)
    mask = make_padding_mask(tokens) & make_causal_mask(4)
    assert mask.shape == (2, 1, 4, 4)
