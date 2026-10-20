import torch
from gpt.model import Head, GptConfig

def make_head(batch_size: int = 32):
    cfg = GptConfig()
    head_size = cfg.n_embd // cfg.n_head

    return Head(cfg, head_size), cfg, head_size

def test_head_output_shape():
    # setup
    head, cfg, hs = make_head()
    x = torch.randn(2, cfg.block_size, cfg.n_embd) # (B, T, C)

    # act
    out = head(x)

    # assert
    assert out.shape == (2, 8, hs), f"Expected (2, 8, {hs}), got {tuple(out.shape)}"

def test_head_rows_sum_to_one():
    # setup
    head, cfg, _ = make_head()

    # act
    head(torch.randn(2, 8, cfg.n_embd))
    w = head.last_weights                       # [B, T, T]
    row_sums = w.sum(dim=-1)                    # [B, T]

    # assert
    assert torch.allclose(row_sums, torch.ones(2, 8)), "attention rows must sum to 1"


def test_attention_masking():
    # setup
    head, cfg, _ = make_head()

    # act
    head(torch.randn(2, 8, cfg.n_embd))
    above_diag = torch.triu(head.last_weights, diagonal=1)   # keep a copy of only the future part (diagonal=1 drops the diagonal)

    # assert
    assert (above_diag == 0).all(), "a token attended to a future token"

def test_head_shorter_sequence():
    # setup
    head, cfg, hs = make_head()

    # act
    out = head(torch.randn(2, 5, cfg.n_embd))   # T = 5 < block_size

    # assert
    assert out.shape == (2, 5, hs)

