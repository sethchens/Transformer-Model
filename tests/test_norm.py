import pytest
import torch
import torch.nn as nn

from gpt.config import GptConfig
from gpt.model import RMSNorm, make_norm, Block, Gpt


# ---------- RMSNorm itself ----------

def test_rmsnorm_shape():
    x = torch.randn(2, 5, 32)
    assert RMSNorm(32)(x).shape == (2, 5, 32)


def test_rmsnorm_output_has_unit_rms():
    # with g = 1 (the init), every token's output should have RMS ≈ 1 (Eq S.4)
    x = torch.randn(2, 5, 32) * 7 + 3
    y = RMSNorm(32)(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_scale_invariant():
    # h -> c*h gives the same output (section S.2.1)
    norm = RMSNorm(32)
    x = torch.randn(2, 5, 32)
    assert torch.allclose(norm(x), norm(10 * x), atol=1e-5)


def test_rmsnorm_not_shift_invariant():
    # h -> h + b changes the output (no re-centering)
    norm = RMSNorm(32)
    x = torch.randn(2, 5, 32)
    assert not torch.allclose(norm(x), norm(x + 5), atol=1e-3)


def test_rmsnorm_gain_is_learnable():
    norm = RMSNorm(32)
    params = list(norm.parameters())
    assert len(params) == 1 and params[0].shape == (32,)
    assert torch.equal(params[0], torch.ones(32))


# ---------- make_norm ----------

def test_make_norm_picks_the_right_class():
    assert isinstance(make_norm(GptConfig(norm="layernorm")), nn.LayerNorm)
    assert isinstance(make_norm(GptConfig(norm="rmsnorm")), RMSNorm)


def test_make_norm_rejects_typos():
    with pytest.raises(ValueError):
        make_norm(GptConfig(norm="rms"))


# ---------- inside the model ----------

def test_block_with_rmsnorm_keeps_shape():
    cfg = GptConfig(norm="rmsnorm")
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    assert Block(cfg)(x).shape == x.shape


def test_default_is_still_the_layernorm_baseline():
    # building with the default and with norm="layernorm" must give the identical model
    torch.manual_seed(0)
    a = Gpt(GptConfig())
    torch.manual_seed(0)
    b = Gpt(GptConfig(norm="layernorm"))
    ids = torch.randint(0, 65, (2, 8))
    assert torch.equal(a(ids)[0], b(ids)[0])


def test_rmsnorm_has_fewer_params():
    # each LayerNorm has γ and δ (2·C); each RMSNorm only g (C); 2 norms per block
    cfg_ln, cfg_rms = GptConfig(), GptConfig(norm="rmsnorm")
    n_ln = sum(p.numel() for p in Gpt(cfg_ln).parameters())
    n_rms = sum(p.numel() for p in Gpt(cfg_rms).parameters())
    assert n_ln - n_rms == cfg_ln.n_layer * 2 * cfg_ln.n_embd


# ---------- the test that was missing last week ----------

def test_generate_with_top_k_and_temp():
    model = Gpt(GptConfig(norm="rmsnorm"))
    ctx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(ctx, 10, temp=0.8, top_k=5)
    assert out.shape == (1, 11)