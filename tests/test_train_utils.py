import pytest
from gpt.config import TrainConfig
from gpt.train_utils import get_lr


def cfg(**kw):
    return TrainConfig(learning_rate=1e-3, **kw)


def test_defaults_turn_both_off():
    # old runs must reproduce: no clip, no warmup
    t = TrainConfig()
    assert t.grad_clip is None and t.warmup_iters == 0


def test_no_warmup_gives_full_lr_from_step_0():
    t = cfg(warmup_iters=0)
    assert get_lr(0, t) == 1e-3
    assert get_lr(4999, t) == 1e-3


def test_step_0_is_small_not_full():
    # the bug from the first draft: step 0 must NOT get the full lr
    t = cfg(warmup_iters=200)
    assert get_lr(0, t) == pytest.approx(1e-3 / 200)


def test_mid_ramp_is_linear():
    t = cfg(warmup_iters=200)
    assert get_lr(99, t) == pytest.approx(0.5e-3)  # (99+1)/200 = 0.5


def test_reaches_full_lr_at_end_of_ramp_and_stays():
    t = cfg(warmup_iters=200)
    assert get_lr(199, t) == pytest.approx(1e-3)
    assert get_lr(200, t) == pytest.approx(1e-3)
    assert get_lr(4999, t) == pytest.approx(1e-3)


def test_ramp_never_decreases():
    t = cfg(warmup_iters=200)
    lrs = [get_lr(s, t) for s in range(400)]
    assert all(a <= b for a, b in zip(lrs, lrs[1:]))


def test_get_lr_does_not_modify_config():
    # the other first-draft bug: computing a value must not change tcfg
    t = cfg(warmup_iters=0)
    get_lr(10, t)
    assert t.warmup_iters == 0