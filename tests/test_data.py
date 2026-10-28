import numpy as np
import torch
import pytest

from gpt.data import split_data, get_batch


@pytest.fixture
def data():
    # uint16 like train.bin; arange makes "next token = this token + 1" easy to check
    return np.arange(1000, dtype=np.uint16)


# ---------- split_data ----------

def test_split_sizes(data):
    tr, va, te = split_data(data)
    assert len(tr) == 800 and len(va) == 100 and len(te) == 100


def test_split_covers_everything(data):
    tr, va, te = split_data(data)
    assert len(tr) + len(va) + len(te) == len(data)


def test_split_no_overlap(data):
    tr, va, te = split_data(data)
    assert tr[-1] < va[0] and va[-1] < te[0]


# ---------- get_batch ----------

def test_batch_shapes(data):
    x, y = get_batch(data, batch_size=4, block_size=8, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)


def test_batch_y_is_x_shifted(data):
    x, y = get_batch(data, batch_size=4, block_size=8, device="cpu")
    assert torch.equal(y, x + 1)          # arange → next token is +1
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_stays_inside(data):
    for _ in range(50):
        x, y = get_batch(data, batch_size=16, block_size=8, device="cpu")
        assert x.min() >= 0 and y.max() < len(data)


def test_batch_dtype_long(data):
    x, y = get_batch(data, batch_size=4, block_size=8, device="cpu")
    assert x.dtype == torch.long and y.dtype == torch.long


def test_batch_works_on_memmap(tmp_path):
    # real-path check: write a .bin, read it back via memmap, batch from it
    path = tmp_path / "toy.bin"
    np.arange(500, dtype=np.uint16).tofile(path)
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    x, y = get_batch(mm, batch_size=4, block_size=8, device="cpu")
    assert torch.equal(y, x + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_batch_on_cuda(data):
    x, y = get_batch(data, batch_size=4, block_size=8, device="cuda")
    assert x.is_cuda and y.is_cuda