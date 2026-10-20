import torch
from gpt.data import split_data, get_batch

# todo: read this whole thing
def make_data(n=100):
    return torch.arange(n)          # token ids 0..n-1, so every value shows its own position


# ---------- split_data ----------

def test_split_sizes():
    train, val, test = split_data(make_data(100))
    assert (len(train), len(val), len(test)) == (80, 10, 10)


def test_split_covers_everything_in_order():
    data = make_data(100)
    train, val, test = split_data(data)
    assert torch.equal(torch.cat([train, val, test]), data)   # nothing lost, nothing reordered


def test_split_no_overlap():
    train, val, test = split_data(make_data(100))
    assert train[-1] < val[0] and val[-1] < test[0]


# ---------- get_batch ----------

def test_batch_shapes():
    x, y = get_batch(make_data(100), batch_size=4, block_size=8, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)


def test_batch_targets_are_shifted_by_one():
    x, y = get_batch(make_data(100), batch_size=4, block_size=8, device="cpu")
    assert torch.equal(y, x + 1)    # with arange data, "next token" means value + 1


def test_batch_stays_inside_data():
    data = make_data(20)
    for _ in range(200):            # random starts, so try many times
        x, y = get_batch(data, batch_size=4, block_size=8, device="cpu")
        assert y.max() <= 19


def test_batch_dtype_is_long():
    x, y = get_batch(make_data(100), batch_size=4, block_size=8, device="cpu")
    assert x.dtype == torch.long and y.dtype == torch.long