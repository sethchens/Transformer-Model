import math
import torch
from gpt.model import GptConfig, MultiHead, FeedForward, Block, Gpt, Tokenizer

CFG = GptConfig()                        # vocab 65, block 8, n_embd 32, 3 layers, 4 heads


def make_ids(B=2, T=8):
    return torch.randint(CFG.vocab_size, (B, T))


# ---------- MultiHead ----------

def test_multihead_keeps_shape():
    out = MultiHead(CFG)(torch.randn(2, 8, CFG.n_embd))
    assert out.shape == (2, 8, CFG.n_embd)


def test_multihead_no_future():
    mh = MultiHead(CFG)
    mh(torch.randn(2, 8, CFG.n_embd))
    w = mh.last_weights                              # [B, n_head, T, T]
    assert (torch.triu(w, diagonal=1) == 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, CFG.n_head, 8))


# ---------- FeedForward / Block ----------

def test_feedforward_keeps_shape():
    assert FeedForward(CFG)(torch.randn(2, 8, CFG.n_embd)).shape == (2, 8, CFG.n_embd)


def test_block_keeps_shape():
    assert Block(CFG)(torch.randn(2, 8, CFG.n_embd)).shape == (2, 8, CFG.n_embd)


# ---------- Gpt ----------

def test_gpt_logits_shape_without_targets():
    logits, loss = Gpt(CFG)(make_ids())
    assert logits.shape == (2, 8, CFG.vocab_size)
    assert loss is None


def test_gpt_untrained_loss_near_log_vocab():
    torch.manual_seed(0)
    _, loss = Gpt(CFG)(make_ids(), make_ids())
    expected = math.log(CFG.vocab_size)              # ≈ 4.17: a uniform guess
    assert abs(loss.item() - expected) < 0.5, f"loss {loss.item():.3f}, expected ≈ {expected:.3f}"


def test_gpt_shorter_sequence():
    logits, _ = Gpt(CFG)(make_ids(T=3))
    assert logits.shape == (2, 3, CFG.vocab_size)


def test_gpt_generate_length_past_block_size():
    model = Gpt(CFG)
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=20)   # 20 > block_size
    assert out.shape == (1, 21)
    assert out.max() < CFG.vocab_size and out.min() >= 0


def test_generate_restores_train_mode():
    model = Gpt(CFG)
    model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=2)
    assert model.training


def test_gpt_learns_one_batch():
    # tiny-batch sanity check: the model must be able to memorize a single batch
    torch.manual_seed(0)
    model = Gpt(CFG)
    x, y = make_ids(), make_ids()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, start = model(x, y)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * start.item(), f"loss only went {start.item():.2f} → {loss.item():.2f}"


# ---------- Tokenizer ----------

def test_tokenizer_round_trip():
    tok = Tokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_tokenizer_ids_are_stable():
    # sorted vocab → the same text always gives the same ids
    assert Tokenizer("cba").encode("abc") == [0, 1, 2]


def test_tokenizer_vocab_size():
    assert Tokenizer("aabbc").n_vocab == 3