import torch
import numpy as np

from gpt.model import Gpt, GptConfig
from gpt.data import get_batch
from evaluation.evaluation import estimate_loss, evaluate_and_save
from gpt.tok import get_tokenizer
from dataclasses import asdict


torch.manual_seed(1123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------ BASELINE ------------------
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
batch_size = 32

# data is handled in prepare_data.py
data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
n = int(0.99 * len(data))
train, val = data[:n], data[n:]

# model and config
tok = get_tokenizer()
cfg = GptConfig(vocab_size=tok.n_vocab, block_size=256, n_embd=384, n_layer=6, n_head=6)
model = Gpt(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model, {"train": train, "val": val}, batch_size, cfg.block_size, device)
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    x, y = get_batch(train, batch_size=batch_size, block_size=cfg.block_size, device=device)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True) # zero out the gradient so it doesnt accumulate
    loss.backward()
    opt.step()

# save the model
torch.save({
    "model": model.state_dict(),
    "config": asdict(cfg),
    "max_iters": max_iters,
    "ir": learning_rate
}, "checkpoints/baseline_tiktoken.pt")

# save the result
evaluate_and_save(model, 'baseline_char', val, 'val', batch_size, max_iters, learning_rate)

# sample text
context = torch.full((1, 1), tok.eot_token, dtype=torch.long, device=device)   # 50256
out = model.generate(context, 200) # (B, T)
out = out[0].tolist()
sample = tok.decode(out)
print (sample)

# ------------------ MODERN ------------------