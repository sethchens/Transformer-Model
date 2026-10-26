import os, urllib.request
import torch

from gpt.model import Gpt, GptConfig, Tokenizer
from gpt.data import split_data, get_batch
from evaluation.perplexity import estimate_loss, evaluate_and_save


torch.manual_seed(1123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------ BASELINE ------------------
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
batch_size = 32

# data
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(url, 'input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tok = Tokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
train, val, test = split_data(data)

# model and config
cfg = GptConfig(vocab_size=tok.n_vocab)
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

# save the result
evaluate_and_save(model, 'baseline_char', val, 'val', batch_size, max_iters, learning_rate)

# sample text
context = torch.zeros(1, 1, dtype=torch.long).to(device) # (B, T)
out = model.generate(context, 200) # (B, T)
out = out[0].tolist()
sample = tok.decode(out)
print (sample)

# ------------------ MODERN ------------------