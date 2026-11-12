import os
import time
import torch
import numpy as np

from gpt.model import Gpt
from gpt.train_utils import get_lr, save_checkpoint
from gpt.generate import generate
from gpt.data import get_batch
from gpt.config import GptConfig, TrainConfig
from evaluation.evaluation import estimate_loss, evaluate_and_save, count_parameter
from gpt.tok import get_tokenizer

os.makedirs("checkpoints", exist_ok=True)
torch.manual_seed(1123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tcfg = TrainConfig(run_name="lr1e-3_warmup_clip", learning_rate=1e-3, grad_clip=1.0, warmup_iters=200)
ckpt_path = f"checkpoints/{tcfg.run_name}.pt"

# ------------------ BASELINE ------------------
# data is handled in prepare_data.py
data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
n = int(0.99 * len(data))
train, val = data[:n], data[n:]

# model and config
tok = get_tokenizer()
cfg = GptConfig(vocab_size=tok.n_vocab, block_size=256, n_embd=384, n_layer=6, n_head=6, norm="layernorm")
model = Gpt(cfg).to(device)
_, total = count_parameter(model)
print(f"params: {total:,}")
opt = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate)

# save the model here once to prevent saving error after a long training is done
save_checkpoint(model, cfg, tcfg, ckpt_path)

# for total training time (M2 from SRS)
t0 = time.perf_counter()

# training loop
for step in range(tcfg.max_iters):
    if step % tcfg.eval_interval == 0 or step == tcfg.max_iters - 1:
        save_checkpoint(model, cfg, tcfg, ckpt_path)
        losses = estimate_loss(model, {"train": train, "val": val}, tcfg.batch_size, cfg.block_size, device)
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    x, y = get_batch(train, batch_size=tcfg.batch_size, block_size=cfg.block_size, device=device)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True) # zero out the gradient so it doesnt accumulate
    loss.backward()
    if tcfg.grad_clip is not None:
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip) # grad clip
    opt.param_groups[0]["lr"] = get_lr(step, tcfg) # lr warmup
    opt.step()

# computing the total training time (M2 from SRS)
print(f"training time: {(time.perf_counter() - t0) / 60:.1f} min")

# save the model
save_checkpoint(model, cfg, tcfg, ckpt_path)

# save the result
evaluate_and_save(model, val, 'val', tcfg)

# sample text
generate(ckpt_path=ckpt_path, device=device)

# ------------------ MODERN ------------------