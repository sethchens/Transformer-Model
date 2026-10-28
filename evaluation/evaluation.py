import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt.model import GptConfig
from gpt.data import get_batch
from dataclasses import asdict
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, device, eval_iters=20):
    # quick progress check: average loss over a few random batches per split
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out

@torch.no_grad()
@torch.no_grad()
def estimate_perplexity(model, data, max_tokens=500_000, batch_size=64):
    """Perplexity over a fixed, in-order slice of held-out data.
    data: 1-D np.ndarray / memmap (uint16) or 1-D torch tensor of token IDs."""
    assert data.ndim == 1, "data must be 1-D token IDs"

    # convert only the slice we evaluate
    d = data[:max_tokens + 1]
    if isinstance(d, np.ndarray):
        d = torch.from_numpy(d.astype(np.int64))
    d = d.long()

    config = model.get_config()
    T = config.block_size
    device = next(model.parameters()).device

    # inputs/targets, trimmed to whole windows → [N, T]
    x, y = d[:-1], d[1:]
    n = (len(x) // T) * T
    assert n > 0, "not enough tokens for one window"
    x = x[:n].view(-1, T)
    y = y[:n].view(-1, T)

    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        logits, _ = model(xb)                                   # [B, T, V]
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              yb.view(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += yb.numel()
    if was_training:
        model.train()

    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll), mean_nll

def count_parameter(model: nn.Module):
    learned_params = {}
    total = 0
    for name, p in model.named_parameters():
        count = p.numel()
        learned_params[name] = count
        total += count

    return learned_params, total

# @torch.no_grad have model not keeping the computation graph
@torch.no_grad
def get_peak_mem_inference(model_cls: type[nn.Module], batch_size: int, config: GptConfig):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() # It reset to the size of the allocated memory currently in use, not 0
        curr_allocation = torch.cuda.memory_allocated()
        # randint(high, size): random ints in [0, high), arranged in shape `size`
        # → valid token IDs (0..vocab-1) in shape [B, T], same as get_batch gives
        x = torch.randint(config.vocab_size, (batch_size, config.block_size), device=device)

        model = model_cls(config).to(device)
        model.eval() # Dropout and other things are off
        _, _ = model(x)
        model.train()

        peak_by_inf = torch.cuda.max_memory_allocated() - curr_allocation # this gives out in bytes
        peak_by_inf_mb = peak_by_inf / (1024 * 1024)

        return peak_by_inf_mb, x.shape

    else:
        return None

def get_peak_mem_train(model_cls: type[nn.Module], batch_size: int, config: GptConfig):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() # It reset to the size of the allocated memory currently in use, not 0
        curr_allocation = torch.cuda.memory_allocated()
        # randint(high, size): random ints in [0, high), arranged in shape `size`
        # → valid token IDs (0..vocab-1) in shape [B, T], same as get_batch gives
        x = torch.randint(config.vocab_size, (batch_size, config.block_size), device=device)
        y = torch.randint(config.vocab_size, (batch_size, config.block_size), device=device)

        # If we pass a model in and use it for the mem peak, it changes the real model
        model = model_cls(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        peak_by_train = torch.cuda.max_memory_allocated() - curr_allocation # this gives out in bytes
        peak_by_train_mb = peak_by_train / (1024 * 1024)

        return peak_by_train_mb, x.shape
    else:
        return None

@torch.no_grad()
def get_token_ps_inf(model_cls: type[nn.Module], config: GptConfig, batch_size: int, n: int = 100):
    if torch.cuda.is_available():
        model = model_cls(config).to(device)
        model.eval() # so model behaves the way it does at inference

        x = torch.randint(config.vocab_size, (batch_size, config.block_size), device=device)
        B, T = x.shape

        # warm up
        for i in range(0, 10):
            _, _ = model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(0, n):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        token_ps = (n * B * T) / elapsed

        return token_ps, x.shape
    else:
        return None

# todo: seperate the evaluation and recording into two 
def evaluate_and_save(model: nn.Module, which: str, val: torch.Tensor, split: str, batch_size: int, max_iters: int, lr, n: int = 100):
    model_cls = type(model)
    config = model.get_config()

    # get model results
    perplexity, mean_nll = estimate_perplexity(model, val)
    _, total = count_parameter(model)
    peak_by_inf_mb, shape_inf = get_peak_mem_inference(model_cls, batch_size, config)
    peak_by_train_mb, shape_train = get_peak_mem_train(model_cls, batch_size, config)
    token_ps, shape_tps_inf = get_token_ps_inf(model_cls, config, batch_size, n)

    # get model configs from config argument

    # get context
    timestamp = datetime.now().isoformat()
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'

    record = {
        "name": which,
        "param": total,
        "split": split,
        "ppl": perplexity,
        "mean_nll": mean_nll,
        "inf_mem_mb": peak_by_inf_mb,
        "train_mem_mb": peak_by_train_mb,
        "tps": token_ps,
        "input_shape": list(shape_inf),
        "config": asdict(config),
        "max_iters": max_iters,
        "lr": lr,
        "timestamp": timestamp,
        "device_name": device_name
    }

    # save each record as jsonl
    with open('evaluation/results.jsonl', "a", encoding="utf-8")as f:
        f.write(json.dumps(record))
        f.write("\n")