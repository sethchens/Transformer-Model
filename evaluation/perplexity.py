import math
import time
import json
import torch
import torch.nn as nn

from gpt.model import GptConfig
from dataclasses import asdict
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def estimate_perplexity(model: nn.Module, val: torch.Tensor):
    assert val.dim() == 1, f"expected 1-D token ids, got shape {tuple(val.shape)}"
    assert val.dtype == torch.long, f"expected torch.long, got {val.dtype}"
    
    config = model.get_config()
    model.eval()
    losses = []
    for i in range(0, len(val) - config.block_size, config.block_size):
        x_val = val[i:i+config.block_size].unsqueeze(0).to(device)
        y_val = val[i+1:i+config.block_size+1].unsqueeze(0).to(device)
        _, loss = model(x_val, y_val)
        losses.append(loss.item()) # .item() turns the loss scaler into a Python float
    mean_nll = sum(losses) / len(losses)
    perplexity = math.exp(mean_nll)
    model.train()
    return mean_nll, perplexity

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
def get_token_ps_inf(model_cls: type[nn.Module], config: GptConfig, n: int = 100):
    if torch.cuda.is_available():
        model = model_cls(config).to(device)
        model.eval() # so model behaves the way it does at inference

        x = torch.randint(config.vocab_size, (config.batch_size, config.block_size), device=device)
        B, T = x.shape

        # warm up
        for i in range(0, 10):
            _, _ = model(X)

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
def evaluate_and_save(model: nn.Module, which: str, batch_size: int, val: torch.Tensor, n: int = 100):
    model_cls = type(model)
    config = model.get_config()

    # get model results
    mean_nll, perplexity = estimate_perplexity(model, val)
    _, total = count_parameter(model)
    peak_by_inf_mb, shape_inf = get_peak_mem_inference(model_cls, config)
    peak_by_train_mb, shape_train = get_peak_mem_train(model_cls, config)
    token_ps, shape_tps_inf = get_token_ps_inf(model_cls, config, n)

    # get model configs from config argument

    # get context
    timestamp = datetime.now().isoformat()
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'

    record = {
        "name": which,
        "param": total,
        "val_ppl": perplexity,
        "val_mean_nll": mean_nll,
        "inf_mem_mb": peak_by_inf_mb,
        "train_mem_mb": peak_by_train_mb,
        "tps": token_ps,
        "input_shape": list(shape_inf),
        "config": asdict(config),
        "timestamp": timestamp,
        "device_name": device_name
    }

    # save each record as jsonl
    with open('evaluation/results.jsonl', "a", encoding="utf-8")as f:
        f.write(json.dumps(record))
        f.write("\n")