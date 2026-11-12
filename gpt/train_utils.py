import torch

from dataclasses import asdict


def save_checkpoint(model, cfg, tcfg, path):
    torch.save({"model": model.state_dict(),
                "config": asdict(cfg),
                "train": asdict(tcfg)}, path)

def get_lr(step, tcfg):
    if tcfg.warmup_iters <= 0:
        return tcfg.learning_rate
    return tcfg.learning_rate * min(1, (step + 1) / tcfg.warmup_iters)