from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GptConfig:
    vocab_size: int = 65
    block_size: int = 8
    n_embd: int = 32
    n_layer: int = 3
    n_head: int = 4    

class Head(nn.Module):
    def __init__(self, config: GptConfig, hs: int):
        super().__init__()
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) # tril = triangle lower
        self.omega_q = nn.Linear(config.n_embd, hs, bias=False) # takes (int, put) where in -> row and out-> col
        self.omega_k = nn.Linear(config.n_embd, hs, bias=False) # and the stored weight (e.g q.weight) is [out, in]
        self.omega_v = nn.Linear(config.n_embd, hs, bias=False)
        self.last_weights = None # for test retrieval; filled in by forward()

    def forward(self, x):
        B, T, C = x.shape # C is n_embd
        q = self.omega_q(x) # omega_q gets broadcasted on batch dim and q is shaped (B, T, hs)
        k = self.omega_k(x)
        v = self.omega_v(x)

        scores = q @ k.transpose(-2, -1) # .T gets B at the last pos; shaped (B, T, T)
        scores = scores * k.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1) # last dim is the key dim where the sum happens along (the m in UDL)
        self.last_weights = weights
        return weights @ v # value v_j (projected) multiplied by a score (weight) along dim m and summed (B, T, hs)

        

    

class Gpt(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self._config = config

    def forward(self):
        pass

    def generate(self):
        pass

    def get_config(self):
        return self._config


if __name__ == '__main__':
    pass