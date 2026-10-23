from dataclasses import dataclass

import math
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

class MultiHead(nn.Module): # the fused version
    def __init__(self, config: GptConfig, n_head: int):
        super().__init__()

        assert config.n_embd % n_head == 0, f"embedding length is not a multiple of head size."
        self.last_weights = None # for test retrieval
        self.n_head = n_head
        self.hs = config.n_embd // n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd) # q, k, v stacked vertically
        self.proj = nn.Linear(config.n_embd, config.n_embd)    # the project weights that makes each head able to contribute to each output term
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.hs).transpose(-3, -2)
        k = k.view(B, T, self.n_head, self.hs).transpose(-3, -2)
        v = v.view(B, T, self.n_head, self.hs).transpose(-3, -2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hs) # q @ k is a dot product which has hs terms
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask off the future attention scores
        scores = F.softmax(scores, dim=-1) # sum along the m (key dim)
        self.last_weights = scores.detach() # for test retrieval
        y = scores @ v
        y = y.transpose(-3, -2).contiguous().view(B, T, C) # concat each head back (transpose doesnt relocate data on mem so .contiguous)

        return self.proj(y)

# UDL figure 12.7 (Parallel neural network)
class FeedForward(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.omega_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.omega_out = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.omega_in(x)  # (B, T, 4C)
        x = F.relu(x)         # (B, T, 4C) 
        x = self.omega_out(x) # (B, T, C)  

        return x

class Block(nn.Module):
    def __init__(self, config: GptConfig, n_head: int):
        super().__init__()
        self.sa = MultiHead(config, n_head)
        self.ffwd = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.n_embd) # it takes on required arg 'normalized_shape', which is the last dim
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # the structure from attention is all you need paper

        # x + ... is a residual connection; most important reason is 
        # that gradient has this fastpass (x) to flow back (backprop)
        x = x + self.sa(x)  
        x = self.ln_1(x)
        x = x + self.ffwd(x)
        x = self.ln_2(x)

        return x
    
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