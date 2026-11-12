from dataclasses import dataclass

@dataclass
class TrainConfig:
    run_name: str = "baseline_tiktoken"
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    batch_size: int = 32
    grad_clip: float | None = None
    warmup_iters: int = 0

@dataclass
class SampleConfig:
    temp: float = 0.8
    top_k: int | None = 50
    n_stories: int = 5
    max_new_tok: int = 200

@dataclass
class GptConfig:
    vocab_size: int = 65
    block_size: int = 8
    n_embd: int = 32
    n_layer: int = 3
    n_head: int = 4
    norm: str = "layernorm"    