# Transformer-Model

My senior project at BYU-Idaho. I built a small GPT-style transformer from scratch in PyTorch to really understand how self-attention works, and trained it on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) to write short kids' stories. I also ran a bunch of experiments to see which training tweaks actually matter.

## What's in here

- `gpt/` has the model itself (attention, feed-forward blocks, LayerNorm or RMSNorm), plus the config, batching, tokenizer, and text generation
- `prepare_data.py` tokenizes TinyStories with the GPT-2 tokenizer (`tiktoken`) and saves it as `.bin` files so training can memory-map them
- `train.py` is the training loop (AdamW, optional LR warmup and gradient clipping, checkpoints)
- `evaluation/` measures perplexity, memory use, and speed, and logs every run to `results.jsonl`
- `dashboard.py` is a Streamlit app for comparing runs
- `tests/` has the pytest tests

## Results so far

Every run uses the same model: about 49M params, 6 layers, 6 heads, 384-dim embeddings, 256-token context. Each one trains for 5,000 steps on my RTX 5060 Ti.

| Run | LR | Warmup | Grad clip | Val perplexity |
|---|---|---|---|---|
| Baseline | 3e-4 | no | no | 6.82 |
| RMSNorm | 3e-4 | no | no | 6.82 |
| Higher LR | 1e-3 | no | no | 355 (stalled) |
| Higher LR + clipping | 1e-3 | no | 1.0 | 6.05 |
| Higher LR + warmup | 1e-3 | 200 steps | no | 6.04 |
| Higher LR + both | 1e-3 | 200 steps | 1.0 | 6.04 |

Bumping the LR to 1e-3 on its own completely stalled training. Adding warmup or clipping (either one works) fixed that and got perplexity from 6.82 down to about 6.04. Swapping LayerNorm for RMSNorm didn't change anything at this size.

## Running it

```bash
pip install -r requirements.txt    # pulls the CUDA 12.8 build of PyTorch

# put TinyStoriesV2-GPT4-train.txt and TinyStoriesV2-GPT4-valid.txt in data/
python prepare_data.py
python train.py                    # trains, evaluates, then prints a few sample stories
streamlit run dashboard.py
pytest
```
