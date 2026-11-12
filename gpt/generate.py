import torch

from gpt.tok import get_tokenizer
from gpt.load_model import load_model
from gpt.config import SampleConfig, TrainConfig

tok = get_tokenizer()

# todo: turn this into generator
def write_stories(model, scfg, device, context=None):
    if context is None:
        context = torch.full((1, 1), tok.eot_token, dtype=torch.long, device=device)

    stories = []
    for _ in range(scfg.n_stories):
        out = model.generate(context, scfg.max_new_tok, scfg.temp, scfg.top_k)
        n_new = out.shape[1] - context.shape[1]
        assert n_new == scfg.max_new_tok, f"expected {scfg.max_new_tok}, got {n_new}"
        out = out[0].tolist()
        sample = tok.decode(out)
        stories.append((n_new, sample))

    return stories

def generate(ckpt_path, device, scfg=None, context=None):
    scfg = scfg or SampleConfig()
    print(f"checkpoint: {ckpt_path}")
    print(f"temp={scfg.temp}, top_k={scfg.top_k}, max_new_tok={scfg.max_new_tok}, n_stories={scfg.n_stories}")
    print("=" * 50)

    model = load_model(ckpt_path, device)
    model.eval()
    stories = write_stories(model, scfg, device, context)

    for i, (n_new, story) in enumerate(stories):
        print(f"--- Story {i+1} ({n_new} new tokens) ---")
        print(story)

if __name__ == "__main__":
    tcfg = TrainConfig()
    ckpt_path = f"checkpoints/{tcfg.run_name}.pt"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(1123)
    print(f"seed=1123")

    generate(ckpt_path=ckpt_path, device=device)