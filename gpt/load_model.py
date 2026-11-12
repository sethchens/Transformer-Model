import torch

from gpt.model import Gpt
from gpt.config import GptConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GptConfig(**ckpt["config"])
    model = Gpt(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    return model