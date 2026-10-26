import torch

def split_data(data: torch.Tensor):
    bin_1 = int(0.8 * len(data))
    bin_2 = int(0.9 * len(data))

    train = data[ : bin_1]
    val   = data[ bin_1: bin_2]
    test  = data[ bin_2: ]

    return train, val, test

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]).to(device)

    return x, y