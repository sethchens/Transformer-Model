import torch
import numpy as np

def split_data(data: torch.Tensor):
    bin_1 = int(0.8 * len(data))
    bin_2 = int(0.9 * len(data))

    train = data[ : bin_1]
    val   = data[ bin_1: bin_2]
    test  = data[ bin_2: ]

    return train, val, test

def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = []
    y = []
    for i in idx:
        chunck = torch.from_numpy(data[int(i):int(i)+block_size+1].astype(np.int64))
        x.append(chunck[:-1])
        y.append(chunck[1:])

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)
    return x, y