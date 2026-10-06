import torch

print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device="cuda")
print((x @ x).sum())