import torch
from torch import nn

# With Learnable Parameters
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = torch.randn(20, 100)
print(f"input shape: {input.shape}")
print(torch.norm(input))
output = m(input)
print(torch.norm(output))