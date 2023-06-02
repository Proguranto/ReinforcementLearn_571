import torch.nn as nn
import torch

loss = nn.MSELoss()
input = torch.randn((1, 3), requires_grad=True)
target = torch.randn(1, 3)
output = loss(input, target)

print(target[0][1])