#  Expand dimension demo
import torch

cdw = torch.tensor([0.2, 1, 0.5])
vec = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
])
cwd_weight = cdw.unsqueeze(-1).repeat(1, 4)
print(cwd_weight)
print(torch.mul(vec, cwd_weight))

# tensor([[0.2000, 0.2000, 0.2000, 0.2000],
#         [1.0000, 1.0000, 1.0000, 1.0000],
#         [0.5000, 0.5000, 0.5000, 0.5000]])
# tensor([[0.2000, 0.2000, 0.2000, 0.2000],
#         [2.0000, 2.0000, 2.0000, 2.0000],
#         [1.5000, 1.5000, 1.5000, 1.5000]])