import torch

a = torch.randint(1, 9, (3, 3))
b = torch.randint(1, 9, (3, 3))
c = torch.mul(a, b)
d = torch.add(a, -1, c)
print(a)
print(b)
print(c)
print(d)