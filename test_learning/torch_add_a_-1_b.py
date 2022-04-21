import torch

a = torch.randint(1, 9, (3, 3))
b = torch.randint(1, 9, (3, 3))
e = torch.randint(1, 9, (3, 3))
c = torch.add(a, -1, b)
d = torch.add(a, torch.mul(b, -1))

print(a)
print(b)
print(e)
print(c)
print(d)

f = torch.add(torch.add(a, b), e)
print(f)
