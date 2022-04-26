import torch

img = torch.randint(1,9, (4, 3, 5, 5))
# a 是索引
# b 是切片
a = img[:, 0 , :, :]
b = img[:, 0:1, :, :]
print(a.shape, b.shape)
print(a)
print(b)