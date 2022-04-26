import torch

# a = torch.randint(1, 9, (3, 3))
# b = torch.randint(1, 9, (3, 3))
# c = torch.mul(a, b)
# d = torch.add(a, -1, c)
# print(a)
# print(b)
# print(c)
# print(d)

img = torch.randn((4,3,5,5), dtype=float)
print(img)
num_images = img.shape[0]
for i in range(0, num_images):
    target_img = img[i, :, :, :]
    print(target_img.shape)

test_img = target_img[0:1, :, :]
print(test_img.shape)
ttt = torch.cos(test_img)
fff = test_img[0,:,:]
print(ttt.shape)
print(fff.shape)

z = torch.zeros(img.shape[0] - 1)
print(z)