from rgb2hsv import RGB2HSV
from hsv2rgb import HSV2RGB
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

pth = r'D:\Program_self\paper_re\data\16.jpg'

img = Image.open(pth).convert("RGB")

tran = transforms.Compose([
    transforms.Resize(3),
    transforms.CenterCrop(3),
    transforms.ToTensor(),
])

img_tensor = tran(img)
img_tensor_batch = img_tensor.unsqueeze(0)

r2h = RGB2HSV()
h2r = HSV2RGB()

print(img_tensor_batch[0, 1, :, :])

img_r2h = r2h(img_tensor_batch)
print(img_r2h[0, 1, :, :])

img_h2r = h2r(img_r2h)
print(img_h2r[0, 1, :, :])
