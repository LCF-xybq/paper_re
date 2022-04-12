from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from vgg import Vgg16
import torch.nn as nn

pth = r'D:\Program_self\paper_re\perceptual_loss\images\content-images\amber.jpg'

def tensorToImg(tensor):
    # image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = tensor[0].detach().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    img = image_numpy.astype(np.uint8)
    return img

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

img = Image.open(pth).convert('RGB')
img_tensor = transform(img).unsqueeze(0)



vgg = Vgg16()
feature_map = vgg(img_tensor)

feat_relu1_2 = feature_map.relu1_2
feat_relu2_2 = feature_map.relu2_2
feat_relu3_3 = feature_map.relu3_3
feat_relu4_3 = feature_map.relu4_3


# img_relu2_2 = tensorToImg(feat_relu2_2)[:,:,0:3]
# print(img_relu2_2.shape)
# img_relu3_3 = tensorToImg(feat_relu3_3)[:,:,0:3]
# print(img_relu3_3.shape)
# img_relu4_3 = tensorToImg(feat_relu4_3)[:,:,0:3]
# print(img_relu4_3.shape)]

# 1x1 kernel 不识别空间模式，只是融合通道

conv0 = nn.Conv2d(64, 3, kernel_size=1, stride=1)
conv1 = nn.Conv2d(128, 3, kernel_size=1, stride=1)
conv2 = nn.Conv2d(256, 3, kernel_size=1, stride=1)
conv3 = nn.Conv2d(512, 3, kernel_size=1, stride=1)
relu = nn.ReLU()

img_relu1_2 = relu(conv0(feat_relu1_2))
img_relu2_2 = relu(conv1(feat_relu2_2))
img_relu3_3 = relu(conv2(feat_relu3_3))
img_relu4_3 = relu(conv3(feat_relu4_3))

img_relu1_2 = tensorToImg(img_relu1_2)
print(img_relu1_2.shape)

img_relu2_2 = tensorToImg(img_relu2_2)
print(img_relu2_2.shape)

img_relu3_3 = tensorToImg(img_relu3_3)
print(img_relu3_3.shape)

img_relu4_3 = tensorToImg(img_relu4_3)
print(img_relu4_3.shape)


img_raw = tensorToImg(img_tensor)
plt.imshow(img_raw)
plt.title("raw")
plt.show()

plt.imshow(img_relu1_2)
plt.title("img_relu1_2")
plt.show()

plt.imshow(img_relu2_2)
plt.title("img_relu2_2")
plt.show()

plt.imshow(img_relu3_3)
plt.title("img_relu3_3")
plt.show()

plt.imshow(img_relu4_3)
plt.title("img_relu4_3")
plt.show()
