import imageio
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    pth = r'D:\Program_self\paper_re\temp\images\content-images\amber.jpg'
    img_arr = imageio.imread(pth)

    img = torch.from_numpy(img_arr)
    out = img.permute(2, 0, 1) # to c h w

    print(out[0,0,0:5])

    # uint8:0~255
    # 223 * 255 = 56865
    # 56865 / 256 = 222 * 256 + 33
    lam_mul = transforms.Lambda(lambda x: x.mul(255))
    out_mul = lam_mul(out)
    print(out_mul[0, 0, 0:5])

    out_mul_manual = torch.mul(out, 255)
    print(out_mul_manual[0, 0, 0:5])
