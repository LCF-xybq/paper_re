import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from model import CURLNet
from utils.metric import psnr, ssim
from utils.save_image import tensor2img

if __name__ == '__main__':
    pth = r'D:\Program_self\paper_re\data\UW\train\raw\16_img_.png'
    img = Image.open(pth).convert("RGB")

    model = CURLNet().cuda()

    # num_ele = [p.numel() for p in model.parameters()]
    # print(sum(num_ele))

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img_tensor = trans(img)
    img_tensor_batch = img_tensor.unsqueeze(0).cuda()

    output,_ = model(img_tensor_batch)

    ref_pth = r'D:\Program_self\paper_re\data\UW\train\ref\16_img_.png'
    gt = Image.open(ref_pth).convert("RGB")
    ref_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    gt_re = trans(gt)

    out = tensor2img(output, min_max=(-1, 1))
    ref = tensor2img(gt_re, min_max=(-1, 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(out)
    ax[1].imshow(ref)
    plt.show()

    v1 = psnr(out, ref)
    v2 = ssim(out, ref)

    print(v1, v2)