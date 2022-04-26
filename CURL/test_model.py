import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from model import CURLNet

if __name__ == '__main__':
    pth = r'D:\Program_self\paper_re\data\UW\train\raw\16_img_.png'
    img = Image.open(pth).convert("RGB")

    model = CURLNet().cuda()

    num_ele = [p.numel() for p in model.parameters()]
    print(sum(num_ele))

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img_tensor = trans(img)
    img_tensor_batch = img_tensor.unsqueeze(0).cuda()

    output,_ = model(img_tensor_batch)

    out_numpy = output[0].cpu().detach().numpy()

    if out_numpy.shape[0] == 3:
        out_numpy = (np.transpose(out_numpy, (1, 2, 0))) * 255.0
    else:
        raise ValueError('didi')

    out_img = out_numpy.astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img)
    ax[1].imshow(out_img)
    plt.show()