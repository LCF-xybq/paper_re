from uiec_model import UIEC
from uiec_dataset import UIECData
from utils.trans import Resize, RandomFlip, RandomCrop, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

def build_loader():
    ann_pth = r'D:\Program_self\paper_re\data\UW\ann.txt'

    trans = transforms.Compose([
        Resize(350),
        RandomCrop(320),
        RandomFlip(),
        ToTensor()
    ])

    dataset = UIECData(ann_pth, trans)

    data_loader = DataLoader(dataset, batch_size=1)

    return data_loader


if __name__ == '__main__':
    model = UIEC().cuda()

    num = [p.numel() for p in model.parameters()]
    print(sum(num))

    data_loader = build_loader()

    data = next(iter(data_loader))

    lr = data['lr'].cuda()
    gt = data['gt'].cuda()

    out = model(lr)

    image_numpy = out[0].cpu().detach().float().numpy()
    if image_numpy.shape[0] == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0

    result = image_numpy.astype(np.uint8)
    plt.imshow(result)
    plt.show()
