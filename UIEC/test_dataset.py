from uiec_dataset import UIECData
from utils.trans import Resize, RandomFlip, RandomCrop, ToTensor

from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    ann_pth = r'D:\Program_self\paper_re\data\UW\ann.txt'

    trans = transforms.Compose([
        Resize(350),
        RandomCrop(320),
        RandomFlip(),
        ToTensor()
    ])

    dataset = UIECData(ann_pth, trans)

    data_loader = DataLoader(dataset, batch_size=4)

    for i, data in enumerate(data_loader):
        print(i, data['lr'].shape, data['gt'].shape)