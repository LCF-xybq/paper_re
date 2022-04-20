from uw_dataset import UWCNNData
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.trans import *

if __name__ == '__main__':
    ann_pth = r'D:\Program_self\paper_re\data\UW\ann.txt'

    # transform = transforms.Compose([
    #     transforms.Resize(550),
    #     transforms.RandomCrop(512),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    # ])

    transform = transforms.Compose([
        Resize(550),
        RandomCrop(512),
        RandomFlip(),
        ToTensor(),
        Normalize()
    ])

    train_dataset = UWCNNData(ann_file=ann_pth, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4)

    for i, data in enumerate(train_loader):
        print(i, len(data['lr']), data['gt'].shape)
