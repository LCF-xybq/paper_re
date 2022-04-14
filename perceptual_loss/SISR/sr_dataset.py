import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def get_mean_std(dataset):
    images = []
    for i,data in enumerate(dataset):
        images.append(data['lr'])

    imgs = torch.stack([img_t for img_t in images], dim=3)
    print(imgs.shape)

    mean = imgs.view(3, -1).mean(dim=1)
    std = imgs.view(3, -1).std(dim=1)
    print(f'mean = {mean}\tstd = {std}')


def load_ann(ann_file):
    lr_images = []
    hr_images = []
    with open(ann_file, 'r') as f:
        for line in f:
            lr_img_pth, hr_img_pth = line.strip().split(' ')
            lr_img = Image.open(lr_img_pth).convert('RGB')
            hr_img = Image.open(hr_img_pth).convert('RGB')
            lr_images.append(lr_img)
            hr_images.append(hr_img)

    return lr_images, hr_images

class SRDataset(Dataset):
    def __init__(self, ann_file, scale, transform):
        super(SRDataset, self).__init__()
        self.scale = scale
        self.transform = transform

        self.lr_images, self.hr_images = load_ann(ann_file)
        self.lr_size = len(self.lr_images)
        self.hr_size = len(self.hr_images)
        assert self.lr_size == self.hr_size

    def __getitem__(self, item):
        if self.transform is not None:
            lr = self.transform(self.lr_images[item])
            hr = self.transform(self.hr_images[item])
        else:
            lr = self.lr_images[item]
            hr = self.hr_images[item]

        return {'lr': lr, 'hr': hr}

    def __len__(self):
        return self.hr_size

# if __name__ == '__main__':
#     ann_pth = r'D:\Program_self\paper_re\data\SR\ann.txt'
#     transform = transforms.Compose([
#         transforms.Resize(300),
#         transforms.CenterCrop(288),
#         transforms.Normalize(mean=[0.4731, 0.4444, 0.4033],
#                              std=[0.2466, 0.2401, 0.2602]),
#         transforms.ToTensor()
#     ])
#
#     train_dataset = SRDataset(ann_file=ann_pth, scale=4, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=4)
#     # get_mean_std(train_dataset)
