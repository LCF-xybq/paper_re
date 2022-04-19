import torch

from PIL import Image
from torch.utils.data import Dataset

def load_ann(ann_file):
    raw_images = []
    ref_images = []
    with open(ann_file, 'r') as f:
        for line in f:
            raw_pth, ref_pth = line.strip().split(' ')
            raw_img = Image.open(raw_pth).convert('RGB')
            ref_img = Image.open(ref_pth).convert('RGB')
            raw_images.append(raw_img)
            ref_images.append(ref_img)

    return raw_images, ref_images

class UWCNNData(Dataset):
    def __init__(self, ann_file, transform):
        super(UWCNNData, self).__init__()
        self.transform = transform

        self.raw_images, self.ref_images = load_ann(ann_file)
        self.raw_size = len(self.raw_images)
        self.ref_size = len(self.ref_images)
        assert self.raw_size == self.ref_size

    def __getitem__(self, item):
        if self.transform is not None:
            raw = self.transform(self.raw_images[item])
            ref = self.transform(self.ref_images[item])
        else:
            raw = self.raw_images[item]
            ref = self.ref_images[item]

        return {'raw': raw, 'ref': ref}

    def __len__(self):
        return self.raw_size
