import os
from torch.utils.data import Dataset

def load_ann(ann_file):
    images = []
    with open(ann_file, 'r') as f:
        for line in f:
            lr_img, hr_img = line.split(' ')




class SRDataset(Dataset):
    def __init__(self, ann_file, lr_folder, hr_folder, scale):
        super(SRDataset, self).__init__()
        assert os.path.isdir(lr_folder)
        assert os.path.isdir(hr_folder)
        assert os.path.isdir(ann_file)

        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.scale = scale

        self.lr_images = load_ann(ann_file)

    def __getitem__(self, item):

