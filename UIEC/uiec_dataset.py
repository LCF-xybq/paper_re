import copy
from torch.utils.data import Dataset
from PIL import Image

def load_ann(ann_file):
    data = []
    with open(ann_file, 'r') as f:
        for line in f:
            lr_pth, gt_pth = line.strip().split(' ')
            lr = Image.open(lr_pth).convert("RGB")
            gt = Image.open(gt_pth).convert("RGB")
            data.append({
                "lr": lr,
                "gt": gt
            })
    return data

class UIECData(Dataset):
    def __init__(self, ann_file, transform):
        super(UIECData, self).__init__()
        self.transform = transform
        self.data = load_ann(ann_file)
        self.size = len(self.data)

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(self.data[item])
        else:
            return self.data[item]

    def __len__(self):
        return self.size
