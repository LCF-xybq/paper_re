import copy
from torch.utils.data import Dataset
from PIL import Image

def load_ann(ann_file):
    data_pth = []
    with open(ann_file, 'r') as f:
        for line in f:
            raw_pth, ref_pth = line.strip().split(' ')
            lr = Image.open(raw_pth).convert('RGB')
            gt = Image.open(ref_pth).convert('RGB')
            data_pth.append({
                "lr": lr,
                "gt": gt
            })
    return data_pth

class UWCNNData(Dataset):
    def __init__(self, ann_file, transform):
        super(UWCNNData, self).__init__()
        self.transform = transform
        self.data = load_ann(ann_file)
        self.size = len(self.data)

    def __getitem__(self, item):
        if self.transform is not None:
            results = copy.deepcopy(self.data[item])
            return self.transform(results)
        else:
            return self.data[item]

    def __len__(self):
        return self.size
