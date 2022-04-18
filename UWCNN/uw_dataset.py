import torch

from torch.utils.data import Dataset


class UWCNNData(Dataset):
    def __init__(self, ann_file, raw_trans, ref_trans):
        super(UWCNNData, self).__init__()
        self.raw_trans = raw_trans
        self.ref_trans = ref_trans
