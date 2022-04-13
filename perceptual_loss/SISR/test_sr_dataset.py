from sr_ann_dataset import SRAnnDataset
from base_sr_dataset import BaseSRDataset
from pipeline import Collect, LoadImageFromFile, ImageToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class RepeatDataset:
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.times * self._ori_len

if __name__ == '__main__':
    lq_folder = r'D:\Program_self\paper_re\data\SR\train\lq'
    gt_folder = r'D:\Program_self\paper_re\data\SR\train\gt'
    ann_file = r'D:\Program_self\paper_re\perceptual_loss\SISR\ann.txt'

    data_pipeline = transforms.Compose([LoadImageFromFile, Collect, ImageToTensor])


    train = SRAnnDataset(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         ann_file=ann_file,
                         pipeline=data_pipeline,
                         scale=4)

    dataset = [RepeatDataset(train, times=1)]
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
