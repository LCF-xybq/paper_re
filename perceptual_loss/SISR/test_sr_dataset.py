from sr_ann_dataset import SRAnnDataset
from base_sr_dataset import BaseSRDataset

from torchvision import transforms
import torch

if __name__ == '__main__':
    lq_folder = r'D:\Program_self\paper_re\data\SR\train\lq'
    gt_folder = r'D:\Program_self\paper_re\data\SR\train\gt'
    ann_file = r'D:\Program_self\paper_re\perceptual_loss\SISR\ann.txt'

    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(288),
        transforms.ToTensor()
    ])

    train = SRAnnDataset(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         ann_file=ann_file,
                         transform=transform,
                         scale=4)

    for i in range(len(train.data_infos)):
        print(train.data_infos[i]['lq_path'], train.data_infos[i]['gt_path'])