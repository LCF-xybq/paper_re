import json
import torch
import sys
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.nn.functional import interpolate

from torchvision import transforms
from torch.utils.data import DataLoader

from sisr_net import SISRNet
from sr_dataset import SRDataset
from perceptual_loss.vgg import Vgg16

class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

def getargs():
    with open('sisr.json', 'r') as f:
        data = json.load(f, object_hook=JSONObject)

        return data

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4731, 0.4444, 0.4033],
                                 std=[0.2466, 0.2401, 0.2602]),
        ])

    lr_transform = transforms.Compose([
        transforms.Resize(72, interpolation='bicubic'),
        transforms.ToTensor()
    ])

    train_dataset = SRDataset(ann_file=args.ann_file, scale=args.scale, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = SISRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    for e in range(args.epochs):
        model.train()
        agg_content_loss = 0.0
        count = 0

        for batch_id, data in enumerate(train_loader):
            n_batch = len(data['lr'])
            count += n_batch

            optimizer.zero_grad()

            # resize x == x / scale
            x = data['lr']
            # print(x.shape)
            x = interpolate(x, size=[72, 72], mode='bicubic', align_corners=False)
            # print(x.shape)



def sisr(args):
    pass

def main():
    args = getargs()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.train:
        check_paths(args)
        train(args)
    else:
        sisr(args)

if __name__ == '__main__':
    main()