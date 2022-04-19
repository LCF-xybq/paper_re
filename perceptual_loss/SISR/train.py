import json
import re

import torch
import sys
import os
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.nn.functional import interpolate

from torchvision import transforms
from torch.utils.data import DataLoader

from sisr_net import SISRNet
from sr_dataset import SRDataset
from perceptual_loss.vgg import Vgg16
from perceptual_loss.util import load_image, save_image

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
    print(f'training on {device}')

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4731, 0.4444, 0.4033],
                                 std=[0.2466, 0.2401, 0.2602]),
            transforms.Lambda(lambda x: x.mul(255))
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
            x = x.to(device)
            y = model(x)

            hr = data['hr']
            hr = hr.to(device )

            feature_y = vgg(y)
            feature_yc = vgg(hr)
            content_loss = args.content_weight * mse_loss(feature_y.relu2_2, feature_yc.relu2_2)

            content_loss.backward()

            optimizer.step()

            agg_content_loss += content_loss.item()

            if (e + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tpercetual loss: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                    agg_content_loss / (args.content_weight * (batch_id + 1))
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (e + 1) % args.log_interval == 0:
                model.eval().cpu()
                ckpt_name = "ckpt_epoch_" + str(e) + ".pth"
                ckpt_path = os.path.join(args.checkpoint_model_dir + ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                model.to(device).train()

    # save model
    model.eval().cpu()
    saved_model_filename = "epoch_" + str(args.epochs) + ".model"
    saved_model_path = os.path.join(args.save_model_dir, saved_model_filename)
    torch.save(model.state_dict(), saved_model_path)

    print("\nDone, SISR model saved at", saved_model_path)

def sisr(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f'Testing on {device}')

    lr_img = load_image(args.lr_image_test)

    lr_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(72),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    lr_img = lr_transform(lr_img)
    lr_img = lr_img.unsqueeze(0).to(device)

    with torch.no_grad():
        model = SISRNet()
        saved_dict = torch.load(args.model)

        for k in list(saved_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del saved_dict[k]

        model.load_state_dict(saved_dict)
        model.to(device)
        model.eval()
        output = model(lr_img).cpu()

    save_image(args.output_image, output[0])


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