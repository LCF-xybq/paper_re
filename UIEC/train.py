import os
import json
import sys
import torch
import time
import torch
import visdom
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.trans import (Resize, RandomFlip, RandomCrop,
                         ToTensor, Normalize)
from utils.Visualizer import Visualizer
from utils.logger import get_root_logger
from loss.ssim_loss import SSIMLoss
from loss.hsv_loss import HSVLoss
from uiec_dataset import UIECData
from uiec_model import UIEC
from utils.checkpoint import resume, load, save_latest, save_epoch
from vgg import Vgg19


class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

def getargs():
    with open('uiec.json', 'r') as f:
        data = json.load(f, object_hook=JSONObject)

    return data

def check_paths(args):
    try:
        if args.save_pth is not None and not (os.path.exists(args.save_pth)):
            os.makedirs(args.save_pth)
        if args.ckpt is not None and not (os.path.exists(args.ckpt)):
            os.makedirs(args.ckpt)
        if args.work_dir is not None and not (os.path.exists(args.work_dir)):
            os.makedirs(args.work_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def train(args):
    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.log_file = os.path.join(args.work_dir, f'{timestamp}.log')

    # create log file
    logger = get_root_logger(log_file=args.log_file, log_level=args.log_level)
    paper_name = 'UIEC2-Net: CNN-based Underwater Image Enhancement Using Two Color Space\n'
    logger.info(paper_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Working on {device}')

    # model
    model = UIEC().to(device)
    vgg = Vgg19(require_grad=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ssim_loss = SSIMLoss()
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    hsv_loss = HSVLoss()

    if args.resume_from:
        start_epoch, ite_num = resume(args.resume_from, model, optimizer, logger)
    if args.load_from:
        load(args.load_from, model, logger)

    num_ele = [p.numel() for p in model.parameters()]
    num_ele_trainable = [p.numel() for p in model.parameters() if p.requires_grad]
    logger.info('Tota Parameters: %d, Training Parameters: %d',
                sum(num_ele), sum(num_ele_trainable))

    trans = transforms.Compose([
        Resize(350),
        RandomCrop(320),
        RandomFlip(),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5]),
    ])

    assert os.path.exists(args.ann_file)
    dataset = UIECData(args.ann_file, transform=trans)
    batch_size = args.num_gpus * args.samples_per_gpu
    num_workers = args.num_gpus * args.workers_per_gpu
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    visualizer = Visualizer()
    vis = visdom.Visdom()

    ite_num = 0
    max_ite_num = args.epochs * len(train_loader)
    running_loss = 0.0

    # False without Normalize in
    save_cfg = True

    print("-------------Handler One------------")
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        logger.info('\nStart Epoch %d -----------', epoch + 1)
        for i, data in enumerate(train_loader):
            data_time = time.time() - t
            ite_num = ite_num + 1
            ite_num1val = ite_num * args.samples_per_gpu

            lr, gt = data['lr'], data['gt']
            lr = lr.to(device)
            gt = gt.to(device)
            output = model(lr)

            feature_gt = vgg(gt)
            feature_out = vgg(output)

            optimizer.zero_grad()

            loss_ssim = SSIMLoss(output, gt)
            loss_l1 = l1_loss(output, gt)
            loss_per = mse_loss(feature_out.relu4_3, feature_gt.relu4_3)
            loss_hsv = hsv_loss(output, gt)

            if epoch < 20:
                
            else:


def test(args):
    pass

def main():
    args = getargs()

    if args.train:
        check_paths(args)
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main()