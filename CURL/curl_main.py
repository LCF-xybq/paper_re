import json
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import visdom
import collections

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.logger import get_root_logger
from model import CURLNet, CURLLoss
from utils.trans import Resize, RandomCrop, RandomFlip, ToTensor, Normalize
from data_set import UWData
from utils.Visualizer import Visualizer
from utils.save_image import normimage
from utils.checkpoint import save_epoch, save_latest

class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

def getargs():
    with open('curl_setting.json', 'r') as f:
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
    np.random.seed(26)
    torch.manual_seed(26)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.log_file = os.path.join(args.work_dir, f'{timestamp}.log')

    logger = get_root_logger(log_file=args.log_file, log_level=args.log_level)
    paper_name = 'CURL: Deep Image Formation and Retouching'
    logger.info(paper_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Working on {device}')

    # model
    net = CURLNet()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-10)
    criterion = CURLLoss(ssim_win_size=5)

    num_ele = [p.numel() for p in net.parameters() if p.requires_grad]
    logger.info('Training Parameters: %d', sum(num_ele))

    trans = transforms.Compose([
        Resize(350),
        RandomCrop(320),
        RandomFlip(),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])

    assert os.path.exists(args.ann_file)
    dataset = UWData(args.ann_file, transform=trans)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    visualizer = Visualizer()
    vis = visdom.Visdom()

    ite_num = 0
    max_ite_num = args.num_epoch * len(train_loader)

    print("-----------------One Piece----------------")
    t = time.time()
    net.train()
    for epoch in range(args.num_epoch):
     #   logger.info('\nStart Epoch %d ^v^ ^v^ ^v^ ^v^ ^v^ ^v^', epoch + 1)
        for i, data in enumerate(train_loader):
            date_time = time.time() - t
            ite_num += 1
            ite_to_show = ite_num * len(data)

            lr, gt = data['lr'], data['gt']
            lr = lr.to(device)
            gt = gt.to(device)
            output, gradient_reg = net(lr)

            optimizer.zero_grad()

            losses = criterion(output, gt, gradient_reg)
            l1_loss = losses.l1
            rgb_loss = losses.rgb
            ssim_loss = losses.ssim
            cosine_rgb_loss = losses.cosine_rgb
            hsv_loss = losses.hsv
            curl_loss = losses.curl

            loss = curl_loss

            loss.backward()
            optimizer.step()

            logger.info('Epoch %d, [%d/%d] lr: %f time: %.3f l1_loss: %.3f '
                        'rgb_loss: %.3f ssim_loss: %.3f cosine_rbg_loss: %.3f '
                        'hsv_loss: %.3f loss: %.3f',
                        epoch + 1, ite_num, max_ite_num,
                        optimizer.param_groups[0]['lr'], date_time,
                        l1_loss, rgb_loss, ssim_loss, cosine_rgb_loss,
                        hsv_loss, curl_loss)

            all_loss = collections.OrderedDict()
            all_loss['loss_l1'] = l1_loss.data.cpu().item()
            all_loss['loss_rgb'] = rgb_loss.data.cpu().item()
            all_loss['loss_ssim'] = ssim_loss.data.cpu().item()
            all_loss['loss_cosin_rgb'] = cosine_rgb_loss.data.cpu().item()
            all_loss['loss_hsv'] = hsv_loss.data.cpu().item()
            all_loss['total_loss'] = loss.data.cpu().item()
            visualizer.plot_current_losses(epoch + 1, float(i) / len(train_loader), all_loss)

            t = time.time()
            if ite_to_show % 20 == 0:
                input_show = normimage(lr, save_cfg=True)
                gt_show = normimage(gt, save_cfg=True)
                out_show = normimage(output, save_cfg=True)

                show = []
                show.append(input_show.transpose([2, 0, 1]))
                show.append(gt_show.transpose([2, 0, 1]))
                show.append(out_show.transpose([2, 0, 1]))
                vis.images(show, nrow=4, padding=5, win=1, opts=dict(title='Raw Paper'))

        if epoch % 10 == 0:
            save_latest(net, optimizer, args.ckpt, epoch, ite_num)
            net.train()

    save_epoch(net, optimizer, args.save_pth, args.num_epoch, ite_num)
    logger.info('Finish Training')


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