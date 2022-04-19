import json
import os
import sys
import time
import torch
import visdom
import collections
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from uw_dataset import UWCNNData
from uwcnn_model import UWCNN
from utils import get_root_logger, Visualizer, normimage
from ssim_loss import SSIMLoss
from util import resume, load, save_latest, save_epoch

class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

def getargs():
    with open('uwcnn.json', 'r') as f:
        data = json.load(f, object_hook=JSONObject)

    return data

def check_paths(args):
    try:
        if not os.path.exists(args.save_pth):
            os.makedirs(args.save_pth)
        if args.ckpt is not None and not (os.path.exists(args.ckpt)):
            os.makedirs(args.ckpt)
        if args.work_dir is not None and not (os.path.exists(args.work_dir)):
            os.makedirs(args.work_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def train(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.log_file = os.path.join(args.work_dir, f'{timestamp}.log')

    # create text log
    logger = get_root_logger(log_file=args.log_file, log_level=args.log_level)
    dash_line = '-' * 60 + '\n'
    logger.info(dash_line)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Working on {device}')
    # model
    model = UWCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ssim_loss = SSIMLoss()
    l1_loss = nn.MSELoss()

    if args.resume_from:
        start_epoch, ite_num = resume(args.resume_from, model, optimizer, logger)
    if args.load_from:
        load(args.load_from, model, logger)

    num_ele = [p.numel() for p in model.parameters()]
    num_ele_grad = [p.numel() for p in model.parameters() if p.requires_grad]
    logger.info('Total Parameters: %d, Trainable Parameters: %s',
                sum(num_ele), sum(num_ele_grad))

    transform = transforms.Compose([
        transforms.Resize(550),
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    dataset = UWCNNData(ann_file=args.ann_file, transform=transform)
    batch_size = args.num_gpus * args.samples_per_gpu
    num_workers = args.num_gpus * args.workers_per_gpu
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)

    visualizer = Visualizer()
    vis = visdom.Visdom()

    ite_num = 0
    # ite 1 time, validate 1 time
    # ite_numxval: ite x time, val 1 time
    ite_num1val = 0
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

            raw, ref = data['raw'], data['ref']
            raw = raw.to(device)
            ref = ref.to(device)
            output = model(raw)

            optimizer.zero_grad()
            loss_ssim = ssim_loss(output, ref)
            loss_l1 = l1_loss(output, ref)
            loss = args.ssim_weight * loss_ssim + args.l1_weight * loss_l1
            loss.backward()
            optimizer.step()

            logger.info('Epoch %d, [%d/%d] lr: %f time: %.3f loss_l1: %f loss_ssim: %f loss: %f',
                        epoch+1, ite_num, max_ite_num, optimizer.param_groups[0]['lr'],
                        data_time, loss_l1, loss_ssim, loss)

            all_loss = collections.OrderedDict()
            all_loss['loss_l1'] = loss_l1.data.cpu()
            all_loss['loss_ssim'] = loss_ssim.data.cpu()
            all_loss['total_loss'] = loss.data.cpu()
            visualizer.plot_current_losses(epoch+1, float(i) / len(train_loader), all_loss)

            # after this iteration
            time_after_iter = time.time() - t
            t = time.time()
            if ite_num1val:
                input_show = normimage(raw, save_cfg=save_cfg)
                ref_show = normimage(ref, save_cfg=save_cfg)
                out_show = normimage(output, save_cfg=save_cfg)

                shows = []
                shows.append(input_show.transpose([2, 0, 1]))
                shows.append(ref_show.transpose([2, 0, 1]))
                shows.append(out_show.transpose([2, 0, 1]))
                vis.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
                ite_num1val = 0

            if ite_num % 2 == 0:
                save_latest(model, optimizer, args.ckpt, epoch, ite_num)
                model.train()

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_epoch(model, optimizer, args.save_pth, epoch, ite_num)
            model.train()

    save_epoch(model, optimizer, args.save_pth, args.epochs, ite_num)
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

if __name__ == "__main__":
    main()
