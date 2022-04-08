import argparse
import os
import sys
import torch
import time
import re
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from transformer_net import TransformerNet
from vgg import Vgg16
from utils import load_image, normalize_batch, gram

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
    np.random.seed(0)
    torch.manual_seed(args.seed)

    # dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = TransformerNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    # before sending to vgg to get style_loss
    # it need to be processed
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    # [ch h w] ==> [batchsize ch h w]
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # 把style图片先normlize
    # 接着送入损失网络（VGG）去获取feature map（vgg返回一个命名元组）
    feature_style = vgg(normalize_batch(style))
    # 文中是一个feature map一个gram，然后求和
    gram_style = [gram(y) for y in feature_style]

    # !!!注意，对style transfer任务来说，一个style image一个网络
    # 就是换一个ys，重train一个
    # 输入x == content image，y = fw(x), 也就是原文的 y hat
    # 所以上面的gram_style就是style loss中的减数
    for e in range(args.epochs):
        model.train()
        # 对于style transfor任务，需要 content_loss, style_loss
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch

            optimizer.zero_grad()

            x = x.to(device)
            y = model(x)

            # 该任务的loss在优化迭代过程中会出255的范围
            # 所以要裁一下
            x = normalize_batch(x)
            y = normalize_batch(y)

            # 前馈网络，去损失网络求接下来要给fw方向传播用的perceputal loss
            # style transfer任务中content loss用的是relu2_2的特征
            # content loss = |feature_y - feature_x|^2 / c*h*w
            # mse loss = |A - B|^2 / n , n == chw个数，
            # content loss 2范数平方，2范数就是平方开根，所以一倒腾就是mse
            feature_y = vgg(y)
            feature_x = vgg(x)
            content_loss = args.content_weight * mse_loss(feature_y.relu2_2, feature_x.relu2_2)

            # 接下来求style loss
            style_loss = 0.0
            for feat_y_hat, gm_y in zip(feature_y, gram_style):
                gram_y_hat = gram(feat_y_hat)
                # y_hat是一个又一个batch， y不是，所以切片固定0维，不然会out of idx
                # style_loss 是矩阵相减的F范数的平方，F范数是(|A|^2)^0.5，所以平方后恰巧又是mse
                style_loss += mse_loss(gram_y_hat, gm_y[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent loss: {:.6f}\tstyle loss: {:.6f}\tperceptual loss: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                model.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(model.state_dict(), ckpt_model_path)
                model.to(device).train()

def stylize(args):
    pass

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
