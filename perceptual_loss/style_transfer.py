import json
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
from util import load_image, normalize_batch, gram, save_image

class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

def getargs():
    with open('style_transformer.json', 'r') as f:
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

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

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
                ckpt_model_path = os.path.join(args.checkpoint_model_dir + ckpt_model_filename)
                torch.save(model.state_dict(), ckpt_model_path)
                # 别忘了 .train() 因为前面有 .eval()
                model.to(device).train()

    # save model
    model.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + '_' + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def stylize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = load_image(args.content_image, args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        saved_dict = torch.load(args.model)

        for k in list(saved_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del saved_dict[k]

        raw_dict = style_model.state_dict()
        change_map = {}
        for old_k, new_k in zip(saved_dict.keys(), raw_dict.keys()):
            change_map[old_k] = new_k

        for k in list(saved_dict.keys()):
            new_k = change_map[k]
            saved_dict.update({new_k: saved_dict.pop(k)})

        for k1, k2 in zip(saved_dict, raw_dict):
            assert k1 == k2

        style_model.load_state_dict(saved_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()

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
        stylize(args)


if __name__ == "__main__":
    main()
