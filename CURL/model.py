import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from util import ImageProcessing
from model_ted import TEDModle
from collections import namedtuple

class CURLLoss(nn.Module):
    def __init__(self, ssim_win_size=5, alpha=0.5):
        super(CURLLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_win_size

    def create_window(self, window_size, num_channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs

    def compute_msssim(self, img1, img2):
        if img1.size() != img2.size():
            raise RuntimeError('Input images must have the same shape (%s vs. %s).' % (
                img1.size(), img2.size()))
        if len(img1.size()) != 4:
            raise RuntimeError(
                'Input images must have four dimensions, not %d' % len(img1.size()))

        if not isinstance(img1, torch.Tensor) and not isinstance(img2, torch.Tensor):
            raise RuntimeError(
                'Input images must be Tensor, not %s' % img1.__class__.__name__)

        weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

        if img1.is_cuda:
            weights = weights.cuda(img1.get_device())

        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.compute_ssim(img1, img2)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        mssim = torch.stack(mssim, dim=0)
        mcs = torch.stack(mcs, dim=0)

        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

        prod = (torch.prod(mcs[0:levels - 1] ** weights[0:levels - 1])
                * (mssim[levels - 1] ** weights[levels - 1]))
        return prod

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        num_images = target_img_batch.shape[0]
        target_image_batch = target_img_batch

        ssim_loss_value = torch.zeros(1, 1).cuda()
        l1_loss_value = torch.zeros(1, 1).cuda()
        # L rgb,nc
        cosine_rgb_loss_value = torch.zeros(1, 1).cuda()
        hsv_loss_value = torch.zeros(1, 1).cuda()
        rgb_loss_value = torch.zeros(1, 1).cuda()

        for i in range(0, num_images):
            # ndim=3
            target_img = target_image_batch[i, :, :, :].cuda()
            # ndim=3
            predicted_img = predicted_img_batch[i, :, :, :].cuda()

            predicted_img_lab = torch.clamp(ImageProcessing.rgb_to_lab(predicted_img), 0, 1)

            target_img_lab = torch.clamp(ImageProcessing.rgb_to_lab(target_img), 0, 1)


            predicted_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(predicted_img), 0, 1)

            target_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(target_img), 0, 1)


            # ndim=3
            predicted_h = (predicted_img_hsv[0:1, :, :] * 2 * math.pi)
            predicted_s = predicted_img_hsv[1:2, :, :]
            predicted_v = predicted_img_hsv[2:3, :, :]
            target_h = (target_img_hsv[0:1, :, :] * 2 * math.pi)
            target_s = target_img_hsv[1:2, :, :]
            target_v = target_img_hsv[2:3, :, :]

            # ndim=4
            target_img_L_ssim = target_img_lab[0:1, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0:1, :, :].unsqueeze(0)

            ssim_value = self.compute_msssim(predicted_img_L_ssim, target_img_L_ssim)
            ssim_loss_value += (1.0 - ssim_value)

            # l hsv = w*||s*v*cos(h)_hat - s*v*sin(h)||
            # ndim=3
            predicted_1 = predicted_v * predicted_s * torch.cos(predicted_h)
            predicted_2 = predicted_v * predicted_s * torch.sin(predicted_h)

            # ndim=3
            target_1 = target_v * target_s * torch.cos(target_h)
            target_2 = target_v * target_s * torch.sin(target_h)

            predicted_img_hsv = torch.stack(
                (predicted_1, predicted_2, predicted_v), 2
            )

            target_img_hsv = torch.stack(
                (target_1, target_2, target_v), 2
            )

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)
            rgb_loss_value += F.l1_loss(predicted_img, target_img)
            hsv_loss_value += F.l1_loss(predicted_img_hsv, target_img_hsv)

            cosine_rgb_loss_value += (1 - torch.mean(
                torch.nn.functional.cosine_similarity(predicted_img, target_img, dim=0)
            ))

        l1_loss_value = l1_loss_value / num_images
        rgb_loss_value = rgb_loss_value / num_images
        ssim_loss_value = ssim_loss_value / num_images
        cosine_rgb_loss_value = cosine_rgb_loss_value / num_images
        hsv_loss_value = hsv_loss_value / num_images

        curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
                     hsv_loss_value + 10 * ssim_loss_value + 1e-6 * gradient_regulariser) / 6

        losses = namedtuple("losses", ['l1', 'rgb', 'ssim', 'cosine_rgb', 'hsv', 'curl'])
        output = losses(l1_loss_value, rgb_loss_value, ssim_loss_value,
                        cosine_rgb_loss_value, hsv_loss_value, curl_loss)

        return output

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3,
                              stride=stride, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        return out

class CURLLayer(nn.Module):
    def __init__(self, in_chans=64, out_chans=64):
        super(CURLLayer, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.init_net()

    def init_net(self):
        self.lab1 = ConvBlock(64, 64)
        self.lab2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lab3 = ConvBlock(64, 64)
        self.lab4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lab5 = ConvBlock(64, 64)
        self.lab6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lab7 = ConvBlock(64, 64)
        self.lab8 = nn.AdaptiveAvgPool2d(1)
        self.lab_fc = nn.Linear(64, 48)

        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        self.dp3 = nn.Dropout(0.5)

        self.rgb1 = ConvBlock(64, 64)
        self.rgb2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rgb3 = ConvBlock(64, 64)
        self.rgb4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rgb5 = ConvBlock(64, 64)
        self.rgb6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rgb7 = ConvBlock(64, 64)
        self.rgb8 = nn.AdaptiveAvgPool2d(1)
        self.rgb_fc = nn.Linear(64, 48)

        self.hsv1 = ConvBlock(64, 64)
        self.hsv2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hsv3 = ConvBlock(64, 64)
        self.hsv4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hsv5 = ConvBlock(64, 64)
        self.hsv6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hsv7 = ConvBlock(64, 64)
        self.hsv8 = nn.AdaptiveAvgPool2d(1)
        self.hsv_fc = nn.Linear(64, 64)

    def forward(self, x):
        x.contiguous() # remove memory holes
        feat = x[:, 3:64, :, :]
        img = x[:,0:3, :, :]

        torch.cuda.empty_cache()
        shape = x.shape

        img_clamped = torch.clamp(img, 0, 1)
        # ndim=3
        img_lab = torch.clamp(
            ImageProcessing.rgb_to_lab(img_clamped.squeeze(0)), 0, 1
        )
        # ndim=4, cat follow channel
        feat_lab = torch.cat((feat, img_lab.unsqueeze(0)), 1)

        x = self.lab1(feat_lab)
        del feat_lab
        x = self.lab2(x)
        x = self.lab3(x)
        x = self.lab4(x)
        x = self.lab5(x)
        x = self.lab6(x)
        x = self.lab7(x)
        x = self.lab8(x)
        x = x.view(x.size()[0], -1)
        x = self.dp1(x)
        L = self.lab_fc(x)

        # curl adjustment in lab space
        img_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(img_lab, L[0, 0:48])
        # ndim=3
        img_rgb = ImageProcessing.lab_to_rgb(img_lab)
        img_rgb = torch.clamp(img_rgb, 0, 1)

        # ndim=4
        feat_rgb = torch.cat((feat, img_rgb.unsqueeze(0)), 1)
        x = self.rgb1(feat_rgb)
        del feat_rgb
        x = self.rgb2(x)
        x = self.rgb3(x)
        x = self.rgb4(x)
        x = self.rgb5(x)
        x = self.rgb6(x)
        x = self.rgb7(x)
        x = self.rgb8(x)
        x = x.view(x.size()[0], -1)
        x = self.dp2(x)
        R = self.rgb_fc(x)

        # ndim=3
        img_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(
            img_rgb, R[0, 0:48]
        )
        # ndim=3
        img_rgb = torch.clamp(img_rgb, 0, 1)

        # ndim=3
        img_hsv = ImageProcessing.rgb_to_hsv(img_rgb)
        img_hsv = torch.clamp(img_hsv, 0, 1)
        feat_hsv = torch.cat((feat, img_hsv.unsqueeze(0)), 1)

        x = self.hsv1(feat_hsv)
        del feat_hsv
        x = self.hsv2(x)
        x = self.hsv3(x)
        x = self.hsv4(x)
        x = self.hsv5(x)
        x = self.hsv6(x)
        x = self.hsv7(x)
        x = self.hsv8(x)
        x = x.view(x.size()[0], -1)
        x = self.dp3(x)
        H = self.hsv_fc(x)

        # ndim=3
        img_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(img_hsv, H[0, 0:64])
        img_hsv = torch.clamp(img_hsv, 0, 1)

        # ndim=3
        img_residual = torch.clamp(ImageProcessing.hsv_to_rgb(img_hsv), 0, 1)
        # ndim=4
        img = torch.clamp(img + img_residual.unsqueeze(0), 0, 1)

        gradient_regulariser = gradient_regulariser_rgb +\
            gradient_regulariser_hsv + gradient_regulariser_lab

        return img, gradient_regulariser

class CURLNet(nn.Module):
    def __init__(self):
        super(CURLNet, self).__init__()
        self.tednet = TEDModle()
        self.curllayer = CURLLayer()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        feat = self.tednet(x)
        img, gradient_regulariser = self.curllayer(feat)
        return img, gradient_regulariser