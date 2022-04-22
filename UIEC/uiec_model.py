import torch
import torch.nn as nn

from utils.colorspace import HSV2RGB, RGB2HSV


class UIEC(nn.Module):
    def __init__(self, pretrained=None):
        super(UIEC, self).__init__()
        self.rgb2hsv = RGB2HSV()
        self.hsv2rgb = HSV2RGB()
        self.pretrained = pretrained

        self.block_rgb1 = BlockRGB(3, 64)
        self.block_rgb2 = BlockRGB(64, 64)
        self.block_rgb3 = BlockRGB(64, 64)
        self.block_rgb4 = BlockRGB(64, 64)
        self.block_rgb5 = BlockRGB(64, 64)
        self.block_rgb6 = BlockRGB(64, 64)
        self.block_rgb7 = BlockRGB(64, 64)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        # HSV Global-Adjust Block
        self.block_hsv1 = BlockHSV(6, 64)
        self.block_hsv2 = BlockHSV(64, 64)
        self.block_hsv3 = BlockHSV(64, 64)
        self.block_hsv4 = BlockHSV(64, 64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.avagepool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 44)

        # Attention Map Block
        self.block_att1 = BlockRGB(9, 64)
        self.block_att2 = BlockRGB(64, 64)
        self.block_att3 = BlockRGB(64, 64)
        self.block_att4 = BlockRGB(64, 64)
        self.block_att5 = BlockRGB(64, 64)
        self.block_att6 = BlockRGB(64, 64)
        self.block_att7 = BlockRGB(64, 64)
        self.conv3 = nn.Conv2d(64, 6, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(6)


    def init_weight(self):
        if self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        out_rgb = self.block_rgb1(x)
        out_rgb = self.block_rgb2(out_rgb)
        out_rgb = self.block_rgb3(out_rgb)
        out_rgb = self.block_rgb4(out_rgb)
        out_rgb = self.block_rgb5(out_rgb)
        out_rgb = self.block_rgb6(out_rgb)
        out_rgb = self.block_rgb7(out_rgb)
        out_rgb = self.bn1(self.conv1(out_rgb))
        out_rgb = torch.sigmoid(out_rgb)
        # option 1, using 0:3 feature map
        out_rgb = out_rgb[:, 0:3, :, :]
        hsv_from_out = self.rgb2hsv(out_rgb)

        hsv_input = torch.cat([hsv_from_out, hsv_from_out], dim=1)
        batch_size =hsv_input.size()[0]

        out_hsv = self.block_hsv1(hsv_input)
        out_hsv = self.block_hsv2(out_hsv)
        out_hsv = self.block_hsv3(out_hsv)
        out_hsv = self.block_hsv4(out_hsv)
        out_hsv = self.lrelu1(self.bn2(self.conv2(out_hsv)))
        out_hsv = self.avagepool(out_hsv).view(batch_size, -1) #flatten, to fc
        out_hsv = self.fc(out_hsv)
        H, S, V, H2S = torch.split(out_hsv, 11, dim=1)
        H_in, S_in, V_in = hsv_input[:, 0:1, :, :], hsv_input[:, 1:2, :, :], hsv_input[:, 2:3, :, :]

        H_out = piece_function_org(H_in, H, 11)
        S_out1 = piece_function_org(S_in, S, 11)
        V_out = piece_function_org(V_in, V, 11)

        S_out2 = piece_function_org(H_in, H2S, 11)
        S_out = (S_out1 + S_out2) / 2

        zero_lab = torch.zeros(S_out.shape).cuda()
        s_t = torch.where(S_out < 0, zero_lab, S_out)
        one_lab = torch.ones(S_out.shape).cuda()
        S_out = torch.where(s_t > 1, one_lab, s_t)

        zero_lab = torch.zeros(V_out.shape).cuda()
        s_t = torch.where(V_out < 0, zero_lab, V_out)
        one_lab = torch.ones(V_out.shape).cuda()
        V_out = torch.where(s_t > 1, one_lab, s_t)

        hsv_out = torch.cat([H_out, S_out, V_out], dim=1)
        curve = torch.cat([H.view(batch_size, 1, -1),
                           S.view(batch_size, 1, -1),
                           V.view(batch_size, 1, -1),
                           H2S.view(batch_size, 1, -1)], dim=1)

        hsv_out_rgb = self.hsv2rgb(hsv_out)
        confindencenet_input = torch.cat([x,
                                          out_rgb,
                                          hsv_out_rgb], dim=1)

        out_att = self.block_att1(confindencenet_input)
        out_att = self.block_att2(out_att)
        out_att = self.block_att3(out_att)
        out_att = self.block_att4(out_att)
        out_att = self.block_att5(out_att)
        out_att = self.block_att6(out_att)
        out_att = self.block_att7(out_att)
        out_att = self.bn3(self.conv3(out_att))
        out_att = torch.sigmoid(out_att)

        confindence_rgb = out_att[:, 0:3, :, :]
        confindence_hsv = out_att[:, 3:6, :, :]

        result = 0.5 * confindence_rgb * out_rgb + \
                         0.5 * confindence_hsv * hsv_out_rgb

        return result
        # option 2, using 1x1 convolution to fusion channel before exchanging


class BlockRGB(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(BlockRGB, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_chans)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        return out

class BlockHSV(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(BlockHSV, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_chans)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = self.lrelu(self.bn(self.conv(x)))
        out = self.pooling(out)
        return out

def piece_function_org(x_m, para_m, M):
    b, c, h, w = x_m.shape
    r_m = para_m[:, 0].view(b, c, 1, 1).expand(b, c, h, w)
    for i in range(M-1):
        para = (para_m[:, i + 1] - para_m[:, i]).view(b, c, 1, 1).expand(b, c, h, w)
        r_m = r_m + para * \
              sgn_m(M * x_m - i * torch.ones(x_m.shape).cuda())
    return r_m

def sgn_m(x):
    # x = torch.Tensor(x)
    zero_lab = torch.zeros(x.shape).cuda()
    # print("one_lab",one_lab)
    s_t = torch.where(x < 0, zero_lab, x)
    one_lab = torch.ones(x.shape).cuda()
    s = torch.where(s_t > 1, one_lab, s_t)
    return s