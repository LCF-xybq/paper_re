import torch.nn as nn

class UIEC(nn.Module):
    def __init__(self, pretrained=None):
        super(UIEC, self).__init__()
        self.block_rgb1 = BlockRGB(3, )



    def init_weight(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)

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