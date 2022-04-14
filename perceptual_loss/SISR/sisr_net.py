import torch.nn as nn
import torch

class SISRNet(nn.Module):
    def __init__(self):
        super(SISRNet, self).__init__()
        # conv
        self.conv1 = Convlayer(3, 64, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        # resblock
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        self.res4 = ResBlock(64)
        # deconv
        self.deconv1 = UpSample(64, 64, kernel_size=3, stride=1, upsample=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpSample(64, 64, kernel_size=3, stride=1, upsample=2)
        self.in3 = nn.InstanceNorm2d(64, affine=True)
        self.deconv3 = Convlayer(64, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        out = self.relu(self.in2(self.deconv1(out)))
        out = self.relu(self.in3(self.deconv2(out)))
        out = self.deconv3(out)

        return out

class Convlayer(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride):
        super(Convlayer, self).__init__()
        reflect = kernel_size // 2
        self.reflect_padding = nn.ReflectionPad2d(reflect)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflect_padding(x)
        out = self.conv(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = Convlayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = Convlayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + x

class UpSample(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, upsample=None):
        super(UpSample, self).__init__()
        self.upsample = upsample
        reflect = kernel_size // 2
        self.reflect_padding = nn.ReflectionPad2d(reflect)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflect_padding(x_in)
        out = self.conv(out)
        return out


# if __name__ == '__main__':
#     model = SISRNet()
#
#     numel_list = [p.numel() for p in model.parameters()]
#     print(sum(numel_list))
#
#     x = torch.randn((1, 3, 72, 72))
#     print(x.shape)
#     out = model(x)
#     print(out.shape)
