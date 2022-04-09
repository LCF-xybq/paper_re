import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # conv
        self.conv1 = Convlayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = Convlayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = Convlayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # resblock
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.res3 = ResBlock(128)
        self.res4 = ResBlock(128)
        self.res5 = ResBlock(128)
        # upsample
        self.deconv1 = UpSample(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpSample(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = Convlayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        out = self.relu(self.in4(self.deconv1(out)))
        out = self.relu(self.in5(self.deconv2(out)))
        out = self.deconv3(out)

        return out

class Convlayer(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride):
        super(Convlayer, self).__init__()
        reflect = kernel_size // 2
        self.reflect_padding = nn.ReflectionPad2d(reflect)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride)

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
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflect_padding(x_in)
        out = self.conv(out)
        return out

