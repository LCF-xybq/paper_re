from collections import namedtuple
from torchvision import models
import torch.nn as nn

class Vgg16(nn.Module):
    """
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True) # relu1_1
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True) # relu1_2
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True) # relu2_1
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True) # relu2_2
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True) # relu3_1
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True) # relu3_2
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True) # relu3_3
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace=True) # relu4_1
      (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True) # relu4_2
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True) # relu4_3
    """
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False

    def forward(self, x):
        out = self.slice1(x)
        relu1_2 = out
        out = self.slice2(out)
        relu2_2 = out
        out = self.slice3(out)
        relu3_3 = out
        out = self.slice4(out)
        relu4_3 = out
        out_features = namedtuple("out_features", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        output = out_features(relu1_2, relu2_2, relu3_3, relu4_3)
        return output