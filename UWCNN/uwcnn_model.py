import torch
import torch.nn as nn

class UWCNN(nn.Module):
    def __init__(self):
        super(UWCNN, self).__init__()
        self.block1 = Block(0)
        self.block2 = Block(1)
        self.block3 = Block(2)
        self.conv4 = nn.Conv2d(147, 3, 3, 1, 1)

        self.init_weight()

    def init_weight(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv4(out)
        return out + x

class Block(nn.Module):
    def __init__(self, idx):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(3 + idx * 48, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out = torch.cat([out1, out2, out3, x], dim=1)
        return out
