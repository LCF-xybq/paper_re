import torch
import torch.nn as nn

class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()
        self.conv = nn.Conv2d(3, 3, 1, 1)
        self.relu = nn.ReLU()
        nn.init.constant_(self.conv.weight, 1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.relu(x)
        print('-------inplace = False--------')
        print(x[:,0,:,:])
        print(out2[:, 0, :, :])
        return out2

class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()
        self.conv = nn.Conv2d(3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.conv.weight, 1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.relu(x)
        print('-------inplace = True--------')
        print(x[:, 0, :, :])
        print(out2[:, 0, :, :])
        return out2

class F3(nn.Module):
    def __init__(self):
        super(F3, self).__init__()
        self.conv = nn.Conv2d(3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.conv.weight, 1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.relu(self.conv(x))
        print('-------inplace = True v2--------')
        print(x[:, 0, :, :])
        print(out[:, 0, :, :])
        return out

if __name__ == '__main__':

    img = torch.tensor([[[[-1.2, -2.1], [7.8, -5.2]],[[5.6, 0.0], [-0.1, 1.2]],[[-2.3, 3.5], [7.6, 6.9]]]])
    print(img[:,0,:,:], img.shape)

    model1 = F1()
    model2 = F2()
    model3 = F3()

    out1 = model1(img)
    img = torch.tensor([[[[-1.2, -2.1], [7.8, -5.2]], [[5.6, 0.0], [-0.1, 1.2]], [[-2.3, 3.5], [7.6, 6.9]]]])
    # print(img[:,0,:,:])
    # print(out1[:, 0, :, :])
    out2 = model2(img)
    img = torch.tensor([[[[-1.2, -2.1], [7.8, -5.2]], [[5.6, 0.0], [-0.1, 1.2]], [[-2.3, 3.5], [7.6, 6.9]]]])
    # print(img[:, 0, :, :])
    # print(out2[:, 0, :, :])
    out3 = model3(img)