import torch
import torch.nn as nn

class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 3, 1, 1)
        self.relu2 = nn.ReLU()
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.constant_(self.conv2.weight, 1)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        print('------- one conv one relu --------')
        out1 = self.relu1(self.conv1(x))
        print(out1[:, 0, :, :])
        out2 = self.relu2(self.conv2(out1))
        print(out2[:, 0, :, :])

class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3, 1, 1)
        self.relu = nn.ReLU()
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.constant_(self.conv2.weight, 1)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        print('------- multi conv one relu --------')
        out1 = self.relu(self.conv1(x))
        print(out1[:, 0, :, :])
        out2 = self.relu(self.conv2(out1))
        print(out2[:, 0, :, :])

if __name__ == '__main__':

    img = torch.tensor([[[[-1.2, -2.1], [7.8, -5.2]],[[5.6, 0.0], [-0.1, 1.2]],[[-2.3, 3.5], [7.6, 6.9]]]])
    print(img[:,0,:,:], img.shape)

    model1 = F1()
    model2 = F2()

    print(model1)
    print(model2)

    model1(img)

    img = torch.tensor([[[[-1.2, -2.1], [7.8, -5.2]], [[5.6, 0.0], [-0.1, 1.2]], [[-2.3, 3.5], [7.6, 6.9]]]])
    model2(img)
