import torch
import torch.nn as nn

__all__ = ['CNN']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, padding='same', padding_mode='zeros'):
        super(BasicBlock, self).__init__()
        self.kernel = kernel
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel, stride=stride,\
                                    padding = padding, padding_mode=padding_mode, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0, act_fun=nn.ReLU(),
                 mid_channels=32, out_channels=64,
                 ksize1 = 5, ksize2 = 5,
                 stride = 2):
        super(CNN, self).__init__()
        #
        self.mean = mean
        self.std = std
        self.act_fun = act_fun
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        self.stride = stride
        self.layers1 = BasicBlock(1,  mid_channels, kernel=(ksize1,ksize2), stride=self.stride, padding='same', padding_mode='circular')
        sksize1 = min(ksize1, 28//self.stride)
        sksize2 = min(ksize2, 28//self.stride)
        self.layers2 = BasicBlock(mid_channels, out_channels, kernel=(sksize1,sksize2), stride=self.stride, padding='same', padding_mode='circular')

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))

        fc = [torch.nn.Linear(4 * 4 * out_channels, 128), self.act_fun, torch.nn.Linear(128, 10)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.layers1(x)
        x = self.layers2(x)

        # pool and classify
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x

    def name(self):
        name = 'cnn'
        name += '-' + str(self.ksize1) + '-' + str(self.ksize2)
        return name