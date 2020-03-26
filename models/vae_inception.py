import torch
from torch.nn import functional as F
from torch import nn
from utils.torch import flatten, un_flatten, reparameterize


#
# Inception V3 cell architectures are based on PyTorch's Inception classifier
# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
#

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionVAE(nn.Module):
    def __init__(self):
        super(InceptionVAE, self).__init__()

        self.encoder = nn.Sequential(
            # 3 x 96 x 96 -> 32 x 48 x 48
            BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1),
            # 32 x 48 x 48 -> 32 x 46 x 46
            BasicConv2d(32, 32, kernel_size=3),
            # 32 x 46 x 46 -> 64 x 46 x 46
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            # 64 x 46 x 46 -> 64 x 23 x 23
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64 x 23 x 23 -> 80 x 23 x 23
            BasicConv2d(64, 80, kernel_size=1),
            # 80 x 23 x 23 -> 192 x 21 x 21
            BasicConv2d(80, 192, kernel_size=3),
            # 192 x 21 x 21 -> 256 x 21 x 21
            InceptionA(192, pool_features=32),
            # 256 x 21 x 21 -> 288 x 21 x 21
            InceptionA(256, pool_features=64),
            # 288 x 21 x 21 -> 288 x 21 x 21
            InceptionA(288, pool_features=64),
            # 288 x 21 x 21 -> 768 x 10 x 10
            InceptionB(288),
            # 768 x 10 x 10 -> 768 x 10 x 10
            InceptionC(768, channels_7x7=128),
            # 768 x 10 x 10 -> 768 x 10 x 10
            InceptionC(768, channels_7x7=160),
            # 768 x 10 x 10 -> 768 x 10 x 10
            InceptionC(768, channels_7x7=160),
            # 768 x 10 x 10 -> 768 x 10 x 10
            InceptionC(768, channels_7x7=192),
            # 768 x 10 x 10 -> 1280 x 4 x 4
            InceptionD(768),
            # 1280 x 4 x 4 -> 2048 x 4 x 4
            InceptionE(1280),
            # 2048 x 4 x 4 -> 2048 x 4 x 4
            InceptionE(2048),
            # 2048 x 4 x 4 -> 2048 x 1 x 1
            nn.MaxPool2d(kernel_size=4),
        )

        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(2048, 64)
        self.fc3 = nn.Linear(64, 2048)

        self.decoder = nn.Sequential(
            # 2048 x 1 x 1 -> 1024 x 4 x 4
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # 1024 x 4 x 4 -> 512 x 6 x 6
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 512 x 6 x 6 -> 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256 x 8 x 8 -> 128 x 10 x 10
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 10 x 10 -> 64 x 30 x 30
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 30 x 30 -> 32 x 32 x 32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 32 x 32 -> 3 x 96 x 96
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = flatten(x)

        mu, log_var = self.fc1(x), self.fc2(x)
        z = reparameterize(mu, log_var)
        x = self.fc3(z)

        x = un_flatten(x, channels=2048, h=1, w=1)
        x = self.decoder(x)

        return x, mu, log_var
