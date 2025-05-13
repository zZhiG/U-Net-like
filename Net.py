import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Conv1x1(nn.Module):
    def __init__(self, in_, out_):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class Conv3x3(nn.Module):
    def __init__(self, in_, out_):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class Down2(nn.Module):
    def __init__(self, in_, out_, h, w):
        super(Down2, self).__init__()
        self.conv1 = Conv3x3(in_, out_)
        self.conv2 = Conv3x3(out_, out_)

        self.proj = Conv1x1(out_*2, out_)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        y = torch.cat([x1, x2], dim=1)
        y = self.proj(y)

        return x2, y


class Down3(nn.Module):
    def __init__(self, in_, out_, h, w):
        super(Down3, self).__init__()
        self.conv1 = Conv3x3(in_, out_)
        self.conv2 = Conv3x3(out_, out_)
        self.conv3 = Conv3x3(out_, out_)

        self.proj = Conv1x1(out_*3, out_)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.proj(y)
        return x3, y

class Up2(nn.Module):
    def __init__(self, in_, out_, h, w):
        super(Up2, self).__init__()
        self.conv1 = Conv3x3(in_, in_)
        self.conv2 = Conv3x3(in_, out_)

        self.proj = Conv1x1(in_*2, in_)

    def forward(self, x, u):
        up = torch.cat([x, u], dim=1)
        up = self.proj(up)
        x1 = self.conv1(up)
        x2 = self.conv2(x1)
        return x2

class Up3(nn.Module):
    def __init__(self, in_, out_, h, w):
        super(Up3, self).__init__()
        self.conv1 = Conv3x3(in_, in_)
        self.conv2 = Conv3x3(in_, in_)
        self.conv3 = Conv3x3(in_, out_)

        self.proj = Conv1x1(in_*2, in_)

    def forward(self, x, u):
        up = torch.cat([x, u], dim=1)
        up = self.proj(up)
        x1 = self.conv1(up)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.down1 = Down2(1, 32, 512, 512)
        self.down2 = Down2(32, 64, 256, 256)
        self.down3 = Down3(64, 128, 128, 128)
        self.down4 = Down3(128, 256, 64, 64)
        self.down5 = Down3(256, 512, 32, 32)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool = nn.MaxUnpool2d(2, 2)

        self.up1 = Up3(512, 256, 32, 32)
        self.up2 = Up3(256, 128, 64, 64)
        self.up3 = Up3(128, 64, 128, 128)
        self.up4 = Up2(64, 32, 256, 256)

        self.res1 = Conv1x1(1, 32)
        self.res2 = Conv1x1(32, 64)
        self.res3 = Conv1x1(64, 128)
        self.res4 = Conv1x1(128, 256)
        self.res5 = Conv1x1(256, 512)

        self.res6 = Conv1x1(512, 256)
        self.res7 = Conv1x1(256, 128)
        self.res8 = Conv1x1(128, 64)
        self.res9 = Conv1x1(64, 32)

        self.proj = Conv1x1(64, 32)

        self.final = nn.Sequential(Conv3x3(32, 32),
                                   nn.Conv2d(32, 2, kernel_size=5, padding=2))

    def forward(self, x):
        x1, u1 = self.down1(x)
        x1 = self.res1(x) + x1
        x11, i1 = self.pool(x1)
        x2, u2 = self.down2(x11)
        x2 = self.res2(x11) + x2
        x22, i2 = self.pool(x2)
        x3, u3 = self.down3(x22)
        x3 = self.res3(x22) + x3
        x33, i3 = self.pool(x3)
        x4, u4 = self.down4(x33)
        x4 = self.res4(x33) + x4
        x44, i4 = self.pool(x4)
        x5, u5 = self.down5(x44)
        x5 = self.res5(x44) + x5
        x55, i5 = self.pool(x5)

        y5 = self.unpool(x55, indices=i5)
        y55 = self.up1(y5, u5)
        y55 = self.res6(y5) + y55
        y4 = self.unpool(y55, indices=i4)
        y44 = self.up2(y4, u4)
        y44 = self.res7(y4) + y44
        y3 = self.unpool(y44, indices=i3)
        y33 = self.up3(y3, u3)
        y33 = self.res8(y3) + y33
        y2 = self.unpool(y33, indices=i2)
        y22 = self.up4(y2, u2)
        y22 = self.res9(y2) + y22
        y1 = self.unpool(y22, indices=i1)

        up = torch.cat([y1, u1], dim=1)
        up = self.proj(up)
        out = self.final(up)
        return out

if __name__ == '__main__':
    pass