import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteAttention(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        self.unet = UNet(n_channels=input_ch*2, n_classes=1)

    def forward(self, data1, data2):
        pred_path = self.unet(outer_concatenation(data1, data2))
        return pred_path.squeeze(1)

class ContrastiveLoss():
    def __init__(self, tau):
        self.tau = tau

    def __call__(self, pred_path, data1, data2, match):
        pred_path_t = pred_path.transpose(1, 2)
        pred_path = F.softmax(pred_path, dim=2)
        pred_path_t = F.softmax(pred_path_t, dim=2)
        g_data2 = torch.matmul(pred_path_t, data1)
        g_data1 = torch.matmul(pred_path, data2)

        dist1 = (data1 - g_data1).pow(2).view(data1.size(0), -1).mean(dim=1)
        dist2 = (data2 - g_data2).pow(2).view(data2.size(0), -1).mean(dim=1)

        loss1 = (match*dist1+(1-match)*F.relu(self.tau-dist1)).mean()
        loss2 = (match*dist2+(1-match)*F.relu(self.tau-dist2)).mean()

        return (loss1+loss2)/2, (dist1+dist2)/2
    
def outer_concatenation(x, y):
    y_expand = y.unsqueeze(1)
    x_expand = x.unsqueeze(2)
    y_repeat = y_expand.repeat(1, y_expand.shape[2], 1, 1)
    x_repeat = x_expand.repeat(1, 1, x_expand.shape[1], 1)
    outer_concat = torch.cat((x_repeat, y_repeat), 3)
    return outer_concat.permute(0, 3, 1, 2).contiguous()


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
