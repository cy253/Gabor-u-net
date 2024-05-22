import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        avg = avg.view(avg.size(0), -1)
        channel_att = self.sigmoid(self.fc(avg).view(avg.size(0), -1, 1, 1, 1))
        return x * channel_att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.sigmoid(self.conv(pool))
        return x * spatial_att

class CBAMModule(nn.Module):
    def __init__(self, in_channels):
        super(CBAMModule, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            GaModule(in_channels, out_channels),
            CBAMModule(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GaborConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GaborConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.init_gabor_weights(kernel_size)

    def init_gabor_weights(self, kernel_size):
        # 创建Gabor卷积核
        sigma = 1.0
        theta = np.pi / 4
        lambda_ = 1.0
        gamma = 0.5
        phi = 0.0
        x0 = y0 = z0 = kernel_size // 2

        for x in range(kernel_size):
            for y in range(kernel_size):
                for z in range(kernel_size):
                    x_prime = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
                    y_prime = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
                    z_prime = z - z0
                    self.conv.weight.data[:, :, x, y, z] = torch.tensor(
                        np.exp(-(x_prime**2 + gamma**2 * y_prime**2 + gamma**2 * z_prime**2) / (2 * sigma**2))
                        * np.cos(2 * np.pi * x_prime / lambda_ + phi))

    def forward(self, x):
        return self.conv(x)


class GaModule(nn.Module):
    def __init__(self, in_channels, out_channels):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = out_channels // 2  # hidden channels
        self.cv1 = GaborConv3d(in_channels, c_)
        self.cv2 = DoubleConv(c_, c_)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes

        self.inc = GaModule(in_channels, 64)
        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # gabor_output = self.gabor_filter(x)
        # x4 = x4 + gabor_output  # Combine U-Net feature with Gabor filter output

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # 创建 VNet 实例
    x = torch.randn(1, 4, 160, 160, 128)
    net = Modified3DUNet(in_channels=4, num_classes=4)
    y = net(x)

    # 打印网络结构
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)