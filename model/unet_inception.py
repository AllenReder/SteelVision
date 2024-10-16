import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PixelShuffle_ICNR(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(PixelShuffle_ICNR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight)


class UnetBlock(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, out_channels):
        super(UnetBlock, self).__init__()
        self.shuf = PixelShuffle_ICNR(up_in_channels, up_in_channels // 2)
        self.bn = nn.BatchNorm2d(up_in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(up_in_channels // 2 + skip_in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, up_in, skip_in):
        up_out = self.shuf(up_in)
        up_out = self.bn(up_out)
        up_out = self.relu(up_out)

        if up_out.size()[2:] != skip_in.size()[2:]:
            skip_in = F.interpolate(skip_in, size=up_out.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([up_out, skip_in], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNetInception(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(UNetInception, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        inception = models.inception_v3(pretrained=True, aux_logits=True)

        # 编码器层
        self.encoder0 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
        )

        self.encoder1 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
        )

        self.encoder2 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
        )

        self.encoder3 = nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )

        self.encoder4 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        # 中心部分
        self.center = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        # 解码器层
        self.up4 = UnetBlock(2048, 2048, 1024)
        self.up3 = UnetBlock(1024, 768, 512)
        self.up2 = UnetBlock(512, 288, 256)
        self.up1 = UnetBlock(256, 192, 128)
        self.up0 = UnetBlock(128, 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 如果输入是灰度图像，重复通道以匹配预期的输入通道数
        if self.n_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        # 编码器
        x0 = self.encoder0(x)  # 大小约为 127x127
        x1 = self.encoder1(x0)  # 大小约为 61x61
        x2 = self.encoder2(x1)  # 大小约为 29x29
        x3 = self.encoder3(x2)  # 大小约为 13x13
        x4 = self.encoder4(x3)  # 大小约为 6x6

        # 中心部分
        x_center = self.center(x4)

        # 解码器
        x = self.up4(x_center, x4)  # 大小约为 12x12
        x = self.up3(x, x3)         # 大小约为 24x24
        x = self.up2(x, x2)         # 大小约为 48x48
        x = self.up1(x, x1)         # 大小约为 96x96
        x = self.up0(x, x0)         # 大小约为 192x192

        # 输出卷积
        logits = self.outc(x)       # 大小约为 192x192

        # 将输出调整到 256x256
        logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)

        return logits


if __name__ == '__main__':
    model = UNetInception()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(y.shape)  # 输出应为 torch.Size([1, 4, 256, 256])