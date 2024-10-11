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

        # ICNR initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight)
        # ICNR initialization can be added here if needed


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
        # print("[DEBUG] up_out.shape:", up_out.shape, "skip_in.shape:", skip_in.shape)

        # 调整skip_in的尺寸，使其与up_out的尺寸匹配
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


class UNetResNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(UNetResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 使用预训练的ResNet34作为编码器
        resnet = models.resnet34(pretrained=True)

        # 提取ResNet的层
        self.encoder0 = nn.Sequential(
            resnet.conv1,  # 输出通道数：64
            resnet.bn1,
            resnet.relu,
        )
        self.encoder1 = nn.Sequential(
            resnet.maxpool,  # 输出通道数：64
            resnet.layer1,   # 输出通道数：64
        )
        self.encoder2 = resnet.layer2  # 输出通道数：128
        self.encoder3 = resnet.layer3  # 输出通道数：256
        self.encoder4 = resnet.layer4  # 输出通道数：512

        # 中心部分
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 解码器部分，使用UnetBlock
        self.up4 = UnetBlock(512, 512, 256)  # up_in_channels, skip_in_channels, out_channels
        self.up3 = UnetBlock(256, 256, 128)
        self.up2 = UnetBlock(128, 128, 64)
        self.up1 = UnetBlock(64, 64, 64)
        self.up0 = UnetBlock(64, 64, 32)

        # 最终输出层
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        # 编码器
        x0 = self.encoder0(x)   # 大小：H/2 x W/2
        x1 = self.encoder1(x0)  # 大小：H/4 x W/4
        x2 = self.encoder2(x1)  # 大小：H/8 x W/8
        x3 = self.encoder3(x2)  # 大小：H/16 x W/16
        x4 = self.encoder4(x3)  # 大小：H/32 x W/32

        # 中心部分
        x_center = self.center(x4)

        # 解码器，使用跳跃连接
        x = self.up4(x_center, x4)  # 输入通道：512，跳跃连接通道：512
        x = self.up3(x, x3)         # 输入通道：256，跳跃连接通道：256
        x = self.up2(x, x2)         # 输入通道：128，跳跃连接通道：128
        x = self.up1(x, x1)         # 输入通道：64，跳跃连接通道：64
        x = self.up0(x, x0)         # 输入通道：64，跳跃连接通道：64

        # 最终输出层
        logits = self.outc(x)
        return logits
