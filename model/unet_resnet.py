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


class UNetResNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(UNetResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        resnet = models.resnet34(pretrained=True)

        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.encoder1 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1,
        )
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.up4 = UnetBlock(512, 512, 256)
        self.up3 = UnetBlock(256, 256, 128)
        self.up2 = UnetBlock(128, 128, 64)
        self.up1 = UnetBlock(64, 64, 64)
        self.up0 = UnetBlock(64, 64, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x_center = self.center(x4)

        x = self.up4(x_center, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    model = UNetResNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    torch.save(model.state_dict(), './result/model.pth')
    from torchviz import make_dot
    x = torch.randn(1, 1, 200, 200) 
    y = model(x)
    make_dot(y, params=dict(model.named_parameters())).render("unet_resnet", format="png")