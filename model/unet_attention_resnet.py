import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class UpAttentionBlock(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, out_channels):
        super(UpAttentionBlock, self).__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(F_g=out_channels, F_l=skip_in_channels, F_int=out_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip_connection):
        x = self.up(x)
        x = F.interpolate(x, size=skip_connection.size()[2:], mode='bilinear', align_corners=False)
        skip_connection = self.attention(g=x, x=skip_connection)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, backbone='resnet34', pretrained=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 加载预训练的 ResNet 作为编码器
        if backbone == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError(f'Backbone {backbone} is not implemented')

        # 输入通道调整
        if n_channels != 3:
            self.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 编码器层
        self.encoder_layers = [
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]

        # 解码器层，带有注意力机制
        self.center = nn.Sequential(
            nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True),
        )

        self.up4 = UpAttentionBlock(filters[4], filters[3], filters[3])
        self.up3 = UpAttentionBlock(filters[3], filters[2], filters[2])
        self.up2 = UpAttentionBlock(filters[2], filters[1], filters[1])
        self.up1 = UpAttentionBlock(filters[1], filters[0], filters[0])
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[0], kernel_size=2, stride=2),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )

        self.outc = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        x0 = self.encoder_layers[0](x)  # 64通道
        x1 = self.encoder_layers[1](x0) # 64通道
        x2 = self.encoder_layers[2](x1) # 128通道
        x3 = self.encoder_layers[3](x2) # 256通道
        x4 = self.encoder_layers[4](x3) # 512通道（对于 ResNet50，为 2048）

        # 中心部分
        x_center = self.center(x4)

        # 解码器部分，带有注意力机制
        x = self.up4(x_center, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.up0(x)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    model = UNet(n_channels=1, n_classes=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(y.shape)  # 输出应为 torch.Size([1, 4, 256, 256])
