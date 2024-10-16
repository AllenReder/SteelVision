import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # 用于引入 Swin Transformer

class OutConv(nn.Module):
    """1x1 Convolution at the output"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
class SwinBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, depths, num_heads, window_size):
        super(SwinBlock, self).__init__()
        # 确保 depths 和 num_heads 是列表或元组
        if isinstance(depths, int):
            depths = [depths]
        if isinstance(num_heads, int):
            num_heads = [num_heads]

        self.swin = timm.create_model(
            'swinv2_small_window8_256',
            pretrained=False,
            in_chans=in_channels,
            num_classes=0,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            img_size=None  # 允许任意输入尺寸
        )

    def forward(self, x):
        x = self.swin.patch_embed(x)
        x = self.swin.forward_features(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, depths, num_heads, window_size):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.swin_block = SwinBlock(in_channels, embed_dim, depths, num_heads, window_size)
        self.conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.swin_block(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, depths, num_heads, window_size, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1x1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.swin_block = SwinBlock(in_channels, embed_dim, depths, num_heads, window_size)
        self.conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if hasattr(self, 'conv1x1'):
            x1 = self.conv1x1(x1)
        # 调整尺寸以匹配跳跃连接
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.swin_block(x)
        x = self.conv(x)
        return x

class SwinUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, bilinear=False):
        super(SwinUNet, self).__init__()
        window_size = 8  # 根据需要调整
        # Swin Transformer V2 参数
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.down1 = Down(embed_dim, embed_dim * 2, embed_dim, depths[0], num_heads[0], window_size)
        self.down2 = Down(embed_dim * 2, embed_dim * 4, embed_dim * 2, depths[1], num_heads[1], window_size)
        self.down3 = Down(embed_dim * 4, embed_dim * 8, embed_dim * 4, depths[2], num_heads[2], window_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(embed_dim * 8, embed_dim * 16 // factor, embed_dim * 8, depths[3], num_heads[3], window_size)

        self.up1 = Up(embed_dim * 16, embed_dim * 8 // factor, embed_dim * 8, depths[3], num_heads[3], window_size, bilinear=bilinear)
        self.up2 = Up(embed_dim * 8, embed_dim * 4 // factor, embed_dim * 4, depths[2], num_heads[2], window_size, bilinear=bilinear)
        self.up3 = Up(embed_dim * 4, embed_dim * 2 // factor, embed_dim * 2, depths[1], num_heads[1], window_size, bilinear=bilinear)
        self.up4 = Up(embed_dim * 2, embed_dim, embed_dim, depths[0], num_heads[0], window_size, bilinear=bilinear)
        self.outc = OutConv(embed_dim, out_channels)

    def forward(self, x):
        x1 = self.inc(x)       # x1: 原始尺寸
        x2 = self.down1(x1)    # x2: 原始尺寸 / 2
        x3 = self.down2(x2)    # x3: 原始尺寸 / 4
        x4 = self.down3(x3)    # x4: 原始尺寸 / 8
        x5 = self.down4(x4)    # x5: 原始尺寸 / 16
        x = self.up1(x5, x4)   # x: 原始尺寸 / 8
        x = self.up2(x, x3)    # x: 原始尺寸 / 4
        x = self.up3(x, x2)    # x: 原始尺寸 / 2
        x = self.up4(x, x1)    # x: 原始尺寸
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    model = SwinUNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    x = torch.randn(1, 1, 256, 256)  # 确保输入尺寸为 256x256
    y = model(x)
    print(y.shape)  # 应该输出 torch.Size([1, 4, 256, 256])
