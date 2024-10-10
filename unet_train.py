import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tool import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.unet import UNet
from dataset import SteelDataset

# 设置数据路径
images_path = './data/images/training'
masks_path = './data/annotations/training'
image_files = sorted([os.path.join(images_path, f) for f in os.listdir(
    images_path) if f.endswith(('.png', '.jpg'))])
mask_files = sorted([os.path.join(masks_path, f)
                    for f in os.listdir(masks_path) if f.endswith(('.png', '.jpg'))])

# 将数据集划分为训练集和验证集
train_images, val_images, train_masks, val_masks = train_test_split(
    image_files, mask_files, test_size=0.1, random_state=42)

# 数据集和数据加载器
train_dataset = SteelDataset(train_images, train_masks)
val_dataset = SteelDataset(val_images, val_masks)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = UNet().to(device)

# 输出模型参数量大小
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {total_params / 1e6:.2f}M")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化学习率调度器
# scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# 训练模型
epochs = 20
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for (images, masks) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss:.4f}")

    # 更新学习率
    # scheduler.step()
    # print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}, Validation Loss: {epoch_val_loss:.4f}")
    # 保存模型
    torch.save(model.state_dict(), f'./saved_model/unet_epoch_{epoch+1}.pth')

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'./train_loss.png')
    plt.close()
