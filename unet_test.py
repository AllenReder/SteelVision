import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tool import *
import time
from model.unet import UNet
from dataset import SteelDataset

# 测试数据集
test_images_path = './data/images/test'
test_masks_path = './data/annotations/test'
test_image_files = sorted([os.path.join(test_images_path, f) for f in os.listdir(
    test_images_path) if f.endswith(('.png', '.jpg'))])
test_mask_files = sorted([os.path.join(test_masks_path, f) for f in os.listdir(
    test_masks_path) if f.endswith(('.png', '.jpg'))])

# 数据集和数据加载器
test_dataset = SteelDataset(test_image_files, test_mask_files, eval=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = UNet().to(device)

# 输出模型参数量大小
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {total_params / 1e6:.2f}M")

# 加载模型
model.load_state_dict(torch.load(
    './saved_model/unet_augu_best.pth'))
model.eval()

# 测试模型
start_time = time.time()
TP = {1: 0, 2: 0, 3: 0}
FP = {1: 0, 2: 0, 3: 0}
FN = {1: 0, 2: 0, 3: 0}
with torch.no_grad():
    total_batches = len(test_loader)
    for batch_idx, (images, masks) in tqdm(enumerate(test_loader), total=total_batches, desc="测试集"):

        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        masks = masks.cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        if batch_idx % 100 == 0:
            show_images([images[0].cpu().squeeze(0), masks[0], preds[0]], ['image', 'mask', 'pred'],
                        save_path=f'./temp/test_{batch_idx}.png', show=False)

        for c in range(1, 4):
            TP[c] += np.sum((preds == c) & (masks == c))
            FP[c] += np.sum((preds == c) & (masks != c))
            FN[c] += np.sum((preds != c) & (masks == c))

end_time = time.time()
fps = len(test_loader) / (end_time - start_time)

iou = {1: 0, 2: 0, 3: 0}
# 计算 mIoU
for c in range(1, 4):
    iou[c] = TP[c] / (TP[c] + FP[c] + FN[c])
    print(f"Class {c} IoU: {iou[c]}")
mIoU = sum(iou.values()) / len(iou)
print(f"Mean IoU (mIoU): {mIoU:.4f}")
print(f"Frames Per Second (FPS): {fps:.2f}")
