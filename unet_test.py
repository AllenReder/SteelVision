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


# 设置数据路径
images_path = './NEU_Seg-main/images/test'
masks_path = './NEU_Seg-main/annotations/test'
image_files = sorted([os.path.join(images_path, f) for f in os.listdir(
    images_path) if f.endswith(('.png', '.jpg'))])
mask_files = sorted([os.path.join(masks_path, f)
                    for f in os.listdir(masks_path) if f.endswith(('.png', '.jpg'))])

# 确保图像和掩膜的数量一致
if len(image_files) != len(mask_files):
    raise ValueError("Number of images and masks do not match.")

# 将数据集划分为训练集和验证集
train_images, val_images, train_masks, val_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42)

# 数据集和数据加载器
train_dataset = SteelDataset(train_images, train_masks)
val_dataset = SteelDataset(val_images, val_masks)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = UNet(num_classes=4).to(device)

# 加载模型
model.load_state_dict(torch.load(
    './saved_model/unet_epoch_65.pth'))
model.eval()

# 测试数据集
test_images_path = './NEU_Seg-main/images/test'
test_masks_path = './NEU_Seg-main/annotations/test'
test_image_files = sorted([os.path.join(test_images_path, f) for f in os.listdir(
    test_images_path) if f.endswith(('.png', '.jpg'))])
test_mask_files = sorted([os.path.join(test_masks_path, f) for f in os.listdir(
    test_masks_path) if f.endswith(('.png', '.jpg'))])

# 确保测试图像和掩膜的数量一致
if len(test_image_files) != len(test_mask_files):
    raise ValueError("Number of test images and masks do not match.")

# 数据集和数据加载器
test_dataset = SteelDataset(test_image_files, test_mask_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化每个类别的 IoU 总和和计数
class_iou = {1: 0, 2: 0, 3: 0}
class_counts = {1: 0, 2: 0, 3: 0}

# 测试模型
start_time = time.time()
with torch.no_grad():
    total_batches = len(test_loader)
    for batch_idx, (images, masks) in tqdm(enumerate(test_loader), total=total_batches, desc="测试集"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        masks = masks.cpu().numpy()
        # print(np.unique(preds), np.unique(masks))
        if batch_idx % 100 == 0:
            show_images([images[0].cpu().squeeze(0), masks[0], preds[0]], ['image', 'mask', 'pred'],
                        save_path=f'./temp/test_{batch_idx}.png')

        for c in range(1, 4):  # 假设类别是1，2，3
            TP = np.sum((preds == c) & (masks == c))
            FP = np.sum((preds == c) & (masks != c))
            FN = np.sum((preds != c) & (masks == c))
            # 计算 IoU 并更新字典
            denominator = TP + FP + FN
            iou = TP / denominator if denominator != 0 else 0
            if iou != 0:
                class_iou[c] += iou
                class_counts[c] += 1
            # if iou < 0.75:
            #     print("IoU: ", iou)
            #     print(f"{c} TP: {TP}, FP: {FP}, FN: {FN}")


end_time = time.time()
fps = len(test_loader) / (end_time - start_time)

# 计算每个类别的平均 IoU
for c in class_iou:
    if class_counts[c] > 0:
        class_iou[c] /= class_counts[c]

# 计算 mIoU
mIoU = sum(class_iou.values()) / len(class_iou)

for c in class_iou:
    print(f"Class {c} IoU: {class_iou[c]}")
print(f"Mean IoU (mIoU): {mIoU:.4f}")
print(f"Frames Per Second (FPS): {fps:.2f}")
