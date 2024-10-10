import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2

class SteelDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Image or mask not found for index {idx}")

        # Resize image and mask to 256x256
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).long()

        image = image.unsqueeze(0)

        # print(f"Image tensor shape: {image.shape}, Channels: {image.shape[0]}")
        # print(f"Mask tensor shape: {mask.shape}, Channels: {mask.shape[0]}")

        return image, mask