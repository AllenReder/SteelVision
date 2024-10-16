import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2


class SteelDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None, eval=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.eval = eval
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15)
        ])
        self.image_augmentations = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ])

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
        mask = mask.unsqueeze(0)

        if not self.eval:
            # Apply additional augmentations
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.augmentations(image)
            torch.manual_seed(seed)
            mask = self.augmentations(mask)
            # Apply image-specific augmentations
            image = self.image_augmentations(image)

        mask = mask.squeeze(0)

        return image, mask
