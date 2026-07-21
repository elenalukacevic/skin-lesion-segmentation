"""
dataset.py
----------
PyTorch Dataset class and DataLoader builder for
the Melanoma Detection Dataset.
Handles loading, resizing, augmentation, and normalization.
"""

import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from config import CFG


def _get_paths():
    """
    Return sorted image and mask file paths.
    Uses preprocessed clean images if available, otherwise originals.
    """
    img_dir = CFG["clean_dir"] if os.path.exists(CFG["clean_dir"]) \
              else CFG["image_dir"]

    imgs  = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    masks = sorted([f for f in os.listdir(CFG["mask_dir"])
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    assert len(imgs) == len(masks), \
        f"Image count ({len(imgs)}) != mask count ({len(masks)})"

    img_paths  = [os.path.join(img_dir,         f) for f in imgs]
    mask_paths = [os.path.join(CFG["mask_dir"], f) for f in masks]
    return img_paths, mask_paths


def _split_indices(n):
    """
    Split dataset indices into train/val/test sets.
    Ratios defined in CFG: 75% train, 15% val, 10% test.
    """
    n_test  = int(n * CFG["test_split"])
    n_val   = int(n * CFG["val_split"])
    n_train = n - n_test - n_val
    idx     = list(range(n))
    random.shuffle(idx)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


class DermDataset(Dataset):
    """
    PyTorch Dataset for dermoscopic image segmentation.
    Loads image-mask pairs, applies resizing, optional augmentation,
    and ImageNet normalization.
    """

    def __init__(self, img_paths, mask_paths, augment=False):
        self.imgs    = img_paths
        self.masks   = mask_paths
        self.augment = augment
        self.sz      = CFG["img_size"]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img  = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.masks[i]).convert("L")

        # Resize to target resolution
        img  = TF.resize(img,  [self.sz, self.sz])
        mask = TF.resize(mask, [self.sz, self.sz],
                         interpolation=TF.InterpolationMode.NEAREST)

        # Data augmentation (training only)
        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-35, 35)
            img   = TF.rotate(img,  angle)
            mask  = TF.rotate(mask, angle)
            if random.random() > 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
            if random.random() > 0.4:
                img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
            if random.random() > 0.4:
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))

        # Normalize using ImageNet mean and std
        img  = TF.normalize(TF.to_tensor(img),
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        # Binarize mask: pixel > 0.5 -> lesion (1), else background (0)
        mask = (TF.to_tensor(mask) > 0.5).float()
        return img, mask


def build_dl_loaders():
    """
    Build train, validation, and test DataLoaders.
    Returns loaders and test set metadata for visualization.
    num_workers=0 required on macOS (MPS + Python 3.13 compatibility).
    """
    img_paths, mask_paths = _get_paths()
    n = len(img_paths)
    train_idx, val_idx, test_idx = _split_indices(n)

    def subset(idx, aug):
        return DermDataset(
            [img_paths[i]  for i in idx],
            [mask_paths[i] for i in idx],
            augment=aug
        )

    print(f"Dataset: {len(train_idx)} train | "
          f"{len(val_idx)} val | {len(test_idx)} test")

    train_dl = DataLoader(subset(train_idx, True),  CFG["batch_size"],
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(subset(val_idx,   False), CFG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(subset(test_idx,  False), CFG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=False)

    return train_dl, val_dl, test_dl, test_idx, img_paths, mask_paths
