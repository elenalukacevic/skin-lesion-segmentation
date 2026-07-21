"""
clf_crop.py
-----------
Stage 1 of the two-stage classifier: use the trained segmentation U-Net
to localize the lesion in each HAM10000/ISIC2018 image and cache a
tightly-cropped version to data/ham_cropped/.

Pipeline per image:
    original -> U-Net mask -> largest component -> bbox + margin (square)
             -> crop from the ORIGINAL-resolution image -> save.

Robust fallbacks:
    * No U-Net checkpoint  -> center-square crop (classifier still runs).
    * Empty / tiny mask    -> center-square crop for that image.

Mirrors the caching + preview idiom of preprocessing.run_preprocessing().
"""

import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

from config import CFG
from clf_config import CLF, DEVICE
from clf_dataset import load_metadata
from unet_model import UNet

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════
# U-NET LOADING
# ══════════════════════════════════════════════════════════════════

def _load_unet():
    """Load the trained U-Net, or return None if no checkpoint exists."""
    path = CLF["unet_weights"]
    if not os.path.exists(path):
        print(f"⚠ No U-Net checkpoint at '{path}'.")
        print("  Falling back to CENTER CROPS. Train the segmentation U-Net "
              "first (python skin-lesions-segmentation.py --phase train_unet) "
              "for lesion-aware cropping.")
        return None
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"✓ Loaded U-Net for cropping -> {path}")
    return model


@torch.no_grad()
def _predict_mask(model, img_bgr):
    """Return a full-resolution binary lesion mask (uint8 {0,1}) for img_bgr."""
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(
        (CLF["unet_img_size"], CLF["unet_img_size"]), Image.LANCZOS)
    x = TF.normalize(TF.to_tensor(pil), _MEAN, _STD).unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    small = (prob > 0.5).astype(np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


# ══════════════════════════════════════════════════════════════════
# BOUNDING BOX / CROP GEOMETRY
# ══════════════════════════════════════════════════════════════════

def _center_bbox(W, H):
    """Largest centered square box."""
    side = min(W, H)
    x0 = (W - side) // 2
    y0 = (H - side) // 2
    return x0, y0, x0 + side, y0 + side


def _lesion_bbox(mask, W, H):
    """
    Square bounding box around the largest connected component of `mask`,
    expanded by CLF['crop_margin']. Returns None if the lesion is too small.
    """
    n, _lbl, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    # component 0 is the background; pick the largest of the rest
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    if stats[k, cv2.CC_STAT_AREA] < CLF["min_lesion_frac"] * W * H:
        return None

    x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
    w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * CLF["crop_margin"])
    side = min(side, min(W, H))  # never larger than the image
    half = side / 2.0

    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x0 = max(0, min(x0, W - int(side)))
    y0 = max(0, min(y0, H - int(side)))
    s = int(side)
    return x0, y0, x0 + s, y0 + s


# ══════════════════════════════════════════════════════════════════
# CACHE BUILD
# ══════════════════════════════════════════════════════════════════

def run_cropping():
    """Crop every labeled image to the lesion ROI and cache under crop_dir."""
    os.makedirs(CLF["crop_dir"], exist_ok=True)
    df = load_metadata()
    model = _load_unet()

    print(f"\n{'─' * 55}")
    print(f"STAGE 1 — LESION CROPPING — {len(df)} images "
          f"({'U-Net' if model else 'center-crop fallback'})")
    print(f"{'─' * 55}")

    n_lesion, n_fallback = 0, 0
    for row in tqdm(df.itertuples(), total=len(df), desc="Cropping"):
        dst = os.path.join(CLF["crop_dir"], row.image_id + ".jpg")
        if os.path.exists(dst):
            continue
        img = cv2.imread(row.path)
        if img is None:
            continue
        H, W = img.shape[:2]

        box = None
        if model is not None:
            box = _lesion_bbox(_predict_mask(model, img), W, H)
        if box is None:
            box = _center_bbox(W, H)
            n_fallback += 1
        else:
            n_lesion += 1

        x0, y0, x1, y1 = box
        cv2.imwrite(dst, img[y0:y1, x0:x1])

    print(f"✓ Crops saved -> {CLF['crop_dir']}  "
          f"(lesion-cropped: {n_lesion}, fallback: {n_fallback})")
    _save_crop_preview(df)


def _save_crop_preview(df, n=4):
    """Save an original-vs-crop preview grid to results/."""
    sample = df.sample(min(n, len(df)), random_state=CFG["random_seed"]).itertuples()
    rows = []
    for row in sample:
        orig = cv2.imread(row.path)
        crop = cv2.imread(os.path.join(CLF["crop_dir"], row.image_id + ".jpg"))
        if orig is None or crop is None:
            continue
        orig_r = cv2.resize(orig, (200, 200))
        crop_r = cv2.resize(crop, (200, 200))
        rows.append(np.concatenate([orig_r, crop_r], axis=1))

    if not rows:
        return
    canvas = np.concatenate(rows, axis=0)
    hdr = np.zeros((28, 400, 3), dtype=np.uint8)
    hdr_pil = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(hdr_pil)
    draw.text((5, 7), "Original", fill=(255, 255, 255))
    draw.text((205, 7), "Lesion crop", fill=(255, 255, 255))
    hdr = cv2.cvtColor(np.array(hdr_pil), cv2.COLOR_RGB2BGR)
    out = np.concatenate([hdr, canvas], axis=0)

    path = os.path.join(CLF["results_dir"], "clf_crop_preview.png")
    cv2.imwrite(path, out)
    print(f"  Preview saved -> {path}")


if __name__ == "__main__":
    run_cropping()
