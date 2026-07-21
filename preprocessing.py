"""
preprocessing.py
----------------
Image preprocessing pipeline for dermoscopic images.
Handles noise removal, hair removal, bubble removal,
and intensity normalization.
"""

import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from config import CFG


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Noise removal using Non-Local Means Denoising.
    Preserves lesion edges while reducing sensor noise.
    Parameters: filter strength=6, search window=21, patch size=7.
    """
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, 6, 6, 7, 21)


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """
    DullRazor-like hair removal:
      1. Black-hat morphological filter -> detects dark linear structures
      2. Thresholding -> binary hair mask
      3. Dilation -> expands mask to cover hair edges
      4. Telea inpainting -> reconstructs pixels under the mask
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask  = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dil_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask     = cv2.dilate(mask, dil_k, iterations=1)
    return cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)


def remove_bubbles(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gel bubble removal (bright circular artifacts):
      1. Convert to LAB color space -> isolate L (luminance) channel
      2. Threshold at 220 -> detect bright bubble pixels
      3. Dilation -> expand mask to cover full bubble area
      4. Telea inpainting -> reconstruct affected pixels
    """
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch    = lab[:, :, 0]
    _, mask = cv2.threshold(l_ch, 220, 255, cv2.THRESH_BINARY)
    dil_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.dilate(mask, dil_k, iterations=2)
    return cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)


def normalize_intensity(img_bgr: np.ndarray) -> np.ndarray:
    """
    Contrast normalization using CLAHE applied only to the L channel
    in LAB color space. Improves lesion-to-skin contrast without
    distorting color information. clip_limit=2.0, tile_grid=(8x8).
    """
    lab          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied sequentially:
    denoise -> remove_hair -> remove_bubbles -> normalize_intensity
    """
    img = denoise(img_bgr)
    img = remove_hair(img)
    img = remove_bubbles(img)
    img = normalize_intensity(img)
    return img


def run_preprocessing():
    """
    Apply the full preprocessing pipeline to all images and save
    results to the clean directory. Skips already processed images.
    Saves a visual before/after preview.
    """
    os.makedirs(CFG["clean_dir"], exist_ok=True)
    files = sorted([f for f in os.listdir(CFG["image_dir"])
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    print(f"\n{'─'*55}")
    print(f"PREPROCESSING — {len(files)} images")
    print(f"{'─'*55}")

    for fname in tqdm(files, desc="Preprocessing"):
        src = os.path.join(CFG["image_dir"], fname)
        dst = os.path.join(CFG["clean_dir"], fname)
        if os.path.exists(dst):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        cv2.imwrite(dst, preprocess_image(img))

    print(f"✓ Clean images saved -> {CFG['clean_dir']}")
    _save_preview(files)


def _save_preview(files, n=4):
    """Save a visual preview comparing original and preprocessed images."""
    sample = random.sample(files, min(n, len(files)))
    rows   = []

    for fname in sample:
        orig  = cv2.imread(os.path.join(CFG["image_dir"], fname))
        clean = cv2.imread(os.path.join(CFG["clean_dir"], fname))
        if orig is None or clean is None:
            continue
        orig_r  = cv2.resize(orig,  (200, 200))
        clean_r = cv2.resize(clean, (200, 200))
        rows.append(np.concatenate([orig_r, clean_r], axis=1))

    if not rows:
        return

    canvas  = np.concatenate(rows, axis=0)
    hdr     = np.zeros((28, 400, 3), dtype=np.uint8)
    hdr_pil = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(hdr_pil)
    draw.text((5,   7), "Original",     fill=(255, 255, 255))
    draw.text((205, 7), "Preprocessed", fill=(255, 255, 255))
    hdr    = cv2.cvtColor(np.array(hdr_pil), cv2.COLOR_RGB2BGR)
    canvas = np.concatenate([hdr, canvas], axis=0)

    path = os.path.join(CFG["results_dir"], "preprocessing_preview.png")
    cv2.imwrite(path, canvas)
    print(f"  Preview saved -> {path}")
