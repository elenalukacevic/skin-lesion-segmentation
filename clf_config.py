"""
clf_config.py
-------------
Configuration for the two-stage lesion CLASSIFICATION task
(HAM10000 / ISIC2018).

Stage 1: the trained segmentation U-Net crops the lesion ROI.
Stage 2: an ImageNet-pretrained CNN classifies the crop into 7 classes.

Reuses DEVICE, seeding and results_dir from the segmentation config
so both tasks share one device/reproducibility setup.
"""

import os
from config import CFG, DEVICE  # noqa: F401  (re-exported for classification modules)


# ══════════════════════════════════════════════════════════════════
# CLASSIFICATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CLF = {
    # ── Paths ──────────────────────────────────────────────────────
    # Root under which the HAM10000/ISIC2018 dataset was extracted.
    # The loader auto-detects the metadata CSV and image folders below.
    "data_root"    : "data",
    "crop_dir"     : os.path.join("data", "ham_cropped"),
    "results_dir"  : CFG["results_dir"],
    # Trained segmentation U-Net used to localize the lesion (Stage 1).
    "unet_weights" : os.path.join(CFG["results_dir"], "unet_best.pth"),

    # ── Classes ────────────────────────────────────────────────────
    # HAM10000 diagnosis codes (order defines the label indices).
    "classes"      : ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],

    # ── Dataset split ratios (grouped by lesion_id) ────────────────
    "val_split"    : 0.15,
    "test_split"   : 0.15,

    # ── Stage 1: cropping ──────────────────────────────────────────
    "unet_img_size"  : CFG["img_size"],   # U-Net inference resolution (256)
    "crop_margin"    : 0.15,              # expand lesion bbox by 15 % each side
    "min_lesion_frac": 0.01,             # mask smaller than this -> center-crop fallback

    # ── Stage 2: classifier ────────────────────────────────────────
    "backbone"       : "efficientnet_b0",  # or "resnet50"
    "pretrained"     : True,
    "img_size"       : 224,
    "batch_size"     : 32,
    "epochs"         : 30,
    "lr_head"        : 1e-3,   # learning rate for the new classification head
    "lr_backbone"    : 1e-4,   # smaller LR for the pretrained backbone
    "weight_decay"   : 1e-4,
    "label_smoothing": 0.05,
}


# Full names for nicer reports / plots.
CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc"  : "Basal cell carcinoma",
    "bkl"  : "Benign keratosis",
    "df"   : "Dermatofibroma",
    "mel"  : "Melanoma",
    "nv"   : "Melanocytic nevi",
    "vasc" : "Vascular lesions",
}

os.makedirs(CLF["crop_dir"], exist_ok=True)
