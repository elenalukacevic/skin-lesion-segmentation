"""
config.py
---------
Shared configuration dictionary and device setup
used across all modules in the pipeline.
"""

import os
import random
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CFG = {
    # Paths
    "image_dir"      : os.path.join("data", "images"),
    "mask_dir"       : os.path.join("data", "masks"),
    "clean_dir"      : os.path.join("data", "images_clean"),
    "results_dir"    : "results",

    # Dataset split ratios
    "val_split"      : 0.15,
    "test_split"     : 0.10,
    "random_seed"    : 42,

    # Deep learning training
    "img_size"       : 256,
    "batch_size"     : 8,
    "num_epochs"     : 60,
    "lr"             : 3e-4,

    # Random Forest
    "rf_img_size"    : 128,
    "rf_max_train"   : 400,
    "rf_pixel_ratio" : 0.15,
    "rf_n_estimators": 100,
    "rf_max_depth"   : 25,
}

# ── Device setup (CUDA > MPS > CPU) ───────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"      # Apple Silicon GPU acceleration
else:
    DEVICE = "cpu"

# ── Reproducibility ───────────────────────────────────────────────
random.seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
torch.manual_seed(CFG["random_seed"])

os.makedirs(CFG["results_dir"], exist_ok=True)

print(f"Device: {DEVICE}")
