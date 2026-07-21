"""
clf_dataset.py
--------------
HAM10000 / ISIC2018 classification dataset.

Responsibilities:
  * Auto-detect the metadata CSV and image folders under data/ (works
    whether labels are a `dx` column or one-hot MEL/NV/... columns).
  * Build an image_id -> original-image-path index.
  * Split by lesion_id (grouped) to prevent the same lesion leaking
    across train/val/test.
  * Serve cropped lesions when the crop cache exists, else originals.
  * Provide class weights + a WeightedRandomSampler to counter the
    heavy class imbalance (nv ~ 67 %).

These auto-detection helpers are also imported by clf_crop.py.
"""

import os
import glob
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from sklearn.model_selection import GroupShuffleSplit

from config import CFG
from clf_config import CLF

_IMG_EXT = (".jpg", ".jpeg", ".png")

# One-hot column name -> HAM diagnosis code (for ISIC2018 GroundTruth CSVs).
_ONEHOT_TO_CODE = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
}


# ══════════════════════════════════════════════════════════════════
# AUTO-DETECTION HELPERS  (shared with clf_crop.py)
# ══════════════════════════════════════════════════════════════════

def _iter_image_files(root, exclude_dirs=()):
    """Yield absolute paths of image files under `root`, skipping excluded dirs."""
    exclude = tuple(os.path.abspath(d) for d in exclude_dirs)
    for dirpath, _dirs, files in os.walk(root):
        ad = os.path.abspath(dirpath)
        if ad.startswith(exclude):
            continue
        for f in files:
            if f.lower().endswith(_IMG_EXT):
                yield os.path.join(dirpath, f)


def build_image_index(root=None):
    """
    Map image_id (filename without extension) -> original image path.

    Excludes the crop cache so this always resolves to *original*
    full-resolution images. If two files share a stem, the first found
    wins (HAM ids like 'ISIC_0024306' are unique in practice).
    """
    root = root or CLF["data_root"]
    index = {}
    for path in _iter_image_files(root, exclude_dirs=[CLF["crop_dir"]]):
        stem = os.path.splitext(os.path.basename(path))[0]
        index.setdefault(stem, path)
    return index


def find_metadata_csv(root=None):
    """
    Locate the metadata CSV under `root`. Prefers files whose name hints
    at HAM/ISIC metadata; falls back to any CSV that has a usable label.
    """
    root = root or CLF["data_root"]
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found under '{root}'. Place the HAM10000/ISIC2018 "
            f"dataset (images + metadata CSV) there.")

    def score(p):
        name = os.path.basename(p).lower()
        s = 0
        for kw, w in (("metadata", 3), ("ham10000", 2),
                      ("groundtruth", 2), ("ground_truth", 2)):
            if kw in name:
                s += w
        return s

    return sorted(csvs, key=score, reverse=True)[0]


def _pick(colnames, *candidates):
    """Return the actual column name matching any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_metadata(root=None):
    """
    Read + normalize the metadata into a DataFrame with columns:
        image_id, label (dx code), lesion_id, age, sex, localization, path

    Handles both label encodings:
        * a `dx` column of diagnosis codes, or
        * one-hot columns (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
    Rows whose image file is missing on disk are dropped.
    """
    root = root or CLF["data_root"]
    csv_path = find_metadata_csv(root)
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # --- image id column ---
    id_col = _pick(cols, "image_id", "image", "image_name", "isic_id")
    if id_col is None:
        id_col = cols[0]  # ISIC GT CSVs put the id first, unnamed as 'image'
    image_id = df[id_col].astype(str).map(
        lambda s: os.path.splitext(os.path.basename(s.strip()))[0])

    # --- label column(s) ---
    dx_col = _pick(cols, "dx", "diagnosis", "label")
    if dx_col is not None:
        label = df[dx_col].astype(str).str.strip().str.lower()
    else:
        onehot = {c: _pick(cols, c) for c in _ONEHOT_TO_CODE}
        present = {k: v for k, v in onehot.items() if v is not None}
        if len(present) < 2:
            raise ValueError(
                f"Could not find a 'dx' column or one-hot label columns in "
                f"{csv_path}. Columns present: {cols}")
        oh = df[[present[k] for k in present]].astype(float).to_numpy()
        keys = list(present.keys())
        label = pd.Series([_ONEHOT_TO_CODE[keys[i]] for i in oh.argmax(axis=1)],
                          index=df.index)

    out = pd.DataFrame({"image_id": image_id.values, "label": label.values})

    # --- optional metadata columns (used later for CSV fusion) ---
    for name, cands in (("lesion_id", ("lesion_id",)),
                        ("age", ("age",)),
                        ("sex", ("sex",)),
                        ("localization", ("localization", "anatom_site_general"))):
        col = _pick(cols, *cands)
        out[name] = df[col].values if col is not None else np.nan

    # --- keep only known classes and resolve image paths ---
    out = out[out["label"].isin(CLF["classes"])].copy()
    index = build_image_index(root)
    out["path"] = out["image_id"].map(index.get)
    missing = out["path"].isna().sum()
    out = out.dropna(subset=["path"]).reset_index(drop=True)

    print(f"Metadata: {csv_path}")
    print(f"  {len(out)} labeled images resolved"
          + (f" ({missing} had no image file on disk, skipped)" if missing else ""))
    return out


def _resolve_serving_path(image_id, original_path):
    """Prefer the cached crop; fall back to the original image."""
    crop = os.path.join(CLF["crop_dir"], image_id + ".jpg")
    return crop if os.path.exists(crop) else original_path


# ══════════════════════════════════════════════════════════════════
# SPLIT (grouped by lesion)
# ══════════════════════════════════════════════════════════════════

def _grouped_split(df):
    """
    Split into train/val/test grouped by lesion_id so images of the same
    lesion never span splits. Falls back to per-image groups (== random)
    when lesion_id is absent.
    """
    groups = df["lesion_id"]
    if groups.isna().all():
        groups = df["image_id"]
    groups = groups.astype(str).values
    seed = CFG["random_seed"]

    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=CLF["test_split"],
                             random_state=seed)
    trainval_i, test_i = next(gss1.split(idx, groups=groups))

    val_frac = CLF["val_split"] / (1.0 - CLF["test_split"])
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_rel, val_rel = next(gss2.split(trainval_i, groups=groups[trainval_i]))
    train_i, val_i = trainval_i[tr_rel], trainval_i[val_rel]

    return df.iloc[train_i].reset_index(drop=True), \
        df.iloc[val_i].reset_index(drop=True), \
        df.iloc[test_i].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class LesionClsDataset(Dataset):
    """Serves (image_tensor, label_idx). Uses cropped lesions when cached."""

    def __init__(self, df, class_to_idx, augment=False):
        self.paths  = [_resolve_serving_path(r.image_id, r.path)
                       for r in df.itertuples()]
        self.labels = [class_to_idx[l] for l in df["label"].tolist()]
        self.augment = augment
        self.sz = CLF["img_size"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = TF.resize(img, [self.sz, self.sz])

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
            img = TF.rotate(img, random.uniform(-35, 35))
            if random.random() > 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            if random.random() > 0.4:
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
            if random.random() > 0.4:
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))

        img = TF.normalize(TF.to_tensor(img),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, self.labels[i]


# ══════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════

def compute_class_weights(train_df, class_to_idx):
    """Inverse-frequency (balanced) class weights aligned to class indices."""
    counts = train_df["label"].value_counts().to_dict()
    n_cls  = len(class_to_idx)
    total  = len(train_df)
    weights = torch.ones(n_cls)
    for cls, idx in class_to_idx.items():
        c = counts.get(cls, 0)
        weights[idx] = total / (n_cls * c) if c > 0 else 0.0
    return weights


def build_clf_loaders(return_frames=False):
    """
    Build train/val/test DataLoaders for classification.

    Returns:
        train_dl, val_dl, test_dl, class_weights, meta
    where meta = {"class_to_idx", "classes", "test_df", "train_df", "val_df"}.
    num_workers=0 / pin_memory=False for macOS + MPS (matches segmentation).
    """
    df = load_metadata()
    classes = CLF["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df, val_df, test_df = _grouped_split(df)
    print(f"Split (by lesion): {len(train_df)} train | "
          f"{len(val_df)} val | {len(test_df)} test")
    print("Train class distribution: "
          + ", ".join(f"{c}={int((train_df['label'] == c).sum())}"
                      for c in classes))

    class_weights = compute_class_weights(train_df, class_to_idx)

    train_ds = LesionClsDataset(train_df, class_to_idx, augment=True)
    val_ds   = LesionClsDataset(val_df,   class_to_idx, augment=False)
    test_ds  = LesionClsDataset(test_df,  class_to_idx, augment=False)

    # WeightedRandomSampler: oversample rare classes during training.
    sample_w = torch.tensor([class_weights[y].item() for y in train_ds.labels],
                            dtype=torch.double)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds),
                                    replacement=True)

    bs = CLF["batch_size"]
    train_dl = DataLoader(train_ds, bs, sampler=sampler,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   bs, shuffle=False,
                          num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  bs, shuffle=False,
                          num_workers=0, pin_memory=False)

    meta = {"class_to_idx": class_to_idx, "classes": classes,
            "test_df": test_df, "train_df": train_df, "val_df": val_df}
    if return_frames:
        return train_dl, val_dl, test_dl, class_weights, meta
    return train_dl, val_dl, test_dl, class_weights, meta


if __name__ == "__main__":
    # Quick sanity check of dataset auto-detection.
    build_clf_loaders()
