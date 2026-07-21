"""
clf_train.py
------------
Stage 2 training + evaluation for the lesion classifier.

Because the dataset is heavily imbalanced (nv ~ 67 %), raw accuracy is
misleading — a model that predicts 'nv' for everything scores ~0.67.
So training pairs a WeightedRandomSampler (clf_dataset) with a
class-weighted CrossEntropyLoss, selects the best model on **macro-F1**,
and reports balanced accuracy, per-class precision/recall and a
confusion matrix.

Outputs (results/):
    clf_<backbone>_best.pth        best checkpoint (by val macro-F1)
    clf_<backbone>_history.json    per-epoch metrics
    clf_<backbone>_history.png     loss / macro-F1 curves
    clf_<backbone>_report.txt      sklearn classification_report on test
    clf_<backbone>_confusion.png   confusion matrix heatmap
    clf_<backbone>_predictions.png sample predictions grid
"""

import os
import json

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from config import CFG
from clf_config import CLF, DEVICE
from clf_dataset import build_clf_loaders, _resolve_serving_path
from clf_model import build_classifier, get_param_groups

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _paths(backbone):
    r = CLF["results_dir"]
    return {
        "ckpt"   : os.path.join(r, f"clf_{backbone}_best.pth"),
        "history": os.path.join(r, f"clf_{backbone}_history.json"),
        "curve"  : os.path.join(r, f"clf_{backbone}_history.png"),
        "report" : os.path.join(r, f"clf_{backbone}_report.txt"),
        "cm"     : os.path.join(r, f"clf_{backbone}_confusion.png"),
        "preds"  : os.path.join(r, f"clf_{backbone}_predictions.png"),
    }


# ══════════════════════════════════════════════════════════════════
# PREDICTION COLLECTION + METRICS
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect(model, loader):
    """Return (y_true, y_pred) numpy arrays over a loader."""
    model.eval()
    ys, ps = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(DEVICE))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(np.asarray(labels))
    return np.concatenate(ys), np.concatenate(ps)


def _metrics(y_true, y_pred):
    return {
        "accuracy"         : accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1"         : f1_score(y_true, y_pred, average="macro",
                                      zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════

def train_classifier(backbone=None):
    backbone = backbone or CLF["backbone"]
    print(f"\n{'═' * 55}")
    print(f"STAGE 2 — TRAINING CLASSIFIER — {backbone}")
    print(f"{'═' * 55}")

    train_dl, val_dl, test_dl, class_weights, meta = build_clf_loaders()
    paths = _paths(backbone)

    model = build_classifier(backbone).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE),
                                    label_smoothing=CLF["label_smoothing"])
    optimizer = optim.AdamW(
        get_param_groups(model, backbone, CLF["lr_head"], CLF["lr_backbone"]),
        weight_decay=CLF["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLF["epochs"])

    best_f1, history = 0.0, []
    print(f"\n{'Epoch':>5} | {'Train L':>7} | {'Val Acc':>7} | "
          f"{'Val Bal':>7} | {'Val F1':>7}")
    print("─" * 48)

    for epoch in range(1, CLF["epochs"] + 1):
        model.train()
        tl = 0.0
        for imgs, labels in tqdm(train_dl, desc=f"Ep {epoch:3d}",
                                 leave=False, ncols=65):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_dl)

        y, p = _collect(model, val_dl)
        m = _metrics(y, p)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tl, 4),
                        **{k: round(v, 4) for k, v in m.items()}})
        print(f"{epoch:>5d} | {tl:>7.4f} | {m['accuracy']:>7.4f} | "
              f"{m['balanced_accuracy']:>7.4f} | {m['macro_f1']:>7.4f}")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), paths["ckpt"])
            print(f"        ✓ best model (macro-F1={best_f1:.4f})")

    with open(paths["history"], "w") as f:
        json.dump(history, f, indent=2)
    _save_history_plot(history, paths["curve"])

    # Final evaluation on the test set with the best checkpoint.
    model.load_state_dict(torch.load(paths["ckpt"], map_location=DEVICE))
    return _evaluate(model, test_dl, meta, backbone, paths)


def evaluate_classifier(backbone=None):
    """Load the best checkpoint and evaluate on the test set."""
    backbone = backbone or CLF["backbone"]
    paths = _paths(backbone)
    if not os.path.exists(paths["ckpt"]):
        print(f"⚠ No checkpoint at '{paths['ckpt']}'. Train first "
              f"(python classify.py --phase train).")
        return None
    _, _, test_dl, _cw, meta = build_clf_loaders()
    model = build_classifier(backbone).to(DEVICE)
    model.load_state_dict(torch.load(paths["ckpt"], map_location=DEVICE))
    return _evaluate(model, test_dl, meta, backbone, paths)


# ══════════════════════════════════════════════════════════════════
# EVALUATION + VISUALS
# ══════════════════════════════════════════════════════════════════

def _evaluate(model, test_dl, meta, backbone, paths):
    classes = meta["classes"]
    y, p = _collect(model, test_dl)
    m = _metrics(y, p)

    print(f"\n{'─' * 48}")
    print(f"TEST SET — {backbone}")
    print(f"  Accuracy          : {m['accuracy']:.4f}")
    print(f"  Balanced accuracy : {m['balanced_accuracy']:.4f}")
    print(f"  Macro-F1          : {m['macro_f1']:.4f}")

    report = classification_report(
        y, p, labels=list(range(len(classes))), target_names=classes,
        digits=4, zero_division=0)
    print("\n" + report)
    with open(paths["report"], "w") as f:
        f.write(f"Backbone: {backbone}\n\n{report}\n")

    cm = confusion_matrix(y, p, labels=list(range(len(classes))))
    _save_confusion_matrix(cm, classes, paths["cm"], backbone)
    _save_pred_grid(model, meta["test_df"], meta["class_to_idx"],
                    classes, paths["preds"])

    print(f"  Report      -> {paths['report']}")
    print(f"  Confusion   -> {paths['cm']}")
    print(f"  Predictions -> {paths['preds']}")
    return {k: round(v, 4) for k, v in m.items()}


def _save_history_plot(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], "C3", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="C3")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["macro_f1"] for h in history], "C0", label="val macro-F1")
    ax2.plot(ep, [h["balanced_accuracy"] for h in history], "C2",
             label="val balanced acc")
    ax2.set_ylabel("val metric"); ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_confusion_matrix(cm, classes, path, backbone):
    """Row-normalized heatmap annotated with raw counts."""
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45,
                                                           ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized) — {backbone}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


@torch.no_grad()
def _save_pred_grid(model, test_df, class_to_idx, classes, path, n=8):
    """Grid of sample test crops annotated with true vs predicted label."""
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    sample = test_df.sample(min(n, len(test_df)),
                            random_state=CFG["random_seed"]).itertuples()
    tiles, sz = [], 160
    for row in sample:
        sp = _resolve_serving_path(row.image_id, row.path)
        pil = Image.open(sp).convert("RGB").resize((sz, sz), Image.LANCZOS)
        x = TF.normalize(TF.to_tensor(pil), _MEAN, _STD).unsqueeze(0).to(DEVICE)
        pred = idx_to_class[int(model(x).argmax(1).item())]
        true = row.label

        tile = np.array(pil)
        pil2 = Image.fromarray(tile)
        d = ImageDraw.Draw(pil2)
        ok = pred == true
        d.rectangle([0, 0, sz - 1, 22], fill=(30, 30, 30))
        d.text((4, 6), f"T:{true} P:{pred}",
               fill=(80, 230, 80) if ok else (240, 80, 80))
        tiles.append(np.array(pil2))

    if not tiles:
        return
    cols = 4
    rows = [np.concatenate(tiles[i:i + cols], axis=1)
            for i in range(0, len(tiles), cols)]
    # pad last row to full width
    if rows and rows[-1].shape[1] < rows[0].shape[1]:
        padw = rows[0].shape[1] - rows[-1].shape[1]
        rows[-1] = np.concatenate(
            [rows[-1], np.zeros((rows[-1].shape[0], padw, 3), np.uint8)], axis=1)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
