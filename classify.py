"""
classify.py
-----------
Entry point for the two-stage lesion CLASSIFICATION pipeline
(HAM10000 / ISIC2018), companion to the segmentation pipeline in
skin-lesions-segmentation.py.

    Stage 1 (crop)  : trained U-Net localizes + crops the lesion ROI.
    Stage 2 (train) : ImageNet-pretrained CNN classifies the crop (7 dx).

Usage:
    python classify.py --phase all                     # crop -> train -> eval
    python classify.py --phase crop
    python classify.py --phase train    --backbone efficientnet_b0
    python classify.py --phase evaluate --backbone resnet50

Prerequisites:
    * Dataset extracted under data/ (metadata CSV + images; auto-detected).
    * For lesion-aware cropping: a trained U-Net at results/unet_best.pth
      (else Stage 1 falls back to center crops).
"""

import os
import json
import argparse

from clf_config import CLF


def _update_metrics(name, metrics):
    """Append classifier metrics into results/clf_metrics.json."""
    path = os.path.join(CLF["results_dir"], "clf_metrics.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[name] = metrics
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Metrics updated -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Lesion classification pipeline")
    parser.add_argument("--phase", default="all",
                        choices=["all", "crop", "train", "evaluate"])
    parser.add_argument("--backbone", default=CLF["backbone"],
                        choices=["efficientnet_b0", "resnet50"])
    args = parser.parse_args()
    CLF["backbone"] = args.backbone  # honor CLI override everywhere downstream

    if args.phase in ("all", "crop"):
        from clf_crop import run_cropping
        print("\n" + "▶" * 3 + " STAGE 1: CROPPING")
        run_cropping()

    if args.phase in ("all", "train"):
        from clf_train import train_classifier
        print("\n" + "▶" * 3 + f" STAGE 2: TRAINING ({args.backbone})")
        metrics = train_classifier(args.backbone)
        _update_metrics(args.backbone, metrics)

    elif args.phase == "evaluate":
        from clf_train import evaluate_classifier
        metrics = evaluate_classifier(args.backbone)
        if metrics:
            _update_metrics(args.backbone, metrics)

    print("\n" + "═" * 55)
    print("DONE — classification outputs are in results/")
    print("═" * 55)


if __name__ == "__main__":
    main()
