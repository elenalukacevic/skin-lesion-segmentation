# Dermoscopic Lesion Segmentation

Seminar project — comparison of segmentation methods applied to the 
Melanoma Detection Dataset for automatic mole segmentation at pixel level.

Implemented methods:
- U-Net — encoder-decoder architecture with skip connections
- SegNet — encoder-decoder architecture with max-pooling indices
- Random Forest — pixel-wise classifier using handcrafted features

Evaluation metrics: Pixel Accuracy, Dice Coefficient, Jaccard Index

Results:
| Model         | Pixel Accuracy | Dice  | IoU    |
|---------------|---------------|--------|--------|
| U-Net         | 0.9461        | 0.8430 | 0.7485 |
| SegNet        | 0.9022        | 0.7751 | 0.6779 |
| Random Forest | 0.8905        | 0.7607 | 0.6445 |
## Getting Started

1. Clone the repository
   git clone https://github.com/elenalukacevic/skin-lesion-segmentation

2. Install dependencies
   pip install -r requirements.txt

3. Set up the dataset
   Download the Melanoma Detection Dataset and place it in:
   data/images/   ← dermoscopic images
   data/masks/    ← binary segmentation masks

4. Run the full pipeline
   python skin-lesion-segmentation.py

   Or run individual phases:
   python skin-lesion-segmentation.py --phase preprocess
   python skin-lesion-segmentation.py --phase train_unet
   python skin-lesion-segmentation.py --phase train_segnet
   python skin-lesion-segmentation.py --phase train_rf
   python skin-lesion-segmentation.py --phase compare

## Classification (HAM10000 / ISIC2018)

A second, two-stage task classifies a dermoscopic image into one of the
7 HAM10000 diagnoses (`akiec, bcc, bkl, df, mel, nv, vasc`):

- **Stage 1 — crop:** the trained segmentation U-Net localizes the lesion
  and caches a tight crop to `data/ham_cropped/`.
- **Stage 2 — classify:** an ImageNet-pretrained CNN (EfficientNet-B0 by
  default, ResNet50 optional) predicts the diagnosis from the crop.

This is the image-only version; metadata (age / sex / localization) fusion
is a planned improvement (the loader already parses those columns).

### Setup

1. Download the [HAM10000 / ISIC2018 raw dataset](https://www.kaggle.com/datasets/nightfury007/ham10000-isic2018-raw)
   and extract it anywhere under `data/`. The loader auto-detects the
   metadata CSV and image folders (it handles both a `dx` column and
   one-hot `MEL,NV,...` labels).
2. (Recommended) Train the segmentation U-Net first so Stage 1 can crop
   lesion-aware; otherwise Stage 1 falls back to center crops.

### Run

    python classify.py --phase all                     # crop -> train -> evaluate
    python classify.py --phase crop
    python classify.py --phase train    --backbone efficientnet_b0
    python classify.py --phase evaluate --backbone resnet50

The dataset is heavily imbalanced (`nv` ≈ 67%), so training uses a
weighted sampler + class-weighted loss and reports **balanced accuracy**,
**macro-F1**, a per-class `classification_report`, and a confusion matrix
(all written to `results/`).
