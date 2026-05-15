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