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
