"""
unet_model.py
-------------
Standalone U-Net definition, extracted verbatim from the segmentation
monolith (skin-lesions-segmentation.py) so the trained checkpoint
`results/unet_best.pth` loads with an identical state_dict.

The monolith's filename contains hyphens and cannot be imported as a
module, so the classification pipeline (clf_crop.py) imports the U-Net
from here instead. Keeping this class byte-compatible with the training
definition is what guarantees the weights load without key mismatches.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))

    def forward(self, x):
        return self.b(x)


class UNet(nn.Module):
    """
    U-Net: encoder-decoder with skip connections.
    Skip connections carry spatial detail from the encoder directly to
    the decoder, enabling precise boundary reconstruction.
    """

    def __init__(self, ic=3, oc=1, feats=[64, 128, 256, 512]):
        super().__init__()
        self.enc  = nn.ModuleList()
        self.ups  = nn.ModuleList()
        self.dec  = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        ch = ic
        for f in feats:
            self.enc.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = nn.Sequential(DoubleConv(feats[-1], feats[-1] * 2),
                                        nn.Dropout2d(0.3))
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.dec.append(DoubleConv(f * 2, f))
        self.out = nn.Conv2d(feats[0], oc, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i, (u, d) in enumerate(zip(self.ups, self.dec)):
            x = u(x); s = skips[i]
            if x.shape != s.shape:
                x = TF.center_crop(x, [s.shape[2], s.shape[3]])
            x = d(torch.cat([s, x], 1))
        return self.out(x)
