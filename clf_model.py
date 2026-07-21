"""
clf_model.py
------------
Stage 2 classifier: an ImageNet-pretrained CNN with its final layer
swapped for a 7-class head.

Transfer learning is the right call here — HAM10000 is small (~10k
images) and heavily imbalanced, so ImageNet features give a much stronger
starting point than training from scratch.

CSV-FUSION HOOK (later improvement): to add metadata (age/sex/localization),
replace the final head with a small MLP that takes torch.cat([cnn_embedding,
tabular_features]); clf_dataset already parses those columns.
"""

import torch.nn as nn
import torchvision.models as tvm

_HEAD_PREFIX = {"efficientnet_b0": "classifier.", "resnet50": "fc."}


def build_classifier(backbone=None, num_classes=None, pretrained=None):
    """
    Build the classification model.

    Returns the model. The final layer is a fresh nn.Linear -> num_classes;
    use get_param_groups() to give it a higher LR than the backbone.
    """
    from clf_config import CLF
    backbone    = backbone    or CLF["backbone"]
    num_classes = num_classes or len(CLF["classes"])
    pretrained  = CLF["pretrained"] if pretrained is None else pretrained

    if backbone == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)

    elif backbone == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. "
                         f"Use 'efficientnet_b0' or 'resnet50'.")

    return model


def get_param_groups(model, backbone, lr_head, lr_backbone):
    """
    Split params into (freshly-initialized head) vs (pretrained backbone)
    so the head can learn faster than the fine-tuned backbone.
    """
    prefix = _HEAD_PREFIX[backbone]
    head, base = [], []
    for name, p in model.named_parameters():
        (head if name.startswith(prefix) else base).append(p)
    return [
        {"params": head, "lr": lr_head},
        {"params": base, "lr": lr_backbone},
    ]
