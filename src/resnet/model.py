"""
ResNet-50 Transfer Learning — CIFAR-10
========================================
Stage 1 — Feature extraction: freeze backbone, train fc only
Stage 2 — Fine-tuning: unfreeze layer3 + layer4, train full model
"""

import torch.nn as nn
from torchvision import models


class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # load pretrained ResNet-50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # replace final fc layer
        in_features = self.backbone.fc.in_features        # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # start in feature extraction mode
        self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_layers(self, layers: list):
        for name, param in self.backbone.named_parameters():
            for layer in layers:
                if layer in name:
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)