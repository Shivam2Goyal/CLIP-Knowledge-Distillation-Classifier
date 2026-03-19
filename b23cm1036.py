import torch
import torch.nn as nn
from torchvision import models


class MyAgeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        backbone = models.resnet18(weights=None)

        # Expose individual layers for extract_features compatibility
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x):
        """Extract 512-d backbone features (used during distillation)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)


def build_model(num_classes=2):
    return MyAgeClassifier(num_classes=num_classes)
