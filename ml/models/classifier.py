"""Auxiliary Convolutional Classifier for Fashion-MNIST image evaluation."""


import torch
import torch.nn.functional as F
from torch import nn


class FashionCNNClassifier(nn.Module):
    """
    Lightweight CNN trained independently on Fashion-MNIST.
    Used as an objective evaluation oracle to measure class-fidelity of generated images.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (B, 1, 28, 28)
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def predict_classes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns predicted class indices and confidence probabilities."""
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidences, preds = torch.max(probs, dim=-1)
        return preds, confidences
