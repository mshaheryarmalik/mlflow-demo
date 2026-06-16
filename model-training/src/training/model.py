"""
CNN architecture for MNIST digit classification.

Architecture overview:
  Conv Block 1: Conv2d(1→32) → BatchNorm → ReLU → MaxPool
  Conv Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool → Dropout
  Classifier:   Linear(64×7×7 → 128) → ReLU → Dropout → Linear(128 → 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple 2-block CNN for MNIST (28×28 grayscale, 10 classes).

    Args:
        num_classes:  Number of output classes (default 10 for MNIST).
        dropout_rate: Dropout probability applied after conv block 2 and in classifier.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25) -> None:
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Classifier head (feature map: 64 × 7 × 7 after two 2×2 pools on 28×28)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: 28×28 → 14×14
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2: 14×14 → 7×7
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))

        x = x.view(x.size(0), -1)           # flatten: 64×7×7 = 3136
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)                   # raw logits


def build_model(num_classes: int = 10, dropout_rate: float = 0.25) -> SimpleCNN:
    """Factory function — returns an initialised SimpleCNN."""
    return SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
