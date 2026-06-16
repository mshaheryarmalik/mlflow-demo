"""
Tests for the model-training package.
These tests run without a GPU or live MLflow server.
"""

import torch
import pytest

from training.config import TrainingConfig
from training.model import SimpleCNN, build_model


# ── Config tests ──────────────────────────────────────────────────────────────


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs == 5
    assert cfg.learning_rate == 1e-3
    assert cfg.num_classes == 10
    assert cfg.dataset == "MNIST"


def test_training_config_to_dict():
    cfg = TrainingConfig(epochs=10, learning_rate=5e-4)
    d = cfg.to_dict()
    assert d["epochs"] == 10
    assert d["learning_rate"] == 5e-4
    assert "dropout_rate" in d
    assert "optimizer" in d


# ── Model tests ───────────────────────────────────────────────────────────────


def test_model_output_shape():
    """Model should output (batch_size, num_classes) logits."""
    model = build_model(num_classes=10)
    model.eval()
    x = torch.randn(4, 1, 28, 28)  # batch of 4 MNIST images
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_model_no_nan():
    """Forward pass should not produce NaN."""
    model = build_model()
    model.eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


def test_model_dropout_rate():
    """Dropout rate should be stored correctly."""
    model = SimpleCNN(num_classes=10, dropout_rate=0.5)
    assert model.dropout.p == 0.5


def test_model_parameter_count():
    """Model should have a reasonable number of parameters for MNIST."""
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    # SimpleCNN should be between 100K and 2M params
    assert 100_000 < total_params < 2_000_000


def test_model_train_step():
    """Single forward + backward pass should work without errors."""
    model = build_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
