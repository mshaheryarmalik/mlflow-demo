"""
Training loop with MLflow experiment tracking.

MLflow features demonstrated:
  - mlflow.pytorch.autolog()    automatic param/metric/model capture
  - mlflow.log_params()         explicit hyperparameter logging
  - mlflow.log_metrics()        per-step and per-epoch metrics
  - mlflow.log_figure()         training curve saved as artifact
  - mlflow.pytorch.log_model()  log model with signature + input example
"""

import os
import random
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.dataset import get_dataloaders
from training.model import build_model

if TYPE_CHECKING:
    from training.model import SimpleCNN

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: TrainingConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return SGD(model.parameters(), lr=cfg.learning_rate,
                   weight_decay=cfg.weight_decay, momentum=0.9)
    return Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)
    return None


# ── Training + validation steps ───────────────────────────────────────────────


def train_epoch(
    model: "SimpleCNN",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    cfg: TrainingConfig,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

        global_step = epoch * len(loader) + step
        if step % cfg.log_every_n_steps == 0:
            mlflow.log_metric("train_step_loss", batch_loss, step=global_step)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_epoch(
    model: "SimpleCNN",
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    return fig


# ── Main training function ────────────────────────────────────────────────────


def train(cfg: TrainingConfig | None = None) -> str:
    """
    Run a full training job and log everything to MLflow.

    Args:
        cfg: TrainingConfig instance. Defaults to TrainingConfig() if None.

    Returns:
        The MLflow run_id of the completed training run.
    """
    if cfg is None:
        cfg = TrainingConfig()

    set_seed(cfg.seed)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # autolog captures model summary, gradients, and system metrics automatically
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)

    device = resolve_device(cfg)
    print(f"[training] Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(num_classes=cfg.num_classes, dropout_rate=cfg.dropout_rate).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    with mlflow.start_run(run_name=f"{cfg.model_name}-lr{cfg.learning_rate}-e{cfg.epochs}") as run:
        # Explicit params (supplement autolog)
        mlflow.log_params(cfg.to_dict())
        mlflow.log_param("device", str(device))
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))

        print(f"[mlflow] Run ID: {run.info.run_id}")
        print(f"[training] Starting {cfg.epochs} epochs on {cfg.dataset}...\n")

        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, cfg
            )
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

            if scheduler:
                scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch {epoch + 1:>3}/{cfg.epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

        # Log training curves as a figure artifact
        fig = plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)

        # Log the best validation accuracy as a summary metric
        best_val_acc = max(val_accs)
        mlflow.log_metric("best_val_acc", best_val_acc)
        print(f"\n[training] Best val accuracy: {best_val_acc:.4f}")

        # Build a model signature from a sample batch
        sample_images, _ = next(iter(val_loader))
        input_example = sample_images[:4].numpy()

        signature = mlflow.models.infer_signature(
            model_input=input_example,
            model_output=model(sample_images[:4].to(device)).detach().cpu().numpy(),
        )

        # Log the model to MLflow with full signature (enables serving)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg.registered_model_name,
        )
        print(f"[mlflow] Model logged and registered as '{cfg.registered_model_name}'")
        print(f"[mlflow] View in UI: http://localhost:5000/#/experiments")

        return run.info.run_id
