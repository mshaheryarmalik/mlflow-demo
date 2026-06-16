"""
Model evaluation and MLflow Model Registry management.

Demonstrates:
  - Loading a model from the registry by name + alias
  - Running full evaluation on the test set
  - Logging evaluation metrics back to the originating run
  - Transitioning a model version to 'champion' alias (production)
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.dataset import get_dataloaders


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Run full evaluation and return a metrics dict.

    Args:
        model:  A loaded PyTorch model in eval mode.
        loader: DataLoader for the evaluation split.
        device: torch.device to run on.

    Returns:
        Dict with keys: accuracy, loss, total_samples.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return {
        "eval_accuracy": correct / total,
        "eval_loss": total_loss / total,
        "eval_total_samples": float(total),
    }


# ── Registry helpers ──────────────────────────────────────────────────────────


def promote_to_champion(
    model_name: str,
    run_id: str,
    min_accuracy: float = 0.98,
    cfg: TrainingConfig | None = None,
) -> bool:
    """
    Promote the latest version of a registered model to the 'champion' alias
    if it meets the minimum accuracy threshold.

    Args:
        model_name:    Registered model name in MLflow.
        run_id:        The run_id that produced the model.
        min_accuracy:  Minimum val accuracy required for promotion.
        cfg:           TrainingConfig (for device + data settings).

    Returns:
        True if promoted, False otherwise.
    """
    if cfg is None:
        cfg = TrainingConfig()

    client = MlflowClient()

    # Fetch latest model version for this run
    versions = client.search_model_versions(f"name='{model_name}'")
    matching = [v for v in versions if v.run_id == run_id]
    if not matching:
        print(f"[registry] No version found for run_id={run_id}")
        return False

    version = matching[0]
    print(f"[registry] Found model version {version.version} (run_id={run_id})")

    # Load model and evaluate
    device = torch.device("cpu")
    _, val_loader = get_dataloaders(cfg)

    model_uri = f"models:/{model_name}/{version.version}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)

    metrics = evaluate_model(model, val_loader, device)
    accuracy = metrics["eval_accuracy"]
    print(f"[registry] Evaluation accuracy: {accuracy:.4f}")

    # Log evaluation metrics back to the originating run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        print(f"[mlflow] Logged eval metrics to run {run_id}")

    if accuracy >= min_accuracy:
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version.version,
        )
        print(
            f"[registry] Promoted version {version.version} to 'champion' alias "
            f"(accuracy={accuracy:.4f} >= threshold={min_accuracy})"
        )
        return True

    print(
        f"[registry] Not promoting — accuracy {accuracy:.4f} < threshold {min_accuracy}"
    )
    return False


def load_champion_model(model_name: str) -> nn.Module:
    """
    Load the 'champion' alias model from the registry.

    Args:
        model_name: Registered model name.

    Returns:
        Loaded PyTorch model in eval mode.
    """
    model_uri = f"models:/{model_name}@champion"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print(f"[registry] Loaded champion model: {model_uri}")
    return model
