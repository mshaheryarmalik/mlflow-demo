"""
run_training.py — CLI entry point for the PyTorch CNN training pipeline.

Demonstrates end-to-end MLflow experiment tracking:
  1. Parse CLI args → build TrainingConfig
  2. Run training with automatic + explicit MLflow logging
  3. Evaluate and promote the best model to 'champion' in the registry
  4. Optionally run multiple experiments for hyperparameter comparison

Usage:
    uv run python scripts/run_training.py
    uv run python scripts/run_training.py --epochs 10 --lr 5e-4
    uv run python scripts/run_training.py --sweep
    uv run python scripts/run_training.py --promote --run-id <RUN_ID>
"""

import argparse
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from training.config import TrainingConfig
from training.evaluate import promote_to_champion
from training.train import train

# Load .env from repo root (two levels up from this script)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def run_hyperparameter_sweep() -> None:
    """
    Run a small grid search over learning rates and batch sizes.
    Each configuration creates a separate MLflow run — compare them in the UI.
    """
    learning_rates = [1e-2, 1e-3, 5e-4]
    batch_sizes = [32, 64]

    print(f"[sweep] Running {len(learning_rates) * len(batch_sizes)} configurations...")
    best_run_id = None
    best_acc = 0.0

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n[sweep] lr={lr}, batch_size={bs}")
            cfg = TrainingConfig(
                learning_rate=lr,
                batch_size=bs,
                epochs=3,  # fewer epochs for sweep speed
            )
            run_id = train(cfg)

            # Read best_val_acc from the completed run
            client = mlflow.MlflowClient()
            run = client.get_run(run_id)
            acc = run.data.metrics.get("best_val_acc", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_run_id = run_id

    print(f"\n[sweep] Best run: {best_run_id} (val_acc={best_acc:.4f})")
    return best_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST with MLflow tracking.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate.")
    parser.add_argument("--experiment", type=str, default="mnist-cnn", help="Experiment name.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a hyperparameter sweep instead of a single run.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the best model version to 'champion' alias after training.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Promote a specific run_id (used with --promote).",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    if args.sweep:
        run_id = run_hyperparameter_sweep()
    else:
        cfg = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            dropout_rate=args.dropout,
            experiment_name=args.experiment,
        )
        run_id = train(cfg)

    if args.promote:
        target_run_id = args.run_id or run_id
        print(f"\n[promote] Evaluating run {target_run_id} for champion promotion...")
        cfg = TrainingConfig()
        promoted = promote_to_champion(
            model_name=cfg.registered_model_name,
            run_id=target_run_id,
            min_accuracy=0.98,
            cfg=cfg,
        )
        if promoted:
            print("[promote] Model is now the 'champion'. Serve it with:")
            print(f"  mlflow models serve -m 'models:/{cfg.registered_model_name}@champion' -p 8080")


if __name__ == "__main__":
    main()
