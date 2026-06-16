"""
Hyperparameter configuration for the CNN training pipeline.
Using a dataclass for clean type-safe config — no heavy framework required.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    All hyperparameters and settings for a single training run.
    These are logged verbatim to MLflow via mlflow.log_params().
    """

    # ── Dataset ───────────────────────────────────────────────────
    dataset: str = "MNIST"
    data_dir: str = "./data"
    num_workers: int = 2

    # ── Model ─────────────────────────────────────────────────────
    model_name: str = "SimpleCNN"
    num_classes: int = 10
    dropout_rate: float = 0.25

    # ── Training ─────────────────────────────────────────────────
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"       # "adam" | "sgd"
    scheduler: str = "cosine"     # "cosine" | "step" | "none"

    # ── MLflow ────────────────────────────────────────────────────
    experiment_name: str = "mnist-cnn"
    registered_model_name: str = "mnist-cnn-model"
    tracking_uri: str = "http://localhost:5000"

    # ── Misc ──────────────────────────────────────────────────────
    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    log_every_n_steps: int = 100

    def to_dict(self) -> dict:
        """Serialise config to a flat dict for mlflow.log_params()."""
        return {
            "dataset": self.dataset,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "seed": self.seed,
        }
