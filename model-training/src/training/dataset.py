"""
PyTorch Dataset and DataLoader helpers for MNIST.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training.config import TrainingConfig


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Return appropriate image transforms for train or validation split.

    Args:
        train: If True, add random augmentations; otherwise just normalize.
    """
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

    if train:
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(cfg: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """
    Download MNIST (if needed) and return train/val DataLoaders.

    Args:
        cfg: TrainingConfig holding data_dir, batch_size, num_workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
