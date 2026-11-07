"""Fashion-MNIST dataset loader with reproducible splits and normalizations."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ml.configs.config import DATA_DIR


def get_transforms(for_model: str = "cgan") -> tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test torchvision transforms tailored for specific model types.
    - cgan: normalizes to [-1, 1] matching Tanh generator activation.
    - vae: normalizes to [0, 1] for Bernoulli/BCE reconstruction loss.
    - classifier: normalizes with standard mean=0.2860, std=0.3530 for Fashion-MNIST.
    """
    if for_model.lower() == "cgan":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif for_model.lower() == "vae":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif for_model.lower() == "classifier":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    else:
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_fashion_mnist_dataloaders(
    for_model: str = "cgan",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    val_split: float = 0.1,
    data_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """
    Download (if needed) and prepare Fashion-MNIST train, val, and test dataloaders.
    """
    target_dir = str(data_dir or DATA_DIR)
    os.makedirs(target_dir, exist_ok=True)

    train_tf, test_tf = get_transforms(for_model=for_model)

    full_train_dataset = datasets.FashionMNIST(
        root=target_dir,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_dataset = datasets.FashionMNIST(
        root=target_dir,
        train=False,
        download=True,
        transform=test_tf,
    )

    if val_split > 0:
        generator = torch.Generator().manual_seed(seed)
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_subset, val_subset = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
