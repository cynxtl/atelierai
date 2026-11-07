"""Configuration constants and hyperparameters for AtelierAI ML pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Official 10 Fashion-MNIST Classes (Index 0 to 9)
FASHION_CLASSES: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Standard lowercase mapping for robust normalization
CLASS_TO_IDX = {name.lower(): i for i, name in enumerate(FASHION_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(FASHION_CLASSES)}

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "ml" / "data" / "FashionMNIST"
CHECKPOINTS_DIR = ROOT_DIR / "ml" / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CGANConfig:
    """Hyperparameters for Conditional GAN training."""
    latent_dim: int = 100
    num_classes: int = 10
    label_embed_dim: int = 50
    channels: int = 1
    img_size: int = 28
    batch_size: int = 64
    lr_g: float = 0.0002
    lr_d: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    epochs: int = 20
    seed: int = 42
    checkpoint_path: Path = CHECKPOINTS_DIR / "cgan_generator.pt"
    discriminator_checkpoint: Path = CHECKPOINTS_DIR / "cgan_discriminator.pt"
    metrics_path: Path = CHECKPOINTS_DIR / "cgan_metrics.json"


@dataclass
class VAEConfig:
    """Hyperparameters for Conditional VAE training."""
    latent_dim: int = 32
    num_classes: int = 10
    label_embed_dim: int = 16
    channels: int = 1
    img_size: int = 28
    batch_size: int = 64
    lr: float = 0.001
    epochs: int = 25
    beta: float = 1.0  # Weight of KL divergence
    seed: int = 42
    checkpoint_path: Path = CHECKPOINTS_DIR / "vae.pt"
    metrics_path: Path = CHECKPOINTS_DIR / "vae_metrics.json"


@dataclass
class ClassifierConfig:
    """Hyperparameters for Auxiliary Fashion-MNIST CNN Classifier."""
    num_classes: int = 10
    batch_size: int = 128
    lr: float = 0.001
    epochs: int = 8
    seed: int = 42
    checkpoint_path: Path = CHECKPOINTS_DIR / "fashion_classifier.pt"
    metrics_path: Path = CHECKPOINTS_DIR / "classifier_metrics.json"


@dataclass
class DistilBERTConfig:
    """Hyperparameters for fine-tuning DistilBERT on clothing text requests."""
    model_name: str = "distilbert/distilbert-base-uncased"
    num_classes: int = 10
    batch_size: int = 16
    lr: float = 3e-5
    epochs: int = 4
    seed: int = 42
    output_dir: Path = CHECKPOINTS_DIR / "distilbert_fashion"
    metrics_path: Path = CHECKPOINTS_DIR / "distilbert_metrics.json"


@dataclass
class EvaluationConfig:
    """Settings for benchmark evaluation."""
    samples_per_class: int = 100
    clip_model_name: str = "openai/clip-vit-base-patch32"
    qwen_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    benchmark_report_path: Path = REPORTS_DIR / "benchmark_results.json"
