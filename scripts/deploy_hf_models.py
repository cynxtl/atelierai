"""
Uploads trained AtelierAI model checkpoints to Hugging Face Model Hub.
Repo: cynstl/atelierai-models (Public)
"""

from pathlib import Path

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "ml" / "checkpoints"
REPO_ID = "cynstl/atelierai-models"

# Load token from ~/.cache/huggingface/token
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text().strip() if token_path.exists() else None

api = HfApi(token=token)

print(f"[1/3] Creating/verifying Hugging Face Model Repository: {REPO_ID}...")
repo_url = create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=False,
    exist_ok=True,
    token=token,
)
print(f"Repository ready at: {repo_url}")

# Create model card README.md
model_card = """---
license: mit
tags:
- pytorch
- generative-ai
- cgan
- vae
- fashion-mnist
- image-generation
datasets:
- fashion_mnist
pipeline_tag: image-to-image
---

# AtelierAI Trained Model Checkpoints

This repository hosts the trained PyTorch model checkpoints and evaluation metrics for **AtelierAI: Text-to-Conditional-Fashion Synthesis Studio**.

## Models Included

1. **Conditional GAN Generator (`cgan_generator.pt`)**:
   - Architecture: Deep Convolutional GAN with **Projection Discriminator** (Miyato & Sugiyama, 2018)
   - Conditioning: 10 Fashion-MNIST apparel classes from latent noise $z \\sim \\mathcal{N}(0, I)$ (100-dim)
   - Resolution: $1 \times 28 \times 28$ grayscale
   - Empirical Fidelity: **97.20% Oracle Match Rate** (Tested on 1,000 samples)
   - Training: 25 epochs on NVIDIA GeForce RTX 3060 CUDA

2. **Conditional VAE (`vae.pt`)**:
   - Architecture: Convolutional Conditional Variational Autoencoder
   - Latent Space: 32-dimensional continuous Gaussian prior
   - Empirical Fidelity: **74.90% Oracle Match Rate**
   - Training: 25 epochs with BCE reconstruction and KL-divergence loss

3. **Auxiliary CNN Classifier (`fashion_classifier.pt`)**:
   - Architecture: 4-layer Convolutional Neural Network with BatchNorm & Dropout
   - Test Accuracy: **90.76%** on real Fashion-MNIST test dataset

4. **Fine-Tuned DistilBERT (`distilbert_fashion/`)**:
   - Base Model: `distilbert-base-uncased`
   - Fine-tuned on natural language apparel prompt intent classification

## Licensing & Attribution
- Fashion-MNIST: Zalando SE (MIT License)
- AtelierAI Checkpoints: MIT License
"""

card_path = CKPT_DIR / "README.md"
card_path.write_text(model_card, encoding="utf-8")

print("\n[2/3] Uploading model files to Hugging Face Hub...")

# Files to upload
files_to_upload = [
    ("cgan_generator.pt", "cgan_generator.pt"),
    ("vae.pt", "vae.pt"),
    ("fashion_classifier.pt", "fashion_classifier.pt"),
    ("cgan_metrics.json", "cgan_metrics.json"),
    ("vae_metrics.json", "vae_metrics.json"),
    ("classifier_metrics.json", "classifier_metrics.json"),
    ("distilbert_metrics.json", "distilbert_metrics.json"),
    ("README.md", "README.md"),
]

for local_name, remote_name in files_to_upload:
    local_file = CKPT_DIR / local_name
    if local_file.exists():
        print(f"  Uploading {local_name} ({local_file.stat().st_size / (1024*1024):.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_name,
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
        )

# Upload distilbert_fashion directory
distilbert_dir = CKPT_DIR / "distilbert_fashion"
if distilbert_dir.exists():
    print("  Uploading distilbert_fashion folder...")
    api.upload_folder(
        folder_path=str(distilbert_dir),
        path_in_repo="distilbert_fashion",
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
    )

print("\n[3/3] Verifying uploaded files on Hugging Face Hub...")
repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model", token=token)
print("Files in repository:")
for f in repo_files:
    print(f"  - {f}")

print(f"\nHugging Face Model Hub upload complete: https://huggingface.co/{REPO_ID}")
