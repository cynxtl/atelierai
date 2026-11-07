"""Training script for Conditional Variational Autoencoder (C-VAE) on Fashion-MNIST."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import optim
from torchvision.utils import make_grid, save_image

from ml.configs.config import REPORTS_DIR, VAEConfig
from ml.data.fashion_mnist import get_fashion_mnist_dataloaders
from ml.models.vae import ConditionalVAE, vae_loss_function

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_vae_sample_grid(model: ConditionalVAE, device: torch.device, output_path: Path):
    """Generates a sample grid from the prior p(z) ~ N(0, I) for all 10 classes."""
    model.eval()
    num_classes = 10
    samples_per_class = 8
    labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
    torch.manual_seed(42)

    with torch.no_grad():
        samples = model.sample(labels, device=device)
        grid = make_grid(samples, nrow=samples_per_class, padding=2, normalize=False)
        save_image(grid, output_path)
    logger.info(f"Saved VAE sample grid to {output_path}")


def generate_reconstruction_grid(model: ConditionalVAE, val_loader, device: torch.device, output_path: Path):
    """Generates an alternating grid of original test images vs their VAE reconstructions."""
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images[:16].to(device), labels[:16].to(device)
            recon, _, _ = model(images, labels)
            # Stack real and recon alternating
            comparison = torch.cat([images.unsqueeze(1), recon.unsqueeze(1)], dim=1).view(-1, 1, 28, 28)
            grid = make_grid(comparison, nrow=8, padding=2, normalize=False)
            save_image(grid, output_path)
            break
    logger.info(f"Saved VAE reconstruction comparison to {output_path}")


def train_vae(config: VAEConfig):
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_loader, val_loader, _ = get_fashion_mnist_dataloaders(
        for_model="vae",
        batch_size=config.batch_size,
        seed=config.seed,
        val_split=0.1,
    )

    model = ConditionalVAE(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        label_embed_dim=config.label_embed_dim,
        channels=config.channels,
        img_size=config.img_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    history = {"train_loss": [], "recon_loss": [], "kld_loss": [], "val_loss": []}
    start_time = time.time()
    best_val_loss = float("inf")
    logger.info(f"Starting C-VAE training for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_kld_loss = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            recon_x, mu, logvar = model(images, labels)
            loss, recon, kld = vae_loss_function(recon_x, images, mu, logvar, beta=config.beta)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_recon_loss += recon.item()
            train_kld_loss += kld.item()
            batch_count += 1

        avg_train_loss = train_total_loss / batch_count
        avg_recon_loss = train_recon_loss / batch_count
        avg_kld_loss = train_kld_loss / batch_count

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                recon_x, mu, logvar = model(images, labels)
                loss, _, _ = vae_loss_function(recon_x, images, mu, logvar, beta=config.beta)
                val_total_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_total_loss / val_batches

        history["train_loss"].append(round(avg_train_loss, 2))
        history["recon_loss"].append(round(avg_recon_loss, 2))
        history["kld_loss"].append(round(avg_kld_loss, 2))
        history["val_loss"].append(round(avg_val_loss, 2))

        logger.info(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.2f} (Recon: {avg_recon_loss:.2f}, KLD: {avg_kld_loss:.2f}) | "
            f"Val Loss: {avg_val_loss:.2f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": config.__dict__,
            }, config.checkpoint_path)

        if epoch % 5 == 0 or epoch == config.epochs:
            grid_path = REPORTS_DIR / f"vae_samples_epoch_{epoch}.png"
            generate_vae_sample_grid(model, device, grid_path)

    # Save final sample grid and reconstruction comparison
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    generate_vae_sample_grid(model, device, REPORTS_DIR / "vae_samples_final.png")
    generate_reconstruction_grid(model, val_loader, device, REPORTS_DIR / "vae_reconstruction_final.png")

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"VAE training completed in {elapsed_time}s.")

    metrics = {
        "final_train_loss": history["train_loss"][-1],
        "final_recon_loss": history["recon_loss"][-1],
        "final_kld_loss": history["kld_loss"][-1],
        "best_val_loss": round(best_val_loss, 2),
        "history": history,
        "training_time_seconds": elapsed_time,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "latent_dim": config.latent_dim,
        "beta": config.beta,
    }

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved VAE metrics to {config.metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional VAE on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight for KLD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = VAEConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
    )
    train_vae(cfg)
