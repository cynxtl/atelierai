"""Training script for Conditional GAN (cGAN) on Fashion-MNIST."""

import argparse
import json
import logging
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from ml.configs.config import CGANConfig, FASHION_CLASSES, REPORTS_DIR
from ml.data.fashion_mnist import get_fashion_mnist_dataloaders
from ml.models.cgan import ConditionalGenerator, ConditionalDiscriminator, weights_init_normal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_grid(generator: nn.Module, latent_dim: int, device: torch.device, output_path: Path):
    """Generates a grid showing 8 samples per class (10 rows x 8 columns) and saves image."""
    generator.eval()
    num_classes = 10
    samples_per_class = 8
    total_samples = num_classes * samples_per_class

    labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
    # Fixed seed for reproducible sample grid visual tracking
    torch.manual_seed(42)
    z = torch.randn(total_samples, latent_dim, device=device)

    with torch.no_grad():
        gen_imgs = generator(z, labels)
        # Denormalize from [-1, 1] to [0, 1]
        gen_imgs = (gen_imgs + 1.0) / 2.0
        grid = make_grid(gen_imgs, nrow=samples_per_class, padding=2, normalize=False)
        save_image(grid, output_path)
    logger.info(f"Saved sample grid to {output_path}")


def train_cgan(config: CGANConfig):
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    train_loader, _, _ = get_fashion_mnist_dataloaders(
        for_model="cgan",
        batch_size=config.batch_size,
        seed=config.seed,
        val_split=0.0,
    )

    # Initialize generator and discriminator
    generator = ConditionalGenerator(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        label_embed_dim=config.label_embed_dim,
        channels=config.channels,
    ).to(device)

    discriminator = ConditionalDiscriminator(
        num_classes=config.num_classes,
        channels=config.channels,
        img_size=config.img_size,
    ).to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    adversarial_loss = nn.BCEWithLogitsLoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr_g, betas=(config.b1, config.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.b1, config.b2))

    history = {"g_loss": [], "d_loss": []}
    start_time = time.time()
    logger.info(f"Starting cGAN training for {config.epochs} epochs...")

    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        batch_count = 0

        for real_imgs, labels in train_loader:
            batch_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # Adversarial ground truths with one-sided label smoothing (0.9 instead of 1.0)
            valid = torch.full((batch_size, 1), 0.9, device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # 1. Real matching images loss
            d_real_out = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(d_real_out, valid)

            # 2. Real mismatched images (mismatched negative pairs to enforce conditioning)
            offset = torch.randint(1, config.num_classes, (batch_size,), device=device)
            wrong_labels = (labels + offset) % config.num_classes
            d_wrong_out = discriminator(real_imgs, wrong_labels)
            d_wrong_loss = adversarial_loss(d_wrong_out, fake)

            # 3. Fake generated images loss
            z = torch.randn(batch_size, config.latent_dim, device=device)
            gen_labels = torch.randint(0, config.num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)

            d_fake_out = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(d_fake_out, fake)

            d_loss = (d_real_loss + 0.5 * d_wrong_loss + 0.5 * d_fake_loss)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generator aims to make Discriminator output 1.0 (valid)
            g_out = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(g_out, valid)

            g_loss.backward()
            optimizer_G.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            batch_count += 1

        epoch_g_loss = running_g_loss / batch_count
        epoch_d_loss = running_d_loss / batch_count

        history["g_loss"].append(round(epoch_g_loss, 4))
        history["d_loss"].append(round(epoch_d_loss, 4))

        logger.info(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f}"
        )

        # Generate sample grid every 5 epochs and at the end
        if epoch % 5 == 0 or epoch == config.epochs:
            grid_path = REPORTS_DIR / f"cgan_samples_epoch_{epoch}.png"
            generate_sample_grid(generator, config.latent_dim, device, grid_path)

    # Save final generator and discriminator checkpoints
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": generator.state_dict(),
        "config": config.__dict__,
    }, config.checkpoint_path)

    torch.save({
        "epoch": config.epochs,
        "model_state_dict": discriminator.state_dict(),
        "config": config.__dict__,
    }, config.discriminator_checkpoint)

    # Save canonical sample grid
    generate_sample_grid(generator, config.latent_dim, device, REPORTS_DIR / "cgan_samples_final.png")

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"cGAN training completed in {elapsed_time}s.")

    # Save metrics
    metrics = {
        "final_g_loss": history["g_loss"][-1],
        "final_d_loss": history["d_loss"][-1],
        "history": history,
        "training_time_seconds": elapsed_time,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "latent_dim": config.latent_dim,
        "lr_g": config.lr_g,
        "lr_d": config.lr_d,
    }

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved cGAN metrics to {config.metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional GAN on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=0.0002, help="Discriminator learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = CGANConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        seed=args.seed,
    )
    train_cgan(cfg)
