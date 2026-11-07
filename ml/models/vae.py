"""Conditional Variational Autoencoder (C-VAE) for Fashion-MNIST."""


import torch
import torch.nn.functional as F
from torch import nn


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (C-VAE).
    Enables class-conditioned generation and direct comparison with cGAN,
    as well as latent reconstruction and interpolation.
    """
    def __init__(
        self,
        latent_dim: int = 32,
        num_classes: int = 10,
        label_embed_dim: int = 16,
        channels: int = 1,
        img_size: int = 28,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.img_size = img_size

        # Spatial label embedding for encoder
        self.encoder_label_embed = nn.Embedding(num_classes, img_size * img_size)

        # Vector label embedding for decoder
        self.decoder_label_embed = nn.Embedding(num_classes, label_embed_dim)

        # Encoder: (image + label_map) = 2 channels
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        decoder_in_dim = latent_dim + label_embed_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(decoder_in_dim, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder_deconv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Spatial condition concatenation
        label_map = self.encoder_label_embed(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        cond_x = torch.cat([x, label_map], dim=1)
        h = self.encoder(cond_x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.decoder_label_embed(labels)
        cond_z = torch.cat([z, label_embed], dim=1)
        h = self.decoder_fc(cond_z)
        h = h.view(h.size(0), 64, 7, 7)
        recon_x = self.decoder_deconv(h)
        return recon_x

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar

    def sample(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        num_samples = labels.size(0)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, labels)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes BCE Reconstruction Loss + beta * KL Divergence.
    """
    # Reconstruction loss across all pixels
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    # KL divergence loss: D_KL( q(z|x) || p(z) )
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss
