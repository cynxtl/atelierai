"""Conditional Generative Adversarial Network (cGAN) for Fashion-MNIST."""

import torch
import torch.nn as nn


def weights_init_normal(m):
    """Initializes weights using normal distribution (DCGAN standard)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Embedding") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator: maps random noise z and class label y to 28x28 grayscale image.
    Uses class embedding concatenated with latent noise vector.
    """
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        label_embed_dim: int = 50,
        channels: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        in_dim = latent_dim + label_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), labels: (B,)
        label_embed = self.label_embedding(labels)
        x = torch.cat([z, label_embed], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        img = self.deconv_blocks(x)
        return img


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator using Projection Discriminator architecture (Miyato & Koyama 2018).
    Extracts deep visual features directly from image channels, then scores validity via:
        D(x, y) = unconditioned_score(phi(x)) + <phi(x), w_y>
    Guarantees strict class-conditioning gradient flow and eliminates conditional mode collapse.
    """
    def __init__(
        self,
        num_classes: int = 10,
        channels: int = 1,
        img_size: int = 28,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = channels
        self.feature_dim = feature_dim

        self.conv_blocks = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Unconditioned logit: real vs fake image realism
        self.linear_uncond = nn.Linear(feature_dim, 1)

        # Class projection embedding: inner product phi(x) . w_y
        self.embed_y = nn.Embedding(num_classes, feature_dim)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # img: (B, 1, 28, 28), labels: (B,)
        h = self.conv_blocks(img)
        phi = self.fc(h)

        out = self.linear_uncond(phi)
        w_y = self.embed_y(labels)
        proj = torch.sum(phi * w_y, dim=1, keepdim=True)
        return out + proj
