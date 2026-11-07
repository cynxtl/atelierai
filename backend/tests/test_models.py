"""Unit tests for ML model architectures and tensor output dimensions."""

import pytest
import torch
from ml.models.cgan import ConditionalGenerator, ConditionalDiscriminator
from ml.models.vae import ConditionalVAE, vae_loss_function
from ml.models.classifier import FashionCNNClassifier


def test_cgan_generator_shape():
    generator = ConditionalGenerator(latent_dim=100, num_classes=10)
    batch_size = 4
    z = torch.randn(batch_size, 100)
    labels = torch.randint(0, 10, (batch_size,))
    out = generator(z, labels)
    assert out.shape == (batch_size, 1, 28, 28)
    # Tanh outputs must be in range [-1, 1]
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_cgan_discriminator_shape():
    discriminator = ConditionalDiscriminator(num_classes=10)
    batch_size = 4
    imgs = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    out = discriminator(imgs, labels)
    assert out.shape == (batch_size, 1)


def test_vae_reconstruction_and_sampling():
    vae = ConditionalVAE(latent_dim=32, num_classes=10)
    batch_size = 4
    imgs = torch.rand(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))

    recon_x, mu, logvar = vae(imgs, labels)
    assert recon_x.shape == (batch_size, 1, 28, 28)
    assert mu.shape == (batch_size, 32)
    assert logvar.shape == (batch_size, 32)
    # Sigmoid outputs must be in range [0, 1]
    assert recon_x.min() >= 0.0
    assert recon_x.max() <= 1.0

    # Test sampling from prior
    samples = vae.sample(labels, device=torch.device("cpu"))
    assert samples.shape == (batch_size, 1, 28, 28)


def test_vae_loss_computation():
    batch_size = 2
    recon = torch.full((batch_size, 1, 28, 28), 0.5)
    target = torch.full((batch_size, 1, 28, 28), 0.5)
    mu = torch.zeros(batch_size, 32)
    logvar = torch.zeros(batch_size, 32)

    total_loss, recon_loss, kld_loss = vae_loss_function(recon, target, mu, logvar)
    assert total_loss > 0
    assert kld_loss == 0.0  # At mu=0, logvar=0, KL divergence to N(0, I) is 0


def test_classifier_shape_and_prediction():
    classifier = FashionCNNClassifier(num_classes=10)
    batch_size = 4
    imgs = torch.randn(batch_size, 1, 28, 28)

    logits = classifier(imgs)
    assert logits.shape == (batch_size, 10)

    preds, confs = classifier.predict_classes(imgs)
    assert preds.shape == (batch_size,)
    assert confs.shape == (batch_size,)
    assert (confs >= 0.0).all() and (confs <= 1.0).all()
