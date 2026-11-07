# Model Card: Conditional VAE (C-VAE)

## Model Overview
- **Model Type**: Convolutional Conditional Variational Autoencoder
- **Framework**: PyTorch 2.7
- **Input Dimensions**: Grayscale image \(X \in [0, 1]^{1 \times 28 \times 28}\), Class label \(y \in \{0, \dots, 9\}\)
- **Latent Dimension**: 32
- **Output Dimensions**: Reconstructed / Sampled image \(X_{recon} \in [0, 1]^{1 \times 28 \times 28}\)

## Architecture Details
- **Encoder**:
  - Concatenates spatial label map to input image (\(2 \times 28 \times 28\))
  - Conv Block 1: Conv2d(2, 32, 4, stride=2, padding=1), BatchNorm2d, LeakyReLU(0.2)
  - Conv Block 2: Conv2d(32, 64, 4, stride=2, padding=1), BatchNorm2d, LeakyReLU(0.2)
  - Dense projections: \(\mu \in \mathbb{R}^{32}\), \(\log \sigma^2 \in \mathbb{R}^{32}\)
- **Reparameterization**:
  - \(\epsilon \sim \mathcal{N}(0, I)\)
  - \(z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2)\)
- **Decoder**:
  - Class embedding: `nn.Embedding(10, 16)`
  - Vector input: \(32 + 16 = 48\)
  - Projection to \(64 \times 7 \times 7\)
  - ConvTranspose2d(64, 32, 4, 2, 1) \(\rightarrow 32 \times 14 \times 14\)
  - ConvTranspose2d(32, 1, 4, 2, 1) \(\rightarrow 1 \times 28 \times 28\), Sigmoid()

## Training Configuration
- Optimizer: Adam (\(\alpha = 0.001\))
- Loss Formulation: Binary Cross-Entropy Reconstruction + \(\beta \times D_{KL}\) (\(\beta = 1.0\))
- Batch Size: 64
- Epochs: 25
- Checkpoint Path: `ml/checkpoints/vae.pt`
