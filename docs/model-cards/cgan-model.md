# Model Card: Conditional GAN (cGAN)

## Model Overview
- **Model Type**: Deep Convolutional Conditional Generative Adversarial Network (cGAN)
- **Framework**: PyTorch 2.7 (CUDA accelerated)
- **Input Dimensions**: Latent noise vector \(z \in \mathbb{R}^{100}\), Class label \(y \in \{0, \dots, 9\}\)
- **Output Dimensions**: Grayscale image tensor \(X \in [-1, 1]^{1 \times 28 \times 28}\)

## Architecture Details
- **Generator**:
  - Class embedding: `nn.Embedding(10, 50)`
  - Combined latent input: \(100 + 50 = 150\) dimensions
  - Linear projection: \(150 \rightarrow 128 \times 7 \times 7\), BatchNorm1d, LeakyReLU(0.2)
  - Deconvolution 1: ConvTranspose2d(128, 64, 4, stride=2, padding=1) \(\rightarrow 64 \times 14 \times 14\)
  - Deconvolution 2: ConvTranspose2d(64, 1, 4, stride=2, padding=1) \(\rightarrow 1 \times 28 \times 28\), Tanh()
- **Discriminator**:
  - Class spatial embedding: `nn.Embedding(10, 28 * 28)` reshaped to \(1 \times 28 \times 28\)
  - Concatenated input: \(2 \times 28 \times 28\)
  - Conv Block 1: Conv2d(2, 64, 4, stride=2, padding=1), LeakyReLU(0.2), Dropout2d(0.25)
  - Conv Block 2: Conv2d(64, 128, 4, stride=2, padding=1), BatchNorm2d, LeakyReLU(0.2), Dropout2d(0.25)
  - Classifier: Linear(128 * 7 * 7, 1)

## Training Configuration
- Optimizer: Adam (\(\alpha_G = 0.0002, \alpha_D = 0.0002, \beta_1 = 0.5, \beta_2 = 0.999\))
- Loss: `nn.BCEWithLogitsLoss()` with one-sided label smoothing (0.9 on real samples)
- Batch Size: 64
- Epochs: 25
- Reproducibility Seed: 42
