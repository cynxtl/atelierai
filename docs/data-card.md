# Data Card: Fashion-MNIST

## 1. Dataset Summary
- **Dataset Name**: Fashion-MNIST
- **Created by**: Zalando Research
- **Purpose**: Direct drop-in replacement for MNIST for benchmarking computer vision and generative algorithms.
- **Classes**: 10 clothing categories (0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot).
- **Split Sizes**: 60,000 training examples, 10,000 testing examples.
- **Dimensions**: 28 × 28 pixels, 1 channel (8-bit grayscale).

## 2. Preprocessing & Augmentation in AtelierAI
- **cGAN Normalization**: `transforms.Normalize((0.5,), (0.5,))` to map values into range `[-1.0, 1.0]` matching generator `Tanh` activations.
- **VAE Normalization**: `transforms.ToTensor()` in range `[0.0, 1.0]` matching `Sigmoid` pixel reconstructions.
- **Classifier Normalization**: `transforms.Normalize((0.2860,), (0.3530,))` with random horizontal flip (\(p=0.5\)) and random cropping with padding.

## 3. Class Overlap & Known Limitations
- High visual overlap between **Shirt** (Class 6) and **T-shirt/top** (Class 0) due to collar subtlety in low-resolution 28×28 grayscale silhouettes.
- Silhouette ambiguity between **Coat** (Class 4) and **Pullover** (Class 2) when sleeves and closures are closed.
- Distinct geometries in **Trouser** (Class 1) and footwear categories (**Sandal**, **Sneaker**, **Ankle boot**) result in superior classification match rates (>95%).
