# AtelierAI System Architecture

## 1. Overview

**AtelierAI** is a decoupled, local-first generative AI studio designed to bridge natural-language apparel descriptions to structured category conditioning and synthetic 28×28 grayscale garment generation on Fashion-MNIST.

The system is engineered for dual operation:
1. **Local Development**: FastAPI backend + React Vite TypeScript frontend with CUDA acceleration on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM).
2. **Hosted Production**: Gradio API adapter targeted for Hugging Face ZeroGPU Spaces, serving a React frontend hosted on Vercel Hobby.

---

## 2. Architecture Diagram

```text
                               +-------------------------------------+
                               |           Web Client / UI           |
                               |    React + Vite + TypeScript + TS   |
                               +-------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         | (Local Mode)                                    | (Hosted Mode)
                         v                                                 v
             +-----------------------+                         +-----------------------+
             |    FastAPI Backend    |                         |  Gradio API Adapter   |
             |   (backend/app/main)  |                         |  (ZeroGPU Spaces SDK) |
             +-----------------------+                         +-----------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |     AtelierInferenceEngine          |
                               |     (ml/inference/service.py)       |
                               +-------------------------------------+
                                  |         |         |         |
           +----------------------+         |         |         +----------------------+
           |                                |         |                                |
           v                                v         v                                v
+----------------------+        +-------------+  +-------------+        +----------------------+
| Qwen2.5-0.5B Parser  |        | cGAN (DCGAN)|  | C-VAE Model |        | Auxiliary CNN / CLIP |
| Structured Schema    |        | Latent 100D |  | Latent 32D  |        | Oracle & Alignment   |
+----------------------+        +-------------+  +-------------+        +----------------------+
```

---

## 3. Core Components

### A. Natural Language Understanding (`ml/models/llm_parser.py`)
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Output Validation**: Pydantic schema enforcing `{ "class_label": str, "count": int (1..8) }`.
- **Defensive Safeguards**: Regex prompt injection interception, length limits (500 chars), out-of-scope garment detection (hats, watches, eyewear, swimwear) with honest rejection responses.
- **Alternative Classifier**: `DistilBertClothingClassifier` fine-tuned on natural language prompts across all 10 Fashion-MNIST classes.

### B. Conditional Image Synthesis
- **Conditional GAN (`ml/models/cgan.py`)**:
  - Generator: Latent noise vector \(z \sim \mathcal{N}(0, I)\) (100-dim) concatenated with learned class embedding (50-dim) \(\rightarrow\) Dense linear projection to \(128 \times 7 \times 7\) \(\rightarrow\) Transposed Convolution Blocks \(\rightarrow\) Tanh activation (\(1 \times 28 \times 28\), values in \([-1, 1]\)).
  - Discriminator: Class label mapped to spatial feature map (\(1 \times 28 \times 28\)) concatenated as a 2nd channel to image \(\rightarrow\) Convolutional layers with LeakyReLU & Dropout \(\rightarrow\) BCEWithLogitsLoss with one-sided label smoothing (0.9).
- **Conditional VAE (`ml/models/vae.py`)**:
  - Encoder: Convolutional downsampling with spatial label conditioning to infer \(\mu\) and \(\log \sigma^2\) (32-dim latent space).
  - Reparameterization: \(z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2)\).
  - Decoder: Concatenates \(z\) with class embedding \(\rightarrow\) Transposed Convolutions \(\rightarrow\) Sigmoid (\(1 \times 28 \times 28\), values in \([0, 1]\)).
  - Loss: Binary Cross-Entropy reconstruction loss + \(\beta \times D_{KL}(q(z|x) \parallel p(z))\).

### C. Multi-Level Verification Oracles
- **Auxiliary CNN Classifier (`ml/models/classifier.py`)**:
  - Lightweight 2-block convolutional neural network trained independently on Fashion-MNIST test set (achieves **90.76% accuracy**).
  - Acts as an objective evaluation oracle to verify whether synthesized images match the user's requested category.
- **Multimodal Alignment (`ml/models/clip_evaluator.py`)**:
  - `openai/clip-vit-base-patch32`
  - Upscales grayscale 28×28 silhouettes to 224×224 3-channel RGB and computes cosine similarity against natural language prompts (e.g. *"a grayscale photo of an ankle boot"*).

---

## 4. Hardware Optimization (RTX 3060 6GB VRAM)
- PyTorch with CUDA 11.8 / 12.x auto-detection.
- Lazy model loading: models remain uninstantiated until their respective pipeline stage is invoked.
- Memory release: batch tensors are clamped, converted to base64 on CPU, and freed to prevent VRAM spikes.
- Gradio adapter configured for minimal GPU holding time using `@spaces.GPU` only around active generation.
