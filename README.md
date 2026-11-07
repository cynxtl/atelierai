# AtelierAI: Text-to-Conditional-Fashion Synthesis Studio

> **An independent, production-engineered AI application that converts natural-language apparel descriptions into verified grayscale clothing images using conditional generative modeling.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.3-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

**AtelierAI** is a decoupled full-stack generative synthesis studio that translates free-form natural language fashion requests (e.g. *"Show me four ankle boots"*) into verified $28 \times 28$ grayscale garment images.

The platform combines five specialized model components across distinct generative, discriminative, causal, and multimodal paradigms:
1. **Qwen2.5-0.5B-Instruct**: Causal autoregressive language model for prompt parsing, entity extraction, and strict Pydantic JSON schema validation.
2. **Conditional GAN (cGAN)**: Deep convolutional adversarial network conditioning on 10 Fashion-MNIST apparel classes from latent noise $z \sim \mathcal{N}(0, I)$.
3. **Conditional VAE (C-VAE)**: Continuous probabilistic latent autoencoder for comparative synthesis, probabilistic reconstruction, and ELBO evaluation.
4. **Auxiliary CNN Classifier**: Independently trained convolutional evaluation oracle (**90.76% test accuracy**) measuring objective class-fidelity.
5. **OpenAI CLIP ViT-B/32**: Multimodal vision-language contrastive model evaluating zero-shot semantic alignment.
6. **Fine-Tuned DistilBERT**: Supervised text classification baseline for natural language clothing intents.

---

## 2. Product Features & Honest Scope

- **Honest Domain Scope**: Operates strictly on Fashion-MNIST's 10 apparel categories in $28 \times 28$ grayscale silhouettes. Does not fabricate photorealistic textures, arbitrary colors, or unconstrained garments.
- **Deterministic Guardrails**: Rejects unsupported items (e.g., watches, hats, jewelry, sunglasses, swimwear) with structured feedback rather than hallucinating an arbitrary category.
- **Prompt Injection Defense**: Intercepts and neutralizes system prompt override patterns before language model execution.
- **Side-by-Side Generative Comparison**: Visual and quantitative comparison between cGAN adversarial generation (sharp, high-contrast) and VAE probabilistic sampling (smooth, mode-stable).
- **Dual Runtime Architecture**: Core inference services are decoupled from API layers, supporting both local **FastAPI** development and hosted **Hugging Face ZeroGPU Gradio** deployments.

---

## 3. System Architecture & Lifecycle

```text
                               +-------------------------------------+
                               |           Web Client / UI           |
                               |    React + Vite + TypeScript + TS   |
                               +-------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         | (Local Mode: :5173 -> :8000)                    | (Hosted Mode: Gradio SDK)
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

### End-to-End Generation Pipeline:
1. **User Prompt**: e.g., *"Generate 2 comfortable leather sandals for the beach."*
2. **Text Parsing**: `QwenClothingParser` checks prompt safety, filters injection, and emits validated JSON: `{"class_label": "sandal", "count": 2}`.
3. **Conditioned Synthesis**: Generator embeds `sandal` (Class #5) and generates 2 synthetic $1 \times 28 \times 28$ tensors.
4. **Oracle Verification**: Auxiliary CNN classifies synthetic images to confirm category match and confidence percentage.
5. **Multimodal Alignment**: CLIP ViT-B/32 computes cosine similarity between synthetic image embeddings and `"a grayscale photo of a sandal"`.
6. **Gallery Delivery**: Images are converted to base64 PNGs and returned alongside full latency and alignment metrics.

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Frontend | React + Vite + TypeScript | React 19, Vite 8, TS 5.7 | Creative synthesis studio interface |
| Styling | Tailwind CSS | 4.3 | Minimalist studio typography & theme |
| Local Backend | FastAPI + Uvicorn | FastAPI 0.136, Uvicorn 0.34 | High-throughput asynchronous local REST API |
| Deployment Adapter | Gradio SDK | 6.31 | Hugging Face ZeroGPU API integration |
| Deep Learning | PyTorch + Torchvision | PyTorch 2.7.1+cu118 | CUDA neural network training & inference |
| Causal LLM | Qwen2.5-0.5B-Instruct | Transformers 4.40+ | In-context prompt parsing & JSON schema |
| Text Classifier | Fine-Tuned DistilBERT | distilbert-base-uncased | Supervised text classification baseline |
| Multimodal Model | CLIP ViT-B/32 | openai/clip-vit-base-patch32 | Zero-shot image-text alignment scoring |
| Dataset | Fashion-MNIST | 60,000 train / 10,000 test | Standardized apparel benchmark dataset |

---

## 5. Prerequisites & Local Installation

### Hardware & Environment
- **OS**: Windows 11 / Linux
- **GPU**: NVIDIA GPU with CUDA support recommended (e.g., RTX 3060 6 GB VRAM). CPU inference supported automatically.
- **Python**: Python 3.11 or 3.12
- **Node.js**: Node 18+ and npm 10+

### Setup Instructions

```powershell
# 1. Clone repository
git clone https://github.com/your-username/FashionAI.git
cd FashionAI

# 2. Install backend dependencies
python -m pip install -r backend/requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## 6. Training Pipelines

All training scripts are fully reproducible, accept configurable seeds and hyperparameters, and save metrics alongside checkpoints.

### Train the Auxiliary Image Classifier (CNN Oracle)
```powershell
python -m ml.training.train_classifier --epochs 8 --batch-size 128 --lr 0.001
```
*Output: `ml/checkpoints/fashion_classifier.pt` (Test Accuracy: **90.76%**)*

### Train the Conditional GAN (cGAN)
```powershell
python -m ml.training.train_cgan --epochs 25 --batch-size 64 --lr-g 0.0002 --lr-d 0.0002
```
*Output: `ml/checkpoints/cgan_generator.pt`, `ml/checkpoints/cgan_metrics.json`, and sample grids in `reports/`*

### Train the Conditional VAE (C-VAE)
```powershell
python -m ml.training.train_vae --epochs 25 --batch-size 64 --lr 0.001 --beta 1.0
```
*Output: `ml/checkpoints/vae.pt`, `ml/checkpoints/vae_metrics.json`, and reconstruction grids in `reports/`*

### Fine-Tune DistilBERT on Natural Language Clothing Prompts
```powershell
python -m ml.training.train_distilbert --epochs 4 --batch-size 16 --lr 3e-5
```
*Output: `ml/checkpoints/distilbert_fashion/`*

---

## 7. Running the Local Application

### Step 1: Start FastAPI Local Backend
```powershell
# Terminal 1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Swagger API documentation will be available at `http://127.0.0.1:8000/docs`.

### Step 2: Start React Frontend
```powershell
# Terminal 2
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 8. Empirical Benchmark Evaluation

Execute the complete automated benchmark suite:
```powershell
python -m scripts.run_benchmarks --samples-per-class 100
```

### A. Natural Language Benchmark (Standalone LLM vs Fallback vs Combined)

We evaluated natural language understanding across the required 10-prompt benchmark, isolating raw model performance from deterministic fallbacks:

| Evaluation Mode | Category Accuracy | Count Accuracy | Schema Validation |
|---|:---:|:---:|:---:|
| **Standalone Qwen Zero-Shot** (`allow_fallback=False`) | **30.0%** | 30.0% | 100.0% |
| **Standalone Qwen Few-Shot** (`allow_fallback=False`) | **90.0%** | 90.0% | 100.0% |
| **Standalone Rule Fallback** | **100.0%** | 100.0% | 100.0% |
| **Combined Pipeline (Production)** | **100.0%** | 100.0% | 100.0% |
| **Fine-Tuned DistilBERT Baseline** | **60.0%** | N/A | N/A |

### B. 20-Request Robustness & Safety Benchmark
- **Robustness Pass Rate**: **95.0%** (19/20) across synonyms, ambiguities, paraphrases, and quantity extremes.
- **Layer V Responsible AI Pass Rate**: **100.0%** (Interception of out-of-scope apparel, prompt injection patterns, and empty queries).

### C. Image Generation Fidelity (cGAN v2 vs Baseline v1 vs VAE, 100 Samples/Class)

| Fashion-MNIST Category | cGAN Baseline (v1) | cGAN Projection (v2) | VAE (C-VAE) | Average CLIP (cGAN) |
|---|:---:|:---:|:---:|:---:|
| 0. T-shirt/top | 94.0% | **99.0%** | 89.0% | 0.3076 |
| 1. Trouser | 100.0% | **100.0%** | 100.0% | 0.2983 |
| 2. Pullover | **0.0%** | **97.0%** | 100.0% | 0.3278 |
| 3. Dress | 100.0% | **100.0%** | 92.0% | 0.2880 |
| 4. Coat | 99.0% | **96.0%** | 22.0% | 0.3020 |
| 5. Sandal | 99.0% | **100.0%** | 50.0% | 0.2623 |
| 6. Shirt | 100.0% | **84.0%** | 11.0% | 0.2748 |
| 7. Sneaker | **23.0%** | **97.0%** | 86.0% | 0.2848 |
| 8. Bag | 100.0% | **100.0%** | 99.0% | 0.3209 |
| 9. Ankle boot | **0.0%** | **99.0%** | 100.0% | 0.3081 |
| **Overall Class-Match Rate** | **71.90%** | **97.20%** | **74.90%** | **0.2975 (cosine)** |

*Note on CLIP Metrics: Cosine similarities are uncalibrated embedding distances on $28 \times 28$ grayscale tensors and must not be interpreted as calibrated classification probabilities.*

---

## 9. Automated Testing

Run the test suite:
```powershell
# Run Python backend and model unit tests
pytest backend/tests/ -v

# Run Frontend TypeScript verification
cd frontend
npm run build
```

---

## 10. Licensing & Dataset Attribution

- **Fashion-MNIST**: Copyright (c) 2017 Zalando SE, released under the **MIT License**.
- **Qwen2.5-0.5B-Instruct**: Developed by Alibaba Cloud, released under **Apache 2.0**.
- **CLIP ViT-B/32**: Developed by OpenAI, released under the **MIT License**.
- **DistilBERT**: Developed by Hugging Face, released under **Apache 2.0**.
- **AtelierAI Application**: Copyright (c) 2026 AtelierAI Contributors, released under the **MIT License**.
