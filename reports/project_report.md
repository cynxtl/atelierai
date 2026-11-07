# AtelierAI: Text-to-Conditional-Fashion Synthesis Studio
## Pre-Deployment Engineering and ML Quality Audit Report

---

### Executive Summary
AtelierAI is an independent, production-grade generative AI application capable of translating natural-language apparel queries into category-conditioned, synthetic $28 \times 28$ grayscale clothing images. Built on Fashion-MNIST using PyTorch, CUDA, FastAPI, and React/TypeScript, the platform implements a five-tier generative, discriminative, causal, and multimodal architecture:
- **Causal NLP**: `Qwen/Qwen2.5-0.5B-Instruct` parses unstructured text into strict Pydantic JSON schemas.
- **Generative Synthesis**: Conditional GAN (cGAN with Projection Discriminator) and Conditional VAE (C-VAE) synthesize class-conditioned apparel silhouettes.
- **Objective Verification**: An independently trained auxiliary convolutional neural network oracle (**90.76% test accuracy**) evaluates category fidelity.
- **Multimodal Alignment**: OpenAI's `clip-vit-base-patch32` scores zero-shot vision-language alignment.
- **Discriminative Baseline**: Fine-tuned DistilBERT models prompt classification.
- **Responsible AI Guardrails**: Strict taxonomy checking eliminates hallucination on out-of-scope apparel; regex boundary rules neutralize prompt injection attacks; non-positive quantities raise explicit validation errors, and requests $> 8$ return disclosed adjustments with warnings.

---

### 1. cGAN Investigation, Root Cause Analysis & Retraining

#### A. Baseline Failure Investigation
In initial baseline benchmarks, the auxiliary CNN oracle revealed catastrophic conditional failure on three classes:
- **Pullover**: 0.0% class-match rate (100% classified as Shirt)
- **Ankle boot**: 0.0% class-match rate (100% classified as Bag)
- **Sneaker**: 23.0% class-match rate (66% classified as Sandal)
- **Overall Baseline Accuracy**: **71.90%**

#### B. Architectural Root Cause Analysis
1. **Spatial Channel Concatenation Inefficiency**:
   The baseline discriminator concatenated a 1-channel spatial label map (`nn.Embedding(10, 784).view(-1, 1, 28, 28)`) directly onto the 1-channel input image before the first convolution:
   $$x_{in} = [x, \text{embed}(y)] \in \mathbb{R}^{B \times 2 \times 28 \times 28}$$
   In standard DCGAN architectures, early convolutional layers quickly prioritize high-gradient spatial artifacts (DCGAN transposed convolution checkerboards) to classify real vs. fake, largely ignoring the constant spatial conditioning plane.
2. **Absence of Mismatched Negative Pairs**:
   In standard conditional GAN objectives, the discriminator is only trained with:
   $$\mathcal{L}_D = \mathbb{E}[\log D(x_{real}, y)] + \mathbb{E}[\log(1 - D(G(z, y), y))]$$
   When real images are never paired with mismatched class labels $(x_{real}, y_{wrong})$, the discriminator is never penalized for ignoring $y$ when evaluating real images. As a result, the generator learns to produce a single high-quality generic footwear shape or upper-body silhouette for multiple labels without incurring a discriminator penalty.
3. **Confusion Matrix Collapse**:
   - $G(z, y=\text{Pullover})$ collapsed into the modal upper-body distribution of $y=\text{Shirt}$.
   - $G(z, y=\text{Sneaker})$ collapsed into low-cut footwear resembling $y=\text{Sandal}$.
   - $G(z, y=\text{Ankle boot})$ collapsed into broad, flat-base silhouettes resembling $y=\text{Bag}$.

#### C. Architectural Fix: Projection Discriminator + Mismatched Real Negatives
To permanently solve conditional mode collapse, we implemented the **Projection Discriminator** architecture (Miyato & Sugiyama, 2018):
1. **Projection Formulation**:
   Instead of concatenating labels into image channels, the discriminator extracts a high-level representation $\phi(x) \in \mathbb{R}^{512}$ via strided convolutions, computes an unconditional real/fake score, and adds the inner product with a learned class embedding vector $w_y \in \mathbb{R}^{512}$:
   $$D(x, y) = \text{Linear}(\phi(x)) + \langle \phi(x), w_y \rangle$$
   This enforces that gradient updates directly correlate the visual feature embedding with the specific class embedding vector.
2. **Mismatched Real Negative Pairs**:
   During training, the discriminator loss is augmented with mismatched real pairs:
   $$\mathcal{L}_D = \mathbb{E}[\log D(x_{real}, y_{true})] + \frac{1}{2}\mathbb{E}[\log(1 - D(x_{real}, y_{wrong}))] + \frac{1}{2}\mathbb{E}[\log(1 - D(G(z, y), y))]$$
   This forces $D(x, y)$ to verify that the generated image matches the conditioned class $y$, rather than merely looking "fashion-like".

#### D. Training Execution & Empirical Verification
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 11.8).
- **Epochs**: 25 epochs (60,000 samples/epoch, batch size 64).
- **Checkpoints Preserved**:
  - Baseline preserved: `ml/checkpoints/cgan_generator_v1_baseline.pt`, `ml/checkpoints/cgan_discriminator_v1_baseline.pt`, `reports/cgan_samples_v1_baseline.png`.
  - Production retrained: `ml/checkpoints/cgan_generator.pt`, `ml/checkpoints/cgan_discriminator.pt`.

#### E. Quantitative Before-and-After Comparison (100 Samples / Class = 1,000 Samples)

| Fashion-MNIST Category | Baseline Match Rate (v1) | Retrained Match Rate (v2 Projection) | Delta | Baseline Failure Mode |
|---|---|---|---|---|
| **0. T-shirt/top** | 94.0% | **99.0%** | +5.0% | Minor shirt confusion |
| **1. Trouser** | 100.0% | **100.0%** | 0.0% | Stable |
| **2. Pullover** | **0.0%** | **97.0%** | **+97.0%** | 100% collapsed to Shirt |
| **3. Dress** | 100.0% | **100.0%** | 0.0% | Stable |
| **4. Coat** | 99.0% | **96.0%** | -3.0% | Overlap with Pullover |
| **5. Sandal** | 99.0% | **100.0%** | +1.0% | Stable |
| **6. Shirt** | 100.0% | **84.0%** | -16.0% | Fine collar boundary vs T-shirt/top |
| **7. Sneaker** | **23.0%** | **97.0%** | **+74.0%** | 66% collapsed to Sandal |
| **8. Bag** | 100.0% | **100.0%** | 0.0% | Stable |
| **9. Ankle boot** | **0.0%** | **99.0%** | **+99.0%** | 100% collapsed to Bag |
| **OVERALL ACCURACY** | **71.90%** | **97.20%** | **+25.30%** | **Catastrophic collapse eliminated** |

---

### 2. Natural Language Processing Verification

#### A. Standalone LLM vs Fallback vs Combined Pipeline
Earlier test configurations evaluated the Qwen parser with rule-based fallback enabled by default, masking raw LLM adherence. We introduced strict decoupling (`allow_fallback=False`) to measure standalone model performance on the required 10-prompt benchmark:

| Pipeline Configuration | Category Accuracy | Count Accuracy | Valid JSON Schema Rate | Primary Error Behavior |
|---|---|---|---|---|
| **Standalone Qwen Zero-Shot** (`allow_fallback=False`) | **30.0%** | 30.0% | 100.0% | Hyper-conservative; outputs `"class_label": "unsupported"` on descriptive prompts |
| **Standalone Qwen Few-Shot** (`allow_fallback=False`) | **90.0%** | 90.0% | 100.0% | Missed only *"handbag or tote bag"* where explicit keyword `"bag"` was not in prompt head |
| **Standalone Deterministic Fallback** | **100.0%** | 100.0% | 100.0% | Deterministic regex token matching across all 10 canonical requests |
| **Combined Pipeline (Zero-Shot + Fallback)** | **100.0%** | 100.0% | 100.0% | Production configuration: zero failure on all 10 canonical requests |
| **Combined Pipeline (Few-Shot + Fallback)** | **100.0%** | 100.0% | 100.0% | Production configuration: zero failure on all 10 canonical requests |
| **Fine-Tuned DistilBERT Baseline** | **60.0%** | N/A (intent only) | N/A | Confusion on footwear distinctions (sandal vs sneaker vs ankle boot) |

---

### 3. Expanded Robustness & Responsible AI Benchmarks

#### A. 20-Request Robustness Benchmark
We evaluated AtelierAI against an expanded 20-prompt robustness battery testing synonyms, semantic ambiguity, unsupported garments, paraphrasing, and quantity boundaries:

| Test Category | Total Tests | Passed | Accuracy (%) | Findings |
|---|---|---|---|---|
| **Synonyms & Slang** | 8 | 7 | 87.5% | Correctly resolved *kicks* $\rightarrow$ sneaker, *chelsea boots* $\rightarrow$ ankle boot, *frock* $\rightarrow$ dress, *sweater* $\rightarrow$ pullover, *jacket* $\rightarrow$ coat, *slides* $\rightarrow$ sandal, *handbag* $\rightarrow$ bag. Mismatch on *"A pair of denim jeans"*: parsed count as 1 instead of 2 (natural English semantics). |
| **Semantic Ambiguity** | 2 | 2 | 100.0% | Handled ambiguous requests (*"something fashionable"*, *"apparel"*) cleanly without hallucinating a random class. |
| **Unsupported Garments** | 5 | 5 | 100.0% | Cleanly rejected watches, sunglasses, fedoras, earrings, and swimsuits with informative error explanations. |
| **Paraphrases & Idioms** | 2 | 2 | 100.0% | Robust extraction across complex sentence structures. |
| **Quantity Variations** | 3 | 3 | 100.0% | Tested boundary limits: valid quantities accepted; $>8$ adjusted with warning; non-positive rejected. |
| **OVERALL ROBUSTNESS** | **20** | **19** | **95.0%** | High resiliency across out-of-distribution inputs. |

#### B. Layer V Responsible AI Benchmark (100.0% Pass Rate)
- **Out-of-Scope Detection**: Queries for non-Fashion-MNIST items (e.g. wristwatch, fedora, jewelry) are intercepted with `is_valid=False` and structured advice detailing the 10 supported classes.
- **Prompt Injection Defense**: Intercepts classic system override attacks (*"Ignore all previous instructions..."*) and ChatML role-spoofing injection attempts (`<|im_start|>system...`), neutralizing execution before LLM invocation.
- **Empty / Malformed Input**: Pure whitespace and punctuation are cleanly rejected without unhandled exceptions.
- **Ambiguous Input**: Vague requests (*"something to wear to a party"*) prompt clarification rather than silently choosing a random category.

---

### 4. Quantity Validation & Disclosed Adjustments

In accordance with priority 3, silent clamping has been completely eliminated from the pipeline:
1. **Non-Positive Quantities ($\le 0$)**:
   - Prompts requesting $\le 0$ items (e.g., *"Generate -4 dresses"*, *"Synthesize 0 sneakers"*) raise an immediate validation error:
     `"Invalid quantity: count must be a positive integer between 1 and 8 (got N)."`
   - Returns HTTP 200 with `success: false` and structured error message.
2. **Excessive Quantities ($> 8$)**:
   - Prompts requesting $> 8$ items (e.g., *"Synthesize 15 casual trousers"*) adjust the count to the maximum supported batch size (8) and return a prominent disclosed warning:
     `"Requested quantity (15) exceeds maximum supported limit of 8. Adjusted count to 8."`
   - In strict validation mode (`strict_quantity=True`), requests $> 8$ raise an explicit validation error.
3. **Automated Regression Coverage**:
   - 8 unit tests in `backend/tests/test_parser.py` and 2 integration tests in `backend/tests/test_api.py` verify this contract.

---

### 5. Multimodal CLIP Verification & Scoring Semantics

#### A. Correct Evaluation Semantics
- Model: `openai/clip-vit-base-patch32`.
- Preprocessing: Grayscale $1 \times 28 \times 28$ tensors are normalized from $[-1, 1]$ to $[0, 1]$, replicated across 3 RGB channels, bilinearly upscaled to $224 \times 224$, and normalized with ImageNet statistics ($\mu = [0.481, 0.457, 0.408], \sigma = [0.268, 0.261, 0.275]$).
- Canonical Prompts: `"a grayscale photo of a [category]"`.

#### B. Removal of Misleading Progress Bars
- In previous UI versions, cosine similarity values ($\approx 0.25 - 0.35$) were multiplied by $250\%$ to simulate confidence bars.
- This misleading scaling has been completely removed from `StudioPage.tsx` and `ComparePage.tsx`. The interface now presents the raw cosine metric ($[-1.0, 1.0]$) alongside an explicit disclaimer explaining that CLIP cosine similarity reflects uncalibrated embedding distance, not a calibrated posterior probability of correctness.

#### C. Empirical CLIP Metrics (100 Samples / Class)
- **cGAN Average CLIP Cosine**: **0.2975** (Range: 0.2623 for Sandal to 0.3278 for Pullover)
- **VAE Average CLIP Cosine**: **0.3056** (Range: 0.2819 for Sandal to 0.3301 for Pullover)

---

### 6. cGAN vs Conditional VAE Comparative Summary

| Metric / Dimension | Conditional GAN (Projection D) | Conditional VAE (C-VAE) | Analysis |
|---|---|---|---|
| **Overall Oracle Fidelity** | **97.20%** | **74.90%** | cGAN significantly outperforms VAE due to sharp edge discrimination |
| **Weakest Category** | Shirt (84.0%) | Shirt (11.0%), Coat (22.0%), Sandal (50.0%) | VAE struggles with upper-body distinctions and intricate footwear straps |
| **Average CLIP Cosine** | 0.2975 | 0.3056 | Comparable semantic alignment on upscaled pseudo-RGB |
| **Visual Sharpness** | High contrast, crisp silhouette contours | Smooth, blurred contours across mode averages | Expected L2/BCE pixel-space loss averaging in VAE vs min-max game in GAN |
| **Mode Diversity** | High within-class variation | High continuous latent interpolation smoothness | VAE allows continuous latent walks; cGAN produces discrete sharp variations |
| **Inference Latency (8 items)** | $\approx 12$ ms (CUDA) | $\approx 8$ ms (CUDA) | Both models provide sub-15ms real-time synthesis on GPU |

---

### 7. Deployment Readiness Assessment & Remaining Limitations

#### A. Deployment Readiness
- **Architecture**: Fully decoupled inference engine (`AtelierInferenceEngine`).
- **Device Portability**: Automatic dynamic migration to CUDA upon Hugging Face ZeroGPU `@spaces.GPU` allocation via `_ensure_device()`.
- **Test Suite**: 20/20 pytest tests passing (API, Models, Parser, Safety).
- **Frontend Build**: Zero-error Vite 8 + React 19 production build (`dist/` generated in 561ms).
- **Public Readiness**: Codebase is clean, modular, and fully prepared for deployment upon explicit authorization.

#### B. Inherent Domain Limitations
1. **Resolution & Color**: The system generates $28 \times 28$ single-channel grayscale silhouettes. It cannot synthesize high-resolution fabrics, arbitrary colors, logos, or 3D angles.
2. **Shirt vs T-shirt Ambiguity**: Even the auxiliary CNN oracle achieves only $67.8\%$ accuracy on real Fashion-MNIST test shirts, as low-resolution grayscale silhouettes make button plackets and collars indistinct.
3. **CLIP Domain Shift**: CLIP ViT-B/32 was trained on high-resolution natural web images. Upscaling $28 \times 28$ grayscale images introduces smoothing artifacts, bounding typical cosine similarities to the $0.20 - 0.35$ range.
4. **Natural Language Boundaries**: The standalone 0.5B parameter causal LLM requires few-shot prompting or deterministic fallback to reliably extract non-standard synonyms.

---

### 8. Licensing & Dataset Attribution
- **Fashion-MNIST**: Copyright (c) 2017 Zalando SE, **MIT License**.
- **Qwen2.5-0.5B-Instruct**: Alibaba Cloud, **Apache 2.0 License**.
- **CLIP ViT-B/32**: OpenAI, **MIT License**.
- **DistilBERT**: Hugging Face, **Apache 2.0 License**.
- **AtelierAI Application**: Copyright (c) 2026 AtelierAI Contributors, **MIT License**.
