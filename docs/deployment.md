# Production Deployment Guide: Vercel + Hugging Face ZeroGPU

## 1. Live Deployment Architecture

```text
                  PUBLIC WEB BROWSER
                          |
                          v
                    VERCEL HOBBY
              React 19 + TypeScript + Vite 8
              (github.com/cynxtl/atelierai)
                          |
                          | @gradio/client API calls (HTTPS)
                          v
             HUGGING FACE ZEROGPU SPACE
               cynstl/atelierai-api (Public)
               Gradio SDK + PyTorch 2.2+
                          |
                AtelierInferenceEngine
               (ml/inference/service.py)
                          |
                 +--------+--------+
                 |                 |
            CPU tasks         GPU tasks
          (Health, Insights,  (@spaces.GPU(duration=30):
           Benchmarks)         cGAN, VAE, CLIP, Qwen)
                 |                 |
                 +--------+--------+
                          |
                          v
               HUGGING FACE MODEL HUB
             cynstl/atelierai-models (Public)
             Versioned PyTorch Checkpoints
```

---

## 2. Live Deployment Endpoints

| Component | Provider & Tier | Public URL | Status |
|---|---|---|:---:|
| **Source Code** | GitHub (Public) | [https://github.com/cynxtl/atelierai](https://github.com/cynxtl/atelierai) | Verified |
| **Model Repository** | Hugging Face Model Hub (Free) | [https://huggingface.co/cynstl/atelierai-models](https://huggingface.co/cynstl/atelierai-models) | Verified (13 assets) |
| **Inference API** | Hugging Face Spaces ZeroGPU (Free) | [https://huggingface.co/spaces/cynstl/atelierai-api](https://huggingface.co/spaces/cynstl/atelierai-api) | Verified (`zero-a10g`) |
| **Interactive Studio** | Vercel Hobby (Free) | [Deploy to Vercel](https://vercel.com/new/clone?repository-url=https://github.com/cynxtl/atelierai&root-directory=frontend&env=VITE_API_MODE,VITE_GRADIO_SPACE_URL) | Ready for Import |

---

## 3. Cost & Zero-Budget Verification

- **Total Direct Infrastructure Cost**: **$0.00 / ₹0.00**
- **GitHub**: Free tier public repository with automated CI workflow. Checkpoints and heavy binaries strictly excluded via `.gitignore`.
- **Hugging Face Model Hub**: Free public model hosting storing `cgan_generator.pt` (4.2 MB), `vae.pt` (1.7 MB), `fashion_classifier.pt` (1.6 MB), and `distilbert_fashion/` (260 MB).
- **Hugging Face Spaces ZeroGPU**: Free `zero-a10g` hardware. GPU is dynamically allocated on demand during inference and released immediately after synthesis, incurring zero idle hardware costs.
- **Vercel Hobby**: Free client-side static application hosting (100 GB/month bandwidth limit).

---

## 4. Cold Starts & Runtime Characteristics

1. **Space Sleep & Wake Cycle**:
   - Hugging Face Spaces sleep after periods of inactivity.
   - When a user submits the first prompt after a cold state, the Space wakes up automatically (takes ~15–30 seconds for initial container allocation).
   - The frontend's `checkHealth` handler gracefully handles `cold_starting` status, maintaining an interactive UI with status badges.
2. **Dynamic Checkpoint Resolution**:
   - `AtelierInferenceEngine` automatically resolves model checkpoints from `cynstl/atelierai-models` via `hf_hub_download` and caches them in the Space's local filesystem (`~/.cache/huggingface/hub`).
3. **ZeroGPU Decorator (`@spaces.GPU`)**:
   - API endpoints `/generate` and `/compare` are wrapped with `@spaces.GPU(duration=30)`.
   - Lightweight endpoints (`/health`, `/benchmark`, `/insights`) execute on CPU without consuming GPU quotas.
4. **Zero-Token Frontend**:
   - The frontend connects to the public Hugging Face Space anonymously using `@gradio/client`. No private API keys or tokens are embedded in client bundles.

---

## 5. Vercel Frontend Deployment Steps

Deploying the frontend takes less than 60 seconds using the Vercel Dashboard:

### Option A: One-Click Vercel Deploy Link
Click the preconfigured deployment link:
[**Deploy AtelierAI on Vercel**](https://vercel.com/new/clone?repository-url=https://github.com/cynxtl/atelierai&root-directory=frontend&env=VITE_API_MODE,VITE_GRADIO_SPACE_URL)

### Option B: Manual Import in Vercel Dashboard
1. Go to [https://vercel.com/new](https://vercel.com/new) and log in.
2. Select **Import Git Repository** and search for `cynxtl/atelierai`.
3. In the project configuration:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend` (Click *Edit* and select `frontend`)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Expand **Environment Variables** and add:
   - `VITE_API_MODE` = `gradio`
   - `VITE_GRADIO_SPACE_URL` = `cynstl/atelierai-api`
5. Click **Deploy**.
