# Production Deployment Guide: Vercel + Hugging Face ZeroGPU

## 1. Architecture Overview

```text
                  PUBLIC USER
                      |
                      v
             VERCEL HOBBY
        React + Vite + TypeScript
                      |
                      | Gradio Client API calls (HTTPS)
                      v
          HUGGING FACE ZEROGPU
             Gradio SDK Space
                      |
            Shared inference services
            (ml/inference/service.py)
                      |
             +--------+--------+
             |                 |
        CPU inference     GPU inference
        when practical    when required (@spaces.GPU)
             |                 |
             +--------+--------+
                      |
                      v
          HUGGING FACE MODEL HUB
        Versioned model checkpoints
```

---

## 2. Free-Tier Verification & Resource Analysis

### A. Vercel Hobby
- **Eligibility**: 100% Free for personal and non-commercial projects.
- **Constraints**: 100 GB bandwidth / month, static site deployment.
- **Design Strategy**: AtelierAI builds as a pure Client-Side SPA (`dist/`). No heavyweight PyTorch models or CUDA binaries run inside Vercel serverless functions.
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### B. Hugging Face Spaces (ZeroGPU)
- **Eligibility**: Free tier ZeroGPU provides dynamic, temporary allocations on NVIDIA A100/L4 GPUs.
- **Requirements**:
  - `sdk: gradio` (Gradio 5+ or 6+)
  - Wrap GPU-intensive generation functions with `@spaces.GPU(duration=60)`.
  - Non-GPU tasks (such as serving benchmark JSON metadata or health checks) run on CPU without holding GPU allocation.
- **Token Security**: No private Hugging Face tokens are bundled in the frontend bundle. Anonymous API calls are supported on Public Spaces.

---

## 3. Deployment Procedure (Awaiting Authorization)

> **Important**: Deployment must only proceed after local validation and explicit authorization.

### Step 1: Push Model Weights to Hugging Face Hub
```bash
huggingface-cli login
huggingface-cli repo create atelier-fashion-models --type model
# Upload checkpoints from ml/checkpoints/
```

### Step 2: Deploy Backend to Hugging Face Space
1. Create a new Space: `atelier-ai-studio` (SDK: Gradio, Hardware: ZeroGPU).
2. Push `deployment/huggingface/` along with `ml/` to the Space repository.

### Step 3: Deploy Frontend to Vercel Hobby
1. Import GitHub repository into Vercel.
2. Set Root Directory to `frontend`.
3. Set environment variable:
   ```bash
   VITE_API_MODE=gradio
   VITE_GRADIO_SPACE_URL=your-username/atelier-ai-studio
   ```
4. Deploy!
