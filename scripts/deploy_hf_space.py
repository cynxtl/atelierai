"""
Deploys AtelierAI Gradio API adapter to Hugging Face Spaces (ZeroGPU).
Target Space: cynstl/atelierai-api (Public, Gradio SDK)
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPACE_ID = "cynstl/atelierai-api"
STAGING_DIR = PROJECT_ROOT / "deployment" / "space_staging"

# Load token from ~/.cache/huggingface/token
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text().strip() if token_path.exists() else None

api = HfApi(token=token)

print(f"[1/4] Creating/verifying Hugging Face Space: {SPACE_ID}...")
space_url = create_repo(
    repo_id=SPACE_ID,
    repo_type="space",
    space_sdk="gradio",
    space_hardware="zero-a10g",
    private=False,
    exist_ok=True,
    token=token,
)
print(f"Space URL: {space_url}")

# [2/4] Prepare staging folder
print("\n[2/4] Assembling Space staging package...")
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Copy app.py
shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "app.py", STAGING_DIR / "app.py")

# Space requirements.txt
space_reqs = """torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
accelerate>=0.28.0
pydantic>=2.6.0
pillow>=10.2.0
numpy>=1.26.0,<3.0.0
huggingface_hub>=0.23.0
spaces>=0.30.0
"""
(STAGING_DIR / "requirements.txt").write_text(space_reqs, encoding="utf-8")

# Space README.md with YAML metadata
space_readme = """---
title: AtelierAI API
emoji: 👗
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# AtelierAI — Text-to-Fashion Synthesis Studio (ZeroGPU Adapter)

This Space exposes the official Gradio API for **AtelierAI**, an independent generative fashion synthesis application built with PyTorch, Qwen2.5, and trained Conditional GANs on Fashion-MNIST.

- **Frontend**: Deployed on Vercel
- **Trained Models**: [cynstl/atelierai-models](https://huggingface.co/cynstl/atelierai-models)
- **License**: MIT
"""
(STAGING_DIR / "README.md").write_text(space_readme, encoding="utf-8")

# Copy ml directory (excluding checkpoints and data)
dest_ml = STAGING_DIR / "ml"
shutil.copytree(
    PROJECT_ROOT / "ml",
    dest_ml,
    ignore=shutil.ignore_patterns("checkpoints", "data", "__pycache__", "*.pt", "*.safetensors")
)

# Copy reports/benchmark_results.json
dest_reports = STAGING_DIR / "reports"
dest_reports.mkdir(parents=True, exist_ok=True)
if (PROJECT_ROOT / "reports" / "benchmark_results.json").exists():
    shutil.copy2(PROJECT_ROOT / "reports" / "benchmark_results.json", dest_reports / "benchmark_results.json")

print(f"Staged {sum(1 for _ in STAGING_DIR.rglob('*') if _.is_file())} files in {STAGING_DIR}")

# [3/4] Upload folder to Hugging Face Space
print(f"\n[3/4] Uploading Space files to {SPACE_ID}...")
api.upload_folder(
    folder_path=str(STAGING_DIR),
    repo_id=SPACE_ID,
    repo_type="space",
    token=token,
)

print(f"\n[4/4] Space upload complete! Checking Space runtime status...")
runtime = api.get_space_runtime(repo_id=SPACE_ID, token=token)
print(f"Space Stage: {runtime.stage}")
print(f"Space Hardware: {runtime.hardware}")

# Clean up staging directory
shutil.rmtree(STAGING_DIR, ignore_errors=True)
print(f"\nHugging Face Space is live at: https://huggingface.co/spaces/{SPACE_ID}")
