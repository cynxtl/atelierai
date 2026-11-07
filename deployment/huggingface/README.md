---
title: AtelierAI Studio
emoji: 👗
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---

# AtelierAI — Text-to-Fashion Synthesis Studio (ZeroGPU Adapter)

This repository provides the Hugging Face Spaces ZeroGPU inference adapter for **AtelierAI**.

## Architecture
- **NLP Parsing**: Qwen2.5-0.5B-Instruct
- **Synthesis Engine**: Conditional GAN (cGAN) and Conditional VAE (C-VAE) trained on Fashion-MNIST
- **Multimodal Evaluation**: CLIP ViT-B/32
- **Objective Oracle**: Auxiliary CNN Classifier (90.76% accuracy)

## Usage
The API endpoints can be accessed directly or via `@gradio/client` in client-side web applications:
- `/generate`: Synthesize conditioned clothing images
- `/compare`: Side-by-side cGAN vs VAE comparison
- `/benchmark`: Retrieve empirical accuracy and safety metrics
