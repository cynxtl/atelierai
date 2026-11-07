"""Hugging Face ZeroGPU Gradio API Deployment Adapter for AtelierAI.
Reuses the shared AtelierInferenceEngine without code duplication.
"""

import os
import pathlib
import sys
from pathlib import Path

# Cross-platform compatibility for Windows checkpoints loaded on Linux (e.g. Hugging Face Spaces)
if os.name != "nt":
    try:
        pathlib.WindowsPath = pathlib.PosixPath
    except Exception:
        pass

import gradio as gr
import torch

# Add project root to sys.path so ml.* imports work seamlessly on Hugging Face Spaces
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "ml").exists():
    ROOT_DIR = CURRENT_DIR
else:
    ROOT_DIR = CURRENT_DIR.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.inference.service import AtelierInferenceEngine

# Conditional Spaces ZeroGPU decorator import
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# Initialize inference engine with automatic Hugging Face Model Hub resolution
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "cynstl/atelierai-models")
CHECKPOINTS_DIR = ROOT_DIR / "ml" / "checkpoints"
engine = AtelierInferenceEngine(
    checkpoints_dir=CHECKPOINTS_DIR if CHECKPOINTS_DIR.exists() else None,
    hf_model_repo=HF_MODEL_REPO,
)


def gpu_decorator(fn):
    """Wraps functions with @spaces.GPU if running on Hugging Face Spaces with ZeroGPU."""
    if has_spaces:
        return spaces.GPU(duration=30)(fn)
    return fn


@gpu_decorator
def api_generate(prompt: str, model_type: str = "cgan", strategy: str = "zero-shot", seed: int = 42):
    """
    Gradio API endpoint for generating Fashion-MNIST images from text prompt.
    Returns structured JSON with base64 images and evaluation metrics.
    """
    try:
        res = engine.generate(
            prompt=prompt,
            model_type=model_type,
            strategy=strategy,
            seed=int(seed) if seed else 42,
            evaluate_clip=True,
        )
        return res
    except Exception as e:
        return {"success": False, "error": str(e)}


@gpu_decorator
def api_compare(prompt: str, seed: int = 42):
    """Gradio API endpoint for side-by-side cGAN vs VAE comparison."""
    try:
        res = engine.compare_cgan_and_vae(prompt=prompt, seed=int(seed) if seed else 42)
        return res
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_benchmark():
    """Returns precomputed benchmark metrics without requiring GPU allocation."""
    return engine.get_benchmark_report()


def api_health():
    """Health status and metadata endpoint."""
    return {
        "status": "healthy",
        "app_name": "AtelierAI",
        "version": "1.0.0",
        "device": "cuda" if has_spaces else str(engine.device),
        "cuda_available": True if has_spaces else torch.cuda.is_available(),
        "gpu_name": "ZeroGPU (NVIDIA A100/H100/T4 on demand)" if has_spaces else "Local CUDA",
        "checkpoints_available": {
            "cgan_generator": True,
            "vae": True,
            "classifier": True,
            "distilbert": True,
        },
    }


def api_insights():
    """Model architecture and syllabus layer insights."""
    return engine.get_insights()


# Gradio UI Interface definition (acts both as interactive demo and callable API for React)
with gr.Blocks(title="AtelierAI: Text-to-Fashion Synthesis Studio") as demo:
    gr.Markdown("# AtelierAI — Text-to-Conditional-Fashion Synthesis Studio")
    gr.Markdown(
        "Interprets natural-language clothing descriptions using Qwen2.5-0.5B-Instruct, "
        "generates 28x28 grayscale garments using a trained Conditional GAN or VAE, "
        "and evaluates image-text alignment using CLIP ViT-B/32."
    )

    with gr.Tab("Studio Synthesis"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Clothing Prompt",
                    placeholder="e.g. Show me four ankle boots.",
                    value="Show me four ankle boots.",
                )
                model_selector = gr.Radio(
                    choices=["cgan", "vae"],
                    value="cgan",
                    label="Generative Architecture",
                )
                strategy_selector = gr.Radio(
                    choices=["zero-shot", "few-shot"],
                    value="zero-shot",
                    label="Prompt Strategy",
                )
                seed_input = gr.Number(value=42, label="Random Seed")
                generate_btn = gr.Button("Synthesize Clothing", variant="primary")

            with gr.Column():
                output_json = gr.JSON(label="Structured Response & Evaluation")

        generate_btn.click(
            fn=api_generate,
            inputs=[prompt_input, model_selector, strategy_selector, seed_input],
            outputs=[output_json],
            api_name="generate",
        )

    with gr.Tab("cGAN vs VAE Comparison"):
        with gr.Row():
            compare_prompt = gr.Textbox(
                label="Comparison Prompt",
                value="Two warm winter pullovers.",
            )
            compare_btn = gr.Button("Compare Models", variant="primary")
        compare_json = gr.JSON(label="Side-by-Side Synthesis Output")

        compare_btn.click(
            fn=api_compare,
            inputs=[compare_prompt],
            outputs=[compare_json],
            api_name="compare",
        )

    with gr.Tab("Benchmark & Metrics"):
        benchmark_btn = gr.Button("Fetch Benchmark Report")
        benchmark_json = gr.JSON(label="Empirical Benchmark Metrics")
        benchmark_btn.click(fn=api_benchmark, outputs=[benchmark_json], api_name="benchmark")

    # API-only endpoints for external React frontend consumption
    with gr.Row(visible=False):
        health_btn = gr.Button("Health Check", visible=False)
        health_out = gr.JSON(visible=False)
        health_btn.click(fn=api_health, outputs=[health_out], api_name="health")

        insights_btn = gr.Button("Model Insights", visible=False)
        insights_out = gr.JSON(visible=False)
        insights_btn.click(fn=api_insights, outputs=[insights_out], api_name="insights")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
