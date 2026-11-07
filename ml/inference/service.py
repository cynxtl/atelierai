"""Shared ML Inference Service for AtelierAI.
Decoupled from API frameworks so it can be imported and executed by both
FastAPI (local development) and Gradio SDK (Hugging Face ZeroGPU).
"""

import base64
import io
import json
import logging
import os
import pathlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Cross-platform compatibility for Windows checkpoints loaded on Linux (e.g. Hugging Face Spaces)
if os.name != "nt":
    try:
        pathlib.WindowsPath = pathlib.PosixPath
    except Exception:
        pass

from ml.configs.config import (
    CHECKPOINTS_DIR,
    FASHION_CLASSES,
    REPORTS_DIR,
)
from ml.models.cgan import ConditionalGenerator
from ml.models.classifier import FashionCNNClassifier
from ml.models.clip_evaluator import ClipEvaluator
from ml.models.distilbert_classifier import DistilBertClothingClassifier
from ml.models.llm_parser import ParseResult, QwenClothingParser
from ml.models.vae import ConditionalVAE

logger = logging.getLogger(__name__)


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """Converts a 1x28x28 PyTorch image tensor to a base64-encoded PNG string."""
    t = tensor.detach().cpu()
    if t.ndim == 3:
        t = t.squeeze(0)

    # Check if tensor is in [-1, 1] (from cGAN) or [0, 1] (from VAE)
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0.0, 1.0)
    img_arr = (t.numpy() * 255.0).astype(np.uint8)

    # Upscale 28x28 to 112x112 for crisp display in browser
    pil_img = Image.fromarray(img_arr, mode="L")
    pil_large = pil_img.resize((112, 112), Image.NEAREST)

    buffer = io.BytesIO()
    pil_large.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


class AtelierInferenceEngine:
    """
    Centralized Inference Engine managing model weights, device placement,
    memory efficiency, and text-to-image synthesis pipeline.
    Supports both local checkpoints and automatic Hugging Face Hub resolution.
    """
    def __init__(
        self,
        checkpoints_dir: Path | None = None,
        hf_model_repo: str | None = None,
        device: str | None = None,
    ):
        self.checkpoints_dir = Path(checkpoints_dir or CHECKPOINTS_DIR)
        self.hf_model_repo = hf_model_repo or os.environ.get("HF_MODEL_REPO", "cynstl/atelierai-models")
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialized AtelierInferenceEngine on device: {self.device}")

        # Models (lazily loaded)
        self.parser = QwenClothingParser(device=str(self.device))
        self.cgan_generator: ConditionalGenerator | None = None
        self.vae_model: ConditionalVAE | None = None
        self.classifier: FashionCNNClassifier | None = None
        self.clip_evaluator: ClipEvaluator | None = None
        self.distilbert: DistilBertClothingClassifier | None = None

    def _get_checkpoint_path(self, filename: str) -> Path:
        """Returns local checkpoint path if exists, otherwise downloads from Hugging Face Hub."""
        local_path = self.checkpoints_dir / filename
        if local_path.exists():
            return local_path

        if self.hf_model_repo:
            try:
                from huggingface_hub import hf_hub_download
                logger.info(f"Checkpoint not found locally at {local_path}. Downloading {filename} from {self.hf_model_repo}...")
                downloaded = hf_hub_download(repo_id=self.hf_model_repo, filename=filename)
                return Path(downloaded)
            except Exception as e:
                logger.error(f"Failed to download {filename} from {self.hf_model_repo}: {e}")

        raise FileNotFoundError(
            f"Checkpoint '{filename}' not found locally at {local_path} and remote download failed."
        )

    def load_cgan(self):
        """Loads trained Conditional GAN generator."""
        if self.cgan_generator is not None:
            return

        ckpt_path = self._get_checkpoint_path("cgan_generator.pt")
        logger.info(f"Loading cGAN from {ckpt_path}...")
        self.cgan_generator = ConditionalGenerator().to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cgan_generator.load_state_dict(checkpoint["model_state_dict"])
        self.cgan_generator.eval()

    def load_vae(self):
        """Loads trained Conditional VAE."""
        if self.vae_model is not None:
            return

        ckpt_path = self._get_checkpoint_path("vae.pt")
        logger.info(f"Loading VAE from {ckpt_path}...")
        self.vae_model = ConditionalVAE().to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.vae_model.load_state_dict(checkpoint["model_state_dict"])
        self.vae_model.eval()

    def load_classifier(self):
        """Loads auxiliary Fashion-MNIST CNN classifier for validation."""
        if self.classifier is not None:
            return

        ckpt_path = self._get_checkpoint_path("fashion_classifier.pt")
        logger.info(f"Loading classifier from {ckpt_path}...")
        self.classifier = FashionCNNClassifier().to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()

    def load_clip(self):
        """Loads CLIP ViT-B/32 evaluator."""
        if self.clip_evaluator is not None and self.clip_evaluator._is_loaded:
            return

        self.clip_evaluator = ClipEvaluator(device=str(self.device))
        self.clip_evaluator.load()

    def load_distilbert(self):
        """Loads fine-tuned DistilBERT classifier."""
        if self.distilbert is not None and self.distilbert._is_loaded:
            return

        model_dir = self.checkpoints_dir / "distilbert_fashion"
        if not model_dir.exists() and self.hf_model_repo:
            try:
                from huggingface_hub import snapshot_download
                logger.info(f"Downloading distilbert_fashion from {self.hf_model_repo}...")
                dl_dir = snapshot_download(repo_id=self.hf_model_repo, allow_patterns=["distilbert_fashion/*"])
                model_dir = Path(dl_dir) / "distilbert_fashion"
            except Exception as e:
                logger.warning(f"Could not download distilbert from {self.hf_model_repo}: {e}")

        if model_dir.exists():
            self.distilbert = DistilBertClothingClassifier(model_dir_or_name=str(model_dir), device=str(self.device))
            self.distilbert.load()
        else:
            logger.warning("DistilBERT checkpoint not found. Text classification comparison will be disabled.")

    def _ensure_device(self):
        """Dynamically ensures model weights are placed on CUDA if GPU becomes available (e.g. ZeroGPU allocation)."""
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device != current_device:
            logger.info(f"Device changed from {self.device} to {current_device}. Migrating models...")
            self.device = current_device
            if self.cgan_generator is not None:
                self.cgan_generator.to(self.device)
            if self.vae_model is not None:
                self.vae_model.to(self.device)
            if self.classifier is not None:
                self.classifier.to(self.device)
            if self.clip_evaluator is not None and self.clip_evaluator.model is not None:
                self.clip_evaluator.model.to(self.device)
                self.clip_evaluator.device = str(self.device)
            if self.distilbert is not None and self.distilbert.model is not None:
                self.distilbert.model.to(self.device)
                self.distilbert.device = str(self.device)

    def parse_prompt(self, prompt: str, strategy: str = "zero-shot") -> ParseResult:
        """Parses natural language prompt into structured schema."""
        return self.parser.parse(prompt, strategy=strategy)

    def generate(
        self,
        prompt: str,
        model_type: str = "cgan",
        strategy: str = "zero-shot",
        evaluate_clip: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Executes complete text-to-clothing generation pipeline:
        1. NLP Parsing (Qwen2.5-0.5B / validated schema)
        2. Image Synthesis (cGAN or VAE)
        3. Multimodal CLIP evaluation
        4. Independent Classifier verification
        """
        self._ensure_device()
        start_time = time.time()

        # Step 1: Parse Prompt
        parse_res = self.parse_prompt(prompt, strategy=strategy)
        if not parse_res.is_valid:
            return {
                "success": False,
                "error": parse_res.error_message,
                "unsupported_requested": parse_res.unsupported_requested,
                "strategy": strategy,
                "prompt": prompt,
            }

        target_idx = parse_res.class_idx
        count = parse_res.count
        target_class_name = parse_res.display_label

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Step 2: Generate with Selected Model
        labels = torch.full((count,), target_idx, dtype=torch.long, device=self.device)

        if model_type.lower() == "vae":
            self.load_vae()
            with torch.no_grad():
                gen_tensors = self.vae_model.sample(labels, device=self.device)
                # VAE tensors are in [0, 1]
        else:
            self.load_cgan()
            with torch.no_grad():
                z = torch.randn(count, self.cgan_generator.latent_dim, device=self.device)
                gen_tensors = self.cgan_generator(z, labels)
                # cGAN tensors are in [-1, 1]

        # Step 3: Run Auxiliary Classifier Verification
        self.load_classifier()
        with torch.no_grad():
            if model_type.lower() == "vae":
                norm_imgs = (gen_tensors - 0.2860) / 0.3530
            else:
                norm_imgs = ((gen_tensors + 1.0) / 2.0 - 0.2860) / 0.3530
            preds, confs = self.classifier.predict_classes(norm_imgs)

        # Step 4: Run CLIP Evaluation (if requested)
        clip_results = None
        if evaluate_clip:
            try:
                self.load_clip()
                clip_summary = self.clip_evaluator.evaluate_class_alignment(gen_tensors, target_idx)
                clip_results = clip_summary
            except Exception as e:
                logger.warning(f"CLIP evaluation failed: {e}")
                clip_results = {"error": str(e)}

        # Encode generated images to Base64
        image_data = []
        for i in range(count):
            b64_str = tensor_to_base64_png(gen_tensors[i])
            pred_idx = preds[i].item()
            pred_class = FASHION_CLASSES[pred_idx]
            conf = round(confs[i].item(), 4)
            image_data.append({
                "index": i,
                "image_b64": b64_str,
                "classified_as": pred_class,
                "classifier_confidence": conf,
                "category_match": (pred_idx == target_idx),
            })

        elapsed = round(time.time() - start_time, 3)

        return {
            "success": True,
            "prompt": prompt,
            "parsed_label": parse_res.class_label,
            "display_label": target_class_name,
            "class_idx": target_idx,
            "count": count,
            "model_type": model_type.lower(),
            "strategy": strategy,
            "images": image_data,
            "clip_evaluation": clip_results,
            "latency_seconds": elapsed,
            "device": str(self.device),
            "warning": parse_res.warning,
        }

    def compare_cgan_and_vae(self, prompt: str, seed: int | None = 42) -> dict[str, Any]:
        """Runs both cGAN and VAE for the exact same prompt to allow side-by-side comparison."""
        parse_res = self.parse_prompt(prompt)
        if not parse_res.is_valid:
            return {
                "success": False,
                "error": parse_res.error_message,
                "prompt": prompt,
            }

        cgan_res = self.generate(prompt, model_type="cgan", seed=seed)
        vae_res = self.generate(prompt, model_type="vae", seed=seed)

        return {
            "success": True,
            "prompt": prompt,
            "parsed_label": parse_res.class_label,
            "display_label": parse_res.display_label,
            "count": parse_res.count,
            "cgan": cgan_res,
            "vae": vae_res,
            "warning": parse_res.warning,
        }

    def get_benchmark_report(self) -> dict[str, Any]:
        """Loads persisted benchmark metrics or indicates if not yet run."""
        for path in [REPORTS_DIR / "benchmark_results.json", Path("reports/benchmark_results.json"), Path("benchmark_results.json")]:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        return {
            "status": "not_yet_generated",
            "message": "Benchmark has not been run yet. Execute scripts/run_benchmarks.py to compute.",
        }

    def get_insights(self) -> dict[str, Any]:
        """Returns architecture and syllabus layer metadata."""
        return {
            "classes": FASHION_CLASSES,
            "cgan": {
                "generator_params": "1.1M",
                "discriminator": "Projection Discriminator",
                "latent_dim": 100,
                "loss": "BCE with Mismatched Negative Pairs",
                "fidelity": "97.2% overall class match",
            },
            "vae": {
                "latent_dim": 32,
                "loss": "BCE Reconstruction + KL Divergence (ELBO)",
                "fidelity": "74.9% overall class match",
            },
            "nlp": {
                "causal_llm": "Qwen/Qwen2.5-0.5B-Instruct",
                "discriminative_classifier": "Fine-Tuned DistilBERT",
            },
            "multimodal": {
                "alignment": "openai/clip-vit-base-patch32",
                "metric": "Uncalibrated Cosine Distance [-1.0, 1.0]",
            },
        }
