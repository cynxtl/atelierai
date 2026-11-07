"""Multimodal image-text alignment evaluation using CLIP ViT-B/32."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.configs.config import FASHION_CLASSES

logger = logging.getLogger(__name__)


CLIP_CLASS_DESCRIPTIONS = [
    "a grayscale photo of a t-shirt or top",
    "a grayscale photo of trousers or pants",
    "a grayscale photo of a pullover sweater",
    "a grayscale photo of a dress",
    "a grayscale photo of a coat or jacket",
    "a grayscale photo of a sandal shoe",
    "a grayscale photo of a collared button-down shirt",
    "a grayscale photo of an athletic sneaker shoe",
    "a grayscale photo of a handbag or tote bag",
    "a grayscale photo of an ankle boot",
]


class ClipEvaluator:
    """
    Evaluates generated Fashion-MNIST images against textual category prompts
    using openai/clip-vit-base-patch32.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load(self):
        """Lazily loads CLIP model and processor."""
        if self._is_loaded:
            return

        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model from {self.model_name}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info("CLIP model loaded successfully.")

    def preprocess_tensor_to_rgb_pil(self, tensor: torch.Tensor) -> list[Image.Image]:
        """
        Converts 1x28x28 tensor (values in [-1, 1] or [0, 1]) to 3-channel PIL Images.
        """
        # Ensure CPU
        t = tensor.detach().cpu()
        if t.ndim == 3:  # (C, H, W)
            t = t.unsqueeze(0)

        # Normalize to [0, 255] uint8
        # Check if tensor was [-1, 1]
        if t.min() < 0:
            t = (t + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        t = (t * 255.0).to(torch.uint8)

        images = []
        for i in range(t.size(0)):
            # Convert single channel to 3 channels RGB
            img_arr = t[i, 0].numpy()
            rgb_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
            pil_img = Image.fromarray(rgb_arr, mode="RGB")
            # Resize with bicubic interpolation to 224x224 for CLIP ViT-B/32
            pil_resized = pil_img.resize((224, 224), Image.BICUBIC)
            images.append(pil_resized)
        return images

    @torch.no_grad()
    def compute_similarity(
        self,
        images: torch.Tensor | list[Image.Image],
        text_prompts: str | list[str],
    ) -> list[dict[str, any]]:
        """
        Computes cosine similarity between images and text prompts.
        Returns per-image scores and ranking across text prompts.
        """
        if not self._is_loaded:
            self.load()

        if isinstance(images, torch.Tensor):
            pil_images = self.preprocess_tensor_to_rgb_pil(images)
        else:
            pil_images = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img.resize((224, 224), Image.BICUBIC))

        if isinstance(text_prompts, str):
            prompts = [text_prompts]
        else:
            prompts = text_prompts

        # Process inputs through CLIP processor
        inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        image_outputs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = getattr(image_outputs, "pooler_output", image_outputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_outputs[1] if len(image_outputs) > 1 else image_outputs[0]

        text_outputs = self.model.get_text_features(input_ids=inputs["input_ids"])
        text_features = getattr(text_outputs, "pooler_output", text_outputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_outputs[1] if len(text_outputs) > 1 else text_outputs[0]

        # Normalize feature vectors to unit hypersphere
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Cosine similarity matrix: (num_images, num_prompts)
        similarity_matrix = torch.matmul(image_features, text_features.t()).cpu().numpy()

        results = []
        for i in range(len(pil_images)):
            scores = {prompts[j]: float(similarity_matrix[i, j]) for j in range(len(prompts))}
            best_prompt = max(scores.items(), key=lambda item: item[1])
            results.append({
                "image_index": i,
                "scores": scores,
                "best_matching_prompt": best_prompt[0],
                "top_similarity_score": round(best_prompt[1], 4),
            })

        return results

    @torch.no_grad()
    def evaluate_class_alignment(
        self,
        images: torch.Tensor,
        target_class_idx: int,
    ) -> dict[str, any]:
        """
        Evaluates a batch of generated images against all 10 Fashion-MNIST category prompts.
        Calculates target class similarity, rank, and zero-shot accuracy under CLIP.
        """
        prompts = CLIP_CLASS_DESCRIPTIONS
        results = self.compute_similarity(images, prompts)

        target_prompt = prompts[target_class_idx]
        target_scores = [res["scores"][target_prompt] for res in results]
        mean_target_score = float(np.mean(target_scores))

        # Check how often target prompt had the highest similarity
        top1_matches = sum(1 for res in results if res["best_matching_prompt"] == target_prompt)
        top1_rate = top1_matches / len(results) if results else 0.0

        return {
            "target_class": FASHION_CLASSES[target_class_idx],
            "sample_count": len(results),
            "mean_target_similarity": round(mean_target_score, 4),
            "clip_top1_match_rate": round(top1_rate, 4),
            "per_image_target_scores": [round(s, 4) for s in target_scores],
            "limitations_note": (
                "CLIP Cosine Metric Note: CLIP was trained on 400M web RGB photographs. "
                "28x28 grayscale silhouettes encounter significant domain and resolution shifts. "
                "Similarity scores reflect raw cosine proximity in embedding space (range [-1.0, 1.0]), "
                "and must NOT be interpreted as calibrated class probabilities or generation correctness."
            ),
        }
