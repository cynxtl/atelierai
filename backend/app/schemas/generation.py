"""Pydantic schemas for AtelierAI API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field


class GeneratedImageItem(BaseModel):
    index: int
    image_b64: str
    classified_as: str
    classifier_confidence: float
    category_match: bool


class ClipEvaluationSummary(BaseModel):
    target_class: str
    sample_count: int
    mean_target_similarity: float
    clip_top1_match_rate: float
    per_image_target_scores: list[float]
    limitations_note: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=500, description="Natural language prompt describing clothing item")
    model_type: str = Field(default="cgan", description="Generative model: 'cgan' or 'vae'")
    strategy: str = Field(default="zero-shot", description="Prompting strategy: 'zero-shot' or 'few-shot'")
    seed: int | None = Field(default=None, description="Optional seed for deterministic generation")
    evaluate_clip: bool = Field(default=True, description="Whether to compute CLIP multimodal alignment")


class GenerateResponse(BaseModel):
    success: bool
    prompt: str
    parsed_label: str | None = None
    display_label: str | None = None
    class_idx: int | None = None
    count: int = 1
    model_type: str = "cgan"
    strategy: str = "zero-shot"
    images: list[GeneratedImageItem] = []
    clip_evaluation: dict[str, Any] | None = None
    latency_seconds: float = 0.0
    device: str = "cpu"
    error: str | None = None
    warning: str | None = None
    unsupported_requested: str | None = None


class CompareRequest(BaseModel):
    prompt: str = Field(..., max_length=500, description="Natural language prompt for side-by-side comparison")
    seed: int | None = Field(default=42, description="Seed for deterministic comparison")


class CompareResponse(BaseModel):
    success: bool
    prompt: str
    parsed_label: str | None = None
    display_label: str | None = None
    count: int = 1
    cgan: dict[str, Any] | None = None
    vae: dict[str, Any] | None = None
    error: str | None = None
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    checkpoints_available: dict[str, bool]
