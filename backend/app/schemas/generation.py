"""Pydantic schemas for AtelierAI API requests and responses."""

from typing import List, Optional, Dict, Any
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
    per_image_target_scores: List[float]
    limitations_note: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=500, description="Natural language prompt describing clothing item")
    model_type: str = Field(default="cgan", description="Generative model: 'cgan' or 'vae'")
    strategy: str = Field(default="zero-shot", description="Prompting strategy: 'zero-shot' or 'few-shot'")
    seed: Optional[int] = Field(default=None, description="Optional seed for deterministic generation")
    evaluate_clip: bool = Field(default=True, description="Whether to compute CLIP multimodal alignment")


class GenerateResponse(BaseModel):
    success: bool
    prompt: str
    parsed_label: Optional[str] = None
    display_label: Optional[str] = None
    class_idx: Optional[int] = None
    count: int = 1
    model_type: str = "cgan"
    strategy: str = "zero-shot"
    images: List[GeneratedImageItem] = []
    clip_evaluation: Optional[Dict[str, Any]] = None
    latency_seconds: float = 0.0
    device: str = "cpu"
    error: Optional[str] = None
    warning: Optional[str] = None
    unsupported_requested: Optional[str] = None


class CompareRequest(BaseModel):
    prompt: str = Field(..., max_length=500, description="Natural language prompt for side-by-side comparison")
    seed: Optional[int] = Field(default=42, description="Seed for deterministic comparison")


class CompareResponse(BaseModel):
    success: bool
    prompt: str
    parsed_label: Optional[str] = None
    display_label: Optional[str] = None
    count: int = 1
    cgan: Optional[Dict[str, Any]] = None
    vae: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    checkpoints_available: Dict[str, bool]
