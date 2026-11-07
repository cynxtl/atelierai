export interface GeneratedImageItem {
  index: number;
  image_b64: string;
  classified_as: string;
  classifier_confidence: number;
  category_match: boolean;
}

export interface ClipEvaluationSummary {
  target_class: string;
  sample_count: number;
  mean_target_similarity: number;
  clip_top1_match_rate: number;
  per_image_target_scores: number[];
  limitations_note: string;
}

export interface GenerateResponse {
  success: boolean;
  prompt: string;
  parsed_label?: string;
  display_label?: string;
  class_idx?: number;
  count: number;
  model_type: 'cgan' | 'vae';
  strategy: 'zero-shot' | 'few-shot';
  images: GeneratedImageItem[];
  clip_evaluation?: ClipEvaluationSummary;
  latency_seconds: number;
  device: string;
  error?: string;
  warning?: string;
  unsupported_requested?: string;
}

export interface CompareResponse {
  success: boolean;
  prompt: string;
  parsed_label?: string;
  display_label?: string;
  count: number;
  cgan?: GenerateResponse;
  vae?: GenerateResponse;
  error?: string;
  warning?: string;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  device: string;
  cuda_available: boolean;
  gpu_name?: string;
  checkpoints_available: Record<string, boolean>;
}

export interface ModelMetrics {
  overall_class_match_rate: number;
  average_clip_similarity: number;
  per_class_match_rate: Record<string, number>;
  per_class_clip_similarity: Record<string, number>;
  weakest_classes: string[];
}

export interface BenchmarkReport {
  timestamp?: string;
  total_benchmark_time_seconds?: number;
  device?: string;
  natural_language_benchmark?: any;
  robustness_benchmark?: any;
  responsible_ai_benchmark?: {
    total_tests: number;
    tests_passed: number;
    pass_rate_percentage: number;
    details: Array<{
      test_type: string;
      prompt: string;
      expected_behavior: string;
      test_passed: boolean;
      reason: string;
    }>;
  };
  image_generation_benchmark?: {
    samples_per_class: number;
    cgan: ModelMetrics;
    vae: ModelMetrics;
    cgan_vs_vae_observations: string[];
  };
}
