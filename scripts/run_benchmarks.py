"""CLI script to execute end-to-end evaluation and generate genuine benchmark reports."""

import argparse
import json
import logging
import time
from pathlib import Path
import torch

from ml.configs.config import (
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    FASHION_CLASSES,
)
from ml.models.cgan import ConditionalGenerator
from ml.models.vae import ConditionalVAE
from ml.models.classifier import FashionCNNClassifier
from ml.models.clip_evaluator import ClipEvaluator
from ml.models.llm_parser import QwenClothingParser
from ml.models.distilbert_classifier import DistilBertClothingClassifier
from ml.evaluation.benchmark import (
    evaluate_natural_language_models,
    evaluate_robustness_benchmark,
    evaluate_image_generation,
    evaluate_responsible_ai,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run complete AtelierAI benchmarks")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Image samples to evaluate per class")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-distilbert", action="store_true", help="Skip DistilBERT evaluation if not trained")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Starting AtelierAI Benchmark Suite on device: {device}")
    start_time = time.time()

    # 1. Load Auxiliary Image Classifier
    classifier_ckpt = CHECKPOINTS_DIR / "fashion_classifier.pt"
    if not classifier_ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint missing at {classifier_ckpt}. Please train it first.")
    classifier = FashionCNNClassifier().to(device)
    clf_data = torch.load(classifier_ckpt, map_location=device, weights_only=False)
    classifier.load_state_dict(clf_data["model_state_dict"])
    classifier.eval()
    logger.info("Loaded Fashion-MNIST auxiliary classifier.")

    # 2. Load cGAN Generator
    cgan_ckpt = CHECKPOINTS_DIR / "cgan_generator.pt"
    if not cgan_ckpt.exists():
        raise FileNotFoundError(f"cGAN checkpoint missing at {cgan_ckpt}. Please train it first.")
    cgan_generator = ConditionalGenerator().to(device)
    cgan_data = torch.load(cgan_ckpt, map_location=device, weights_only=False)
    cgan_generator.load_state_dict(cgan_data["model_state_dict"])
    cgan_generator.eval()
    logger.info("Loaded cGAN generator.")

    # 3. Load VAE
    vae_ckpt = CHECKPOINTS_DIR / "vae.pt"
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint missing at {vae_ckpt}. Please train it first.")
    vae_model = ConditionalVAE().to(device)
    vae_data = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae_model.load_state_dict(vae_data["model_state_dict"])
    vae_model.eval()
    logger.info("Loaded Conditional VAE.")

    # 4. Load CLIP Evaluator
    logger.info("Loading CLIP ViT-B/32 evaluator...")
    clip_evaluator = ClipEvaluator(device=str(device))
    clip_evaluator.load()

    # 5. Load Parser & DistilBERT
    logger.info("Initializing Qwen clothing parser...")
    qwen_parser = QwenClothingParser(device=str(device))
    qwen_parser.load()

    distilbert_clf = None
    distilbert_dir = CHECKPOINTS_DIR / "distilbert_fashion"
    if not args.skip_distilbert and distilbert_dir.exists():
        logger.info(f"Loading fine-tuned DistilBERT from {distilbert_dir}...")
        distilbert_clf = DistilBertClothingClassifier(model_dir_or_name=str(distilbert_dir), device=str(device))
        distilbert_clf.load()

    # --- Run Benchmarks ---

    # A. Natural Language Benchmark (10 Canonical Requests with Standalone vs Fallback vs Combined)
    nlp_results = evaluate_natural_language_models(qwen_parser, distilbert=distilbert_clf)

    # B. Robustness Benchmark (20 Diverse & Adversarial Requests)
    robustness_results = evaluate_robustness_benchmark(qwen_parser)

    # C. Layer V Responsible AI Benchmark
    responsible_ai_results = evaluate_responsible_ai(qwen_parser)

    # D. Image Generation & Fidelity Benchmark
    image_results = evaluate_image_generation(
        cgan_generator=cgan_generator,
        vae_model=vae_model,
        classifier=classifier,
        clip_evaluator=clip_evaluator,
        samples_per_class=args.samples_per_class,
        device=device,
    )

    elapsed = round(time.time() - start_time, 2)

    # Compile comprehensive report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_benchmark_time_seconds": elapsed,
        "device": str(device),
        "natural_language_benchmark": nlp_results,
        "robustness_benchmark": robustness_results,
        "responsible_ai_benchmark": responsible_ai_results,
        "image_generation_benchmark": image_results,
    }

    report_path = REPORTS_DIR / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Benchmark execution complete in {elapsed}s!")
    logger.info(f"Results saved to {report_path}")

    # Print summary highlights
    print("\n" + "=" * 65)
    print("           ATELIER-AI BENCHMARK RESULTS SUMMARY")
    print("=" * 65)
    print("Ten-Request Benchmark Category Accuracy:")
    print(f"  - Standalone Qwen Zero-Shot (No Fallback): {nlp_results['standalone_qwen_zero_shot']['category_accuracy']}%")
    print(f"  - Standalone Qwen Few-Shot  (No Fallback): {nlp_results['standalone_qwen_few_shot']['category_accuracy']}%")
    print(f"  - Standalone Rule Fallback:               {nlp_results['standalone_rule_fallback']['category_accuracy']}%")
    print(f"  - Combined Pipeline Zero-Shot:            {nlp_results['combined_pipeline_zero_shot']['category_accuracy']}%")
    print(f"  - Combined Pipeline Few-Shot:             {nlp_results['combined_pipeline_few_shot']['category_accuracy']}%")
    if nlp_results['distilbert_category_accuracy'] is not None:
        print(f"  - Fine-Tuned DistilBERT:                  {nlp_results['distilbert_category_accuracy']}%")
    print(f"\n20-Request Robustness Benchmark Pass Rate: {robustness_results['pass_rate_percentage']}%")
    print(f"Layer V Responsible AI Pass Rate:          {responsible_ai_results['pass_rate_percentage']}%")
    print(f"\nImage Generation Category Match Rate:")
    print(f"  - cGAN Overall:   {image_results['cgan']['overall_class_match_rate']}%")
    print(f"  - VAE Overall:    {image_results['vae']['overall_class_match_rate']}%")
    print(f"  - cGAN Avg CLIP:  {image_results['cgan']['average_clip_similarity']} (cosine)")
    print(f"  - VAE Avg CLIP:   {image_results['vae']['average_clip_similarity']} (cosine)")
    print(f"Weakest Classes (cGAN): {', '.join(image_results['cgan']['weakest_classes'])}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
