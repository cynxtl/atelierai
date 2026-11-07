# Model Card: CLIP ViT-B/32 Multimodal Evaluator

## Model Overview
- **Model Name**: `openai/clip-vit-base-patch32`
- **Developer**: OpenAI
- **Architecture**: Vision Transformer (ViT-B/32) Image Encoder + Transformer Text Encoder
- **Embedding Dimension**: 512

## AtelierAI Preprocessing Pipeline
1. Grayscale tensor (\(1 \times 28 \times 28\)) duplicated across 3 channels to form pseudo-RGB.
2. Clamped and scaled to uint8 range \([0, 255]\).
3. Resized to \(224 \times 224\) using bicubic interpolation.
4. Normalized with CLIP Image Processor.

## Cosine Similarity Formulation
Given normalized image feature \(u = \frac{f(x)}{\|f(x)\|_2}\) and normalized text feature \(v = \frac{g(t)}{\|g(t)\|_2}\):
\[
\text{Similarity}(x, t) = u \cdot v = \cos(\theta)
\]
Scores reflect relative alignment between the synthesized garment and category prompt.
