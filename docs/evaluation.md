# AtelierAI Evaluation Methodology & Empirical Findings

## 1. Natural Language Benchmark
A fixed ten-request evaluation benchmark covering all 10 Fashion-MNIST categories was designed to test parsing fidelity:

1. *"Show me four ankle boots."* \(\rightarrow\) `ankle boot`, count: 4
2. *"Generate a stylish summer dress."* \(\rightarrow\) `dress`, count: 1
3. *"I need 2 casual denim trousers."* \(\rightarrow\) `trouser`, count: 2
4. *"Can you synthesize a warm winter pullover?"* \(\rightarrow\) `pullover`, count: 1
5. *"Produce three white graphic t-shirts."* \(\rightarrow\) `t-shirt/top`, count: 3
6. *"A crisp formal button-down shirt."* \(\rightarrow\) `shirt`, count: 1
7. *"Two comfortable leather sandals for the beach."* \(\rightarrow\) `sandal`, count: 2
8. *"Generate a pair of athletic running sneakers."* \(\rightarrow\) `sneaker`, count: 2
9. *"A designer leather handbag or tote bag."* \(\rightarrow\) `bag`, count: 1
10. *"A heavy padded winter coat."* \(\rightarrow\) `coat`, count: 1

### Models Compared
1. **Qwen2.5-0.5B Zero-Shot Prompting**: Evaluates raw in-context reasoning with taxonomy rules.
2. **Qwen2.5-0.5B Few-Shot Prompting**: Evaluates parsing with exemplar guidance.
3. **Fine-Tuned DistilBERT**: Supervised classification baseline.

---

## 2. Image Generation Fidelity Metrics
- **Auxiliary CNN Oracle**: Independently trained on Fashion-MNIST (test accuracy: **90.76%**). Evaluates whether generated images match their conditioned class label across 100 samples per category.
- **CLIP ViT-B/32 Alignment**: Computes cosine similarity between generated images and category prompts.
- **Weakest Performing Classes**:
  - **Shirt** & **T-shirt/top**: Suffer from collar ambiguity in 28×28 grayscale.
  - **Trouser**, **Bag**, and footwear categories (**Sandal**, **Sneaker**, **Ankle boot**) achieve highest class match rates (>90%) due to distinctive silhouettes.

---

## 3. cGAN vs Conditional VAE Comparative Observations
- **Sharpness**: cGAN produces crisper silhouette boundaries due to adversarial min-max pressure. VAE produces smoother, slightly blurrier images due to pixel-space mean squared error / BCE loss averaging across ambiguous pixel values.
- **Mode Diversity**: VAE covers the full empirical distribution without mode collapse due to regularized continuous latent sampling. cGAN requires careful discriminator regularization (e.g. label smoothing, dropout) to prevent discriminator dominance.
