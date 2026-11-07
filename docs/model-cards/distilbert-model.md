# Model Card: Fine-Tuned DistilBERT Clothing Classifier

## Model Overview
- **Base Architecture**: `distilbert/distilbert-base-uncased`
- **Developer**: Hugging Face
- **Parameters**: 66.9 Million
- **Role in AtelierAI**: Discriminative text-classification baseline comparing fine-tuning against prompt-based LLMs.

## Training Details
- **Dataset**: Diverse natural language clothing prompt corpus with train/validation/test splits (70/15/15) designed without template leakage.
- **Classes**: 10 Fashion-MNIST apparel classes.
- **Optimizer**: AdamW (\(\alpha = 3 \times 10^{-5}\), weight decay = 0.01) with linear warmup schedule.
- **Epochs**: 4
- **Checkpoint Location**: `ml/checkpoints/distilbert_fashion/`
